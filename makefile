# OBJS = MovieList.o Movie.o NameList.o Name.o Iterator.o
CC = g++
DEBUG = -ggdb
BASENAME = `basename liposomes.cpp .cpp`
CFLAGS = -Wall `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`
LFLAGS = -Wall

liposomes : liposomes.cpp
	$(CC) $(DEBUG) $(CFLAGS) -o $(BASENAME) liposomes.cpp $(LIBS)

# MovieList.o : MovieList.h MovieList.cpp Movie.h NameList.h Name.h Iterator.h
# 	$(CC) $(CFLAGS) MovieList.cpp

clean:
	\rm *.o *~ liposomes