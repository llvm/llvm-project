CXX = clang++
CXXFLAGS = -g -O0 -std=c++17 -pthread

all: simple complex

simple: simple.cpp
	$(CXX) $(CXXFLAGS) -o simple simple.cpp

complex: complex.cpp
	$(CXX) $(CXXFLAGS) -o complex complex.cpp

clean:
	rm -f simple complex

.PHONY: all clean
