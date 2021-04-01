CXX_SOURCES := main.cpp
CXXFLAGS := -DBUILD=47

a.out: a.device.out

include Makefile.rules

a.device.out:
	$(CXX) $(CXXFLAGS) -DBUILD=74 -o $@ $(SRCDIR)/main.cpp
