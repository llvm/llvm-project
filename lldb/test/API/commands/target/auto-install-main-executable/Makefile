CXX_SOURCES := main.cpp
CXXFLAGS := -DBUILD=47

a.out: a.device.out

include Makefile.rules

a.device.out:
	$(CXX) $(ASAN_LDFLAGS) $(CXXFLAGS) -DBUILD=74 -o $@ $(SRCDIR)/main.cpp
