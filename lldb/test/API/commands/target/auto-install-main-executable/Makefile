CXX_SOURCES := main.cpp
CXXFLAGS := -DBUILD=\"stock\"

a.out: a.device.out

include Makefile.rules

a.device.out:
	$(CXX) $(CXXFLAGS) -DBUILD=\"device\" -o $@ $(SRCDIR)/main.cpp
