# We don't have any standard include directories, so we can't
# parse the test_common.h header we usually inject as it includes
# system headers.
NO_TEST_COMMON_H := 1

CXXFLAGS_EXTRAS = -I $(SRCDIR)/root/usr/include/c++/v1/ -I $(SRCDIR)/root/usr/include/ -nostdinc -nostdinc++ -DBUILDING_OUTSIDE_LLDB=1
CXX_SOURCES := main.cpp

include Makefile.rules
