# We don't have any standard include directories, so we can't
# parse the test_common.h header we usually inject as it includes
# system headers.
NO_TEST_COMMON_H := 1

# Take the libc++ from the build directory (which will be later deleted).
CXXFLAGS_EXTRAS = -I $(BUILDDIR)/root/usr/include/c++/v1/ -I $(BUILDDIR)/root/usr/include/ -nostdinc -nostdinc++
CXX_SOURCES := main.cpp

include Makefile.rules
