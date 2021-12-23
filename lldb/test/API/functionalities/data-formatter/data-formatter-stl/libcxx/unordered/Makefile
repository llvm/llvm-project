CXX_SOURCES := main.cpp

# Work around "exception specification in declaration does not match previous
# declaration" errors present in older libc++ releases. This error was fixed in
# the 3.8 release.
CFLAGS_EXTRAS := -fno-exceptions

USE_LIBCPP := 1
include Makefile.rules
