DYLIB_NAME := unlikely_name
DYLIB_CXX_SOURCES := foo.cpp
CXX_SOURCES := main.cpp
USE_LIBDL := 1

dylib.so: dylib.c
	$(MAKE) -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_NAME=dylib DYLIB_C_SOURCES=dylib.c

dylib_loader: dylib_loader.c
	$(CC) $(CFLAGS) $(LDFLAGS) -Wl,-rpath "-Wl,$(shell pwd)" -o $@ $^

all: dylib.so dylib_loader

include Makefile.rules
