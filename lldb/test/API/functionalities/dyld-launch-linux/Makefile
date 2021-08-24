CXX_SOURCES := main.cpp
LD_EXTRAS := -Wl,-rpath "-Wl,$(shell pwd)"
USE_LIBDL := 1

include Makefile.rules

# The following shared library will be used to test breakpoints under dynamic loading
libsignal_file.so:  signal_file.cpp
	$(MAKE) -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_CXX_SOURCES=signal_file.cpp DYLIB_NAME=signal_file

a.out: libsignal_file.so main.cpp
	$(MAKE) -f $(MAKEFILE_RULES) \
                CXX_SOURCES=main.cpp LD_EXTRAS=libsignal_file.so

