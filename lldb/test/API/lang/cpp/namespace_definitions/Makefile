LD_EXTRAS := -L. -la -lb
CXX_SOURCES := main.cpp

a.out: liba libb

include Makefile.rules

liba:
	$(MAKE) -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_NAME=a DYLIB_CXX_SOURCES=a.cpp

libb:
	$(MAKE) -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_NAME=b DYLIB_CXX_SOURCES=b.cpp

