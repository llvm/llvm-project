LIB_PREFIX := svr4lib
LD_EXTRAS := -L. -lsvr4lib_a -lsvr4lib_b\"
CXX_SOURCES := main.cpp
USE_LIBDL := 1
MAKE_DSYM := NO

a.out: svr4lib_a svr4lib_b_quote

include Makefile.rules

svr4lib_a:
	$(MAKE) -f $(MAKEFILE_RULES) \
		DYLIB_NAME=svr4lib_a DYLIB_CXX_SOURCES=svr4lib_a.cpp \
		DYLIB_ONLY=YES

svr4lib_b_quote:
	$(MAKE) -f $(MAKEFILE_RULES) \
		DYLIB_NAME=svr4lib_b\\\" DYLIB_CXX_SOURCES=svr4lib_b_quote.cpp \
		DYLIB_ONLY=YES
