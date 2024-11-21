LD_EXTRAS := -lobjc -framework Foundation -L. -lTest -lTestExt
OBJC_SOURCES := main.m

all: a.out

a.out: libTest.dylib libTestExt.dylib

include Makefile.rules

libTest.dylib:	Test/Test.m
	mkdir -p Test
	"$(MAKE)" MAKE_DSYM=YES -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_NAME=Test DYLIB_OBJC_SOURCES=Test/Test.m \
		LD_EXTRAS="-lobjc -framework Foundation" \
		CFLAGS_EXTRAS=-I$(SRCDIR)

libTestExt.dylib: TestExt/TestExt.m
	mkdir -p TestExt
	"$(MAKE)" MAKE_DSYM=YES -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_NAME=TestExt DYLIB_OBJC_SOURCES=TestExt/TestExt.m \
		LD_EXTRAS="-lobjc -framework Foundation -lTest -L." \
		CFLAGS_EXTRAS=-I$(SRCDIR)

