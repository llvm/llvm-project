EXE := fat.out

ifdef FAT64_DSYM
	DSFLAGS_EXTRAS=-fat64
endif

include Makefile.rules

all: fat.out

fat.out: fat.x86_64h.out fat.x86_64.out
	lipo -fat64 -create -o $@ $^

fat.x86_64.out: fat.x86_64.o
	$(CC) -isysroot $(SDKROOT) -target x86_64-apple-macosx10.9 -o $@ $<

fat.x86_64h.out: fat.x86_64h.o
	$(CC) -isysroot $(SDKROOT) -target x86_64h-apple-macosx10.9 -o $@ $<

fat.x86_64.o: main.c
	$(CC) -isysroot $(SDKROOT) -g -O0 -target x86_64-apple-macosx10.9 -c -o $@ $<

fat.x86_64h.o: main.c
	$(CC) -isysroot $(SDKROOT) -g -O0 -target x86_64h-apple-macosx10.9 -c -o $@ $<
