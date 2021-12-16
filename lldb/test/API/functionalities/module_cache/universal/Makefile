EXE := testit

include Makefile.rules

all: testit

testit: testit.x86_64 testit.arm64
	lipo -create -o testit $^

testit.arm64: testit.arm64.o
	$(CC) -isysroot $(SDKROOT) -target arm64-apple-macosx10.9 -o testit.arm64 $<

testit.x86_64: testit.x86_64.o
	$(CC) -isysroot $(SDKROOT) -target x86_64-apple-macosx10.9 -o testit.x86_64 $<

testit.arm64.o: main.c
	$(CC) -isysroot $(SDKROOT) -g -O0 -target arm64-apple-macosx10.9 -c -o testit.arm64.o $<

testit.x86_64.o: main.c
	$(CC) -isysroot $(SDKROOT) -g -O0 -target x86_64-apple-macosx10.9 -c -o testit.x86_64.o $<
