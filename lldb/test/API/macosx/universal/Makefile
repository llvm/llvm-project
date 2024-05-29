EXE := testit

include Makefile.rules

all: testit

testit: testit.x86_64h testit.x86_64
	lipo -create -o testit $^

testit.x86_64h: testit.x86_64h.o
	$(CC) -isysroot $(SDKROOT) -target x86_64h-apple-macosx10.9 -o testit.x86_64h $(call path_wrapper,$<)

testit.x86_64: testit.x86_64.o
	$(CC) -isysroot $(SDKROOT) -target x86_64-apple-macosx10.9 -o testit.x86_64 $(call path_wrapper,$<)

testit.x86_64h.o: main.c
	$(CC) -isysroot $(SDKROOT) -g -O0 -target x86_64h-apple-macosx10.9 -c -o testit.x86_64h.o $(call path_wrapper,$<)

testit.x86_64.o: main.c
	$(CC) -isysroot $(SDKROOT) -g -O0 -target x86_64-apple-macosx10.9 -c -o testit.x86_64.o $(call path_wrapper,$<)
