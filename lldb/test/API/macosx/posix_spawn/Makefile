EXE := fat.out
LIPO=lipo

include Makefile.rules

all: fat.out

x86_64.out: x86_64.c
	$(CC) -isysroot $(SDKROOT) -target x86_64-apple-macosx10.9 -o x86_64.out $<

x86_64h.out: x86_64h.c
	$(CC) -isysroot $(SDKROOT) -target x86_64h-apple-macosx10.9 -o x86_64h.out $<

arm64.out: arm64.c
	$(CC) -isysroot $(SDKROOT) -target arm64-apple-macosx10.9 -o arm64.out $<

fat.out: x86_64.out x86_64h.out arm64.out
	$(LIPO) -o fat.out -create $^
