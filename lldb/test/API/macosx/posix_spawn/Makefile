EXE := fat.out
LIPO=lipo

include Makefile.rules

SLICES ?= x86_64.out x86_64h.out arm64.out

all: fat.out

x86_64.out: x86_64.c
	$(CC) $(ASAN_LDFLAGS) -isysroot $(SDKROOT) -target x86_64-apple-macosx10.9 -o x86_64.out $<

x86_64h.out: x86_64h.c
	$(CC) $(ASAN_LDFLAGS) -isysroot $(SDKROOT) -target x86_64h-apple-macosx10.9 -o x86_64h.out $<

arm64.out: arm64.c
	$(CC) $(ASAN_LDFLAGS) -isysroot $(SDKROOT) -target arm64-apple-macosx10.9 -o arm64.out $<

arm64e.out: arm64e.c
	$(CC) $(ASAN_LDFLAGS) -isysroot $(SDKROOT) -target arm64e-apple-macosx11.0 -o arm64e.out $<

fat.out: $(SLICES)
	$(LIPO) -o fat.out -create $^
