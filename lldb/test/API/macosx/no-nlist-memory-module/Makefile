C_SOURCES := main.c
LD_EXTRAS = -Wl,-rpath "-Wl,$(shell pwd)" -L. -lno-nlists

.PHONY: build-libno-nlists
all: build-libno-nlists a.out

include Makefile.rules

build-libno-nlists: no-nlists.c no-nlist-sect.s
	$(CC) $(CFLAGS) -c -o no-nlists.o $(<D)/no-nlists.c
	$(CC) $(CFLAGS) -c -o no-nlist-sect.o $(<D)/no-nlist-sect.s
	$(LD) -dynamiclib -o libno-nlists.dylib no-nlists.o no-nlist-sect.o -install_name "@executable_path/libno-nlists.dylib"

clean::
	rm -rf no-nlists.o no-nlist-sect.o main.o main.s a.out a.out.dSYM libno-nlists.dylib libno-nlists.dylib.dSYM
