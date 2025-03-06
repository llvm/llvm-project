C_SOURCES := main.c
LD_EXTRAS = -Wl,-rpath "-Wl,$(shell pwd)" -L. -lno-nlists -lhas-nlists

.PHONY: build-libno-nlists build-libhas-nlists
all: build-libno-nlists build-libhas-nlists a.out

include Makefile.rules

build-libno-nlists: no-nlists.c no-nlist-sect.s
	$(CC) $(CFLAGS) -c -o no-nlists.o $(<D)/no-nlists.c
	$(CC) $(CFLAGS) -c -o no-nlist-sect.o $(<D)/no-nlist-sect.s
	$(LD) -dynamiclib -o libno-nlists.dylib no-nlists.o no-nlist-sect.o -install_name "@executable_path/libno-nlists.dylib"

build-libhas-nlists: has-nlists.c
	$(CC) $(CFLAGS) -c -o has-nlists.o $(<D)/has-nlists.c
	$(LD) -dynamiclib -o libhas-nlists.dylib has-nlists.o  -install_name "@executable_path/libhas-nlists.dylib"

clean::
	rm -rf *.o *.dylib a.out *.dSYM
