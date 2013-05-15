all: a.out liba.dylib libb.dylib libc.dylib libd.dylib

CC ?= clang
ifeq "$(ARCH)" ""
	ARCH = x86_64
endif
CFLAGS ?=-arch $(ARCH) -g -O0
CWD := $(shell pwd)

all: a.out hidden/libd.dylib

a.out: main.o libd.dylib
	$(CC) $(CFLAGS) -o a.out main.o -L. -ld

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

hidden/libd.dylib: b.o
	$(CC) $(CFLAGS) -dynamiclib -o hidden/libd.dylib d.o
	dsymutil -o hidden/libd.dylib.dSYM hidden/libd.dylib

liba.dylib: a.o libb.dylib
	$(CC) $(CFLAGS) -dynamiclib -install_name "@executable_path/liba.dylib" -o liba.dylib a.o -L. -lb
	dsymutil liba.dylib

a.o: a.c
	$(CC) $(CFLAGS) -c a.c

libb.dylib: b.o
	$(CC) $(CFLAGS) -dynamiclib -install_name "@executable_path/libb.dylib" -o libb.dylib b.o
	dsymutil libb.dylib

b.o: b.c
	$(CC) $(CFLAGS) -c b.c

libc.dylib: c.o
	$(CC) $(CFLAGS) -dynamiclib -install_name "@executable_path/libc.dylib" -o libc.dylib c.o
	dsymutil libc.dylib

c.o: c.c
	$(CC) $(CFLAGS) -c c.c

libd.dylib: d.o
	$(CC) $(CFLAGS) -dynamiclib -install_name "$(CWD)/libd.dylib" -o libd.dylib d.o
	dsymutil libd.dylib

d.o: d.c
	$(CC) $(CFLAGS) -c d.c

clean:
	rm -rf *.o *~ *.dylib a.out *.dSYM hidden/*
