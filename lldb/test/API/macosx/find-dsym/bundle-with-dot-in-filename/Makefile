C_SOURCES := main.c
EXE := find-bundle-with-dots-in-fn

include Makefile.rules

$(EXE): hide.app

hide.app: bundle.c
	$(CC) $(ASAN_LDFLAGS) $(CFLAGS) -dynamiclib -o com.apple.sbd $<
	mkdir com.apple.sbd.xpc
	mv com.apple.sbd com.apple.sbd.xpc/
	mkdir -p com.apple.sbd.xpc.dSYM/Contents/Resources/DWARF
	mv com.apple.sbd.dSYM/Contents/Resources/DWARF/com.apple.sbd com.apple.sbd.xpc.dSYM/Contents/Resources/DWARF/
	rm -rf com.apple.sbd.dSYM
	mkdir hide.app
	tar cf - com.apple.sbd.xpc com.apple.sbd.xpc.dSYM | ( cd hide.app;tar xBpf -)
