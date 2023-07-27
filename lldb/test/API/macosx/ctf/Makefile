C_SOURCES := test.c
MAKE_DSYM := YES

ifeq "$(COMPRESS_CTF)" "YES"
	COMPRESS := -c
else
	 COMPRESS :=
endif

all: a.out a.ctf

include Makefile.rules

a.ctf: a.out.dSYM
	ctfconvert $(COMPRESS) \
		-l a \
		-o a.ctf \
		a.out.dSYM/Contents/Resources/DWARF/a.out
	$(OBJCOPY) \
		-R __DWARF,__debug_line \
		-R __DWARF,__debug_aranges \
		-R __DWARF,__debug_info \
		-R __DWARF,__debug_abbrev \
		-R __DWARF,__debug_str \
		-R __DWARF,__apple_names \
		-R __DWARF,__apple_namespac \
		-R __DWARF,__apple_types \
		-R __DWARF,__apple_objc \
		a.ctf a.ctf
	rm -rf a.out.dSYM
