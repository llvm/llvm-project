C_SOURCES := main.c

all: dirsymlink

dirreal: a.out
	$(call RM_RF,$@)
	mkdir $@
	$(OBJCOPY) --only-keep-debug $< $@/stripped.debug
	$(OBJCOPY) --strip-all --add-gnu-debuglink=$@/stripped.debug $< $@/stripped.out

dirsymlink: dirreal
	$(call RM_RF,$@)
	mkdir $@
	ln -s ../$</stripped.out $@/stripped.symlink

include Makefile.rules
