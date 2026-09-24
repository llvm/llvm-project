C_SOURCES := main.c
EXE := deep-bundle
CFLAGS_EXTRAS := -F.
LD_EXTRAS := -framework MyFramework

include Makefile.rules

main.o: hide.app

hide.app: myframework.c MyFramework.h Info.plist
	$(CC) $(ASAN_LDFLAGS) $(CFLAGS) -install_name $(shell pwd)/MyFramework.framework/Versions/A/MyFramework -dynamiclib -o MyFramework $<
	mkdir -p MyFramework.framework/Versions/A/Headers
	mkdir -p MyFramework.framework/Versions/A/Resources
	cp MyFramework MyFramework.framework/Versions/A
	cp $(SRCDIR)/MyFramework.h MyFramework.framework/Versions/A/Headers
	cp $(SRCDIR)/Info.plist MyFramework.framework/Versions/A/Resources
	( cd MyFramework.framework/Versions ; ln -s A Current )
	( cd MyFramework.framework/ ; ln -s Versions/Current/Headers . )
	( cd MyFramework.framework/ ; ln -s Versions/Current/MyFramework . )
	( cd MyFramework.framework/ ; ln -s Versions/Current/Resources . )
	mv MyFramework.dSYM MyFramework.framework.dSYM
	mkdir hide.app
	rm -f MyFramework
	tar cf - MyFramework.framework MyFramework.framework.dSYM | ( cd hide.app;tar xBpf -)
