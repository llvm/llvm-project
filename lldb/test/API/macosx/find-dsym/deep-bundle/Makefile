EXE := deep-bundle
all: clean $(EXE)

include Makefile.rules

$(EXE):
	$(CC) $(CFLAGS) -install_name $(shell pwd)/MyFramework.framework/Versions/A/MyFramework -dynamiclib -o MyFramework $(SRCDIR)/myframework.c
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
	$(CC) $(CFLAGS) -o deep-bundle $(SRCDIR)/main.c -F. -framework MyFramework

clean::
	rm -rf a.out a.out.dSYM deep-bundle deep-bundle.dSYM MyFramework.framework MyFramework.framework.dSYM MyFramework MyFramework.dSYM hide.app
