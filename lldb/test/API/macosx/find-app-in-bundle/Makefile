C_SOURCES := main.c
EXE := TestApp
all: clean TestApp.app/Contents/MacOS/TestApp

include Makefile.rules

TestApp.app/Contents/MacOS/TestApp: $(EXE) $(DSYM)
	cp -r $(SRCDIR)/TestApp.app .
	mv TestApp TestApp.app/Contents/MacOS/TestApp
	mv TestApp.dSYM TestApp.app.dSYM

clean::
	rm -rf $(EXE) TestApp.app TestApp.app.dSYM
