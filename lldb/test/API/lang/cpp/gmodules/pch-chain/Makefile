include Makefile.rules

OBJECTS += main.o

$(EXE): $(BUILDDIR)/main.o

$(BUILDDIR)/main.o: main.cpp
	$(CC) -cc1 -emit-pch -x c++-header -fmodule-format=obj -fmodules -O0 -dwarf-ext-refs -debug-info-kind=standalone $(SRCDIR)/base-pch.h -o base-pch.h.gch
	$(CC) -cc1 -emit-pch -x c++-header -fmodule-format=obj -fmodules -O0 -dwarf-ext-refs -debug-info-kind=standalone -include-pch base-pch.h.gch $(SRCDIR)/pch.h -o pch.h.gch
	$(CC) -cc1 -emit-obj -x c++ -fmodules -O0 -dwarf-ext-refs -debug-info-kind=standalone -include-pch pch.h.gch $(SRCDIR)/main.cpp -o $(BUILDDIR)/main.o
