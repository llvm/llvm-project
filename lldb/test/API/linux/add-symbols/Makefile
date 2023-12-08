CXX_SOURCES := main.cpp
LD_EXTRAS := -Wl,--build-id=none

all: stripped.out

stripped.out : a.out
	$(OBJCOPY) --remove-section=.note.gnu.build-id --remove-section=.gnu_debuglink --strip-debug $< $@

include Makefile.rules
