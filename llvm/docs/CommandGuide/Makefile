##===- docs/CommandGuide/Makefile --------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

ifdef BUILD_FOR_WEBSITE

# This special case is for keeping the CommandGuide on the LLVM web site
# up to date automatically as the documents are checked in. It must build
# the POD files to HTML only and keep them in the src directories. It must also
# build in an unconfigured tree, hence the ifdef. To use this, run
# make -s BUILD_FOR_WEBSITE=1 inside the cvs commit script.

POD  := $(wildcard *.pod)
HTML := $(patsubst %.pod, html/%.html, $(POD))
MAN  := $(patsubst %.pod, man/man1/%.1, $(POD))
PS   := $(patsubst %.pod, ps/%.ps, $(POD))

all: $(HTML) $(MAN) $(PS)

.SUFFIXES:
.SUFFIXES: .html .pod .1 .ps

html/%.html: %.pod
	pod2html --css=manpage.css --htmlroot=. \
	  --podpath=. --noindex --infile=$< --outfile=$@ --title=$*

man/man1/%.1: %.pod
	pod2man --release=1.5 --center="LLVM Command Guide" $< $@

ps/%.ps: man/man1/%.1
	groff -Tps -man $< > $@

clean:
	rm -f pod2htm*.*~~ $(HTML) $(MAN) $(PS)

else 

LEVEL := ../..

include $(LEVEL)/Makefile.common

POD := $(wildcard $(PROJ_SRC_DIR)/*.pod)

EXTRA_DIST := $(POD) index.html

HTML = $(patsubst $(PROJ_SRC_DIR)/%.pod, $(PROJ_OBJ_DIR)/%.html, $(POD))
MAN = $(patsubst $(PROJ_SRC_DIR)/%.pod, $(PROJ_OBJ_DIR)/%.1, $(POD))
PS = $(patsubst $(PROJ_SRC_DIR)/%.pod, $(PROJ_OBJ_DIR)/%.ps, $(POD))

.SUFFIXES:
.SUFFIXES: .html .pod .1 .ps

$(HTML) : html/.dir man/.dir man/man1/.dir ps/.dir

html: $(HTML)

$(PROJ_OBJ_DIR)/%.html: %.pod
	$(POD2HTML) --css=manpage.css --htmlroot=. --podpath=. \
	  --noindex --infile=$< --outfile=$@ --title=$*

$(PROJ_OBJ_DIR)/%.1: %.pod
	$(POD2MAN) --release=$(LLVMVersion) \
	  --center="LLVM Command Guide" $< $@

$(PROJ_OBJ_DIR)/%.ps: $(PROJ_OBJ_DIR)/%.1
	$(GROFF) -Tps -man $< > $@

clean-local::
	$(Verb) $(RM) -f pod2htm*.*~~ $(HTML) $(MAN) $(PS)

HTML_DIR := $(PROJ_docsdir)/html/CommandGuide
MAN_DIR  := $(PROJ_mandir)/man1
PS_DIR   := $(PROJ_docsdir)/ps

install-local:: $(HTML) $(MAN) $(PS)
	$(Echo) Installing HTML CommandGuide Documentation
	$(Verb) $(MKDIR) $(HTML_DIR)
	$(Verb) $(DataInstall) $(HTML) $(HTML_DIR)
	$(Verb) $(DataInstall) $(PROJ_SRC_DIR)/index.html $(HTML_DIR)
	$(Verb) $(DataInstall) $(PROJ_SRC_DIR)/manpage.css $(HTML_DIR)
	$(Echo) Installing MAN CommandGuide Documentation
	$(Verb) $(MKDIR) $(MAN_DIR)
	$(Verb) $(DataInstall) $(MAN) $(MAN_DIR)
	$(Echo) Installing PS CommandGuide Documentation
	$(Verb) $(MKDIR) $(PS_DIR)
	$(Verb) $(DataInstall) $(PS) $(PS_DIR)

uninstall-local::
	$(Echo) Uninstalling Documentation
	$(Verb) $(RM) -rf $(LLVM_DOCSDIR)

printvars::
	$(Echo) "POD            : " '$(POD)'
	$(Echo) "HTML           : " '$(HTML)'

endif
