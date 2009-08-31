##===- docs/CommandGuide/Makefile --------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

ifdef BUILD_FOR_WEBSITE
# This special case is for keeping the CommandGuide on the LLVM web site
# up to date automatically as the documents are checked in. It must build
# the POD files to HTML only and keep them in the src directories. It must also
# build in an unconfigured tree, hence the ifdef. To use this, run
# make -s BUILD_FOR_WEBSITE=1 inside the cvs commit script.
SRC_DOC_DIR=
DST_HTML_DIR=html/
DST_MAN_DIR=man/man1/
DST_PS_DIR=ps/

# If we are in BUILD_FOR_WEBSITE mode, default to the all target.
all:: html man ps

clean:
	rm -f pod2htm*.*~~ $(HTML) $(MAN) $(PS)

# To create other directories, as needed, and timestamp their creation
%/.dir:
	-mkdir $* > /dev/null
	date > $@

else

# Otherwise, if not in BUILD_FOR_WEBSITE mode, use the project info.
LEVEL := ../..
include $(LEVEL)/Makefile.common

SRC_DOC_DIR=$(PROJ_SRC_DIR)/
DST_HTML_DIR=$(PROJ_OBJ_DIR)/
DST_MAN_DIR=$(PROJ_OBJ_DIR)/
DST_PS_DIR=$(PROJ_OBJ_DIR)/

endif


POD  := $(wildcard $(SRC_DOC_DIR)*.pod)
HTML := $(patsubst $(SRC_DOC_DIR)%.pod, $(DST_HTML_DIR)%.html, $(POD))
MAN  := $(patsubst $(SRC_DOC_DIR)%.pod, $(DST_MAN_DIR)%.1, $(POD))
PS   := $(patsubst $(SRC_DOC_DIR)%.pod, $(DST_PS_DIR)%.ps, $(POD))

# The set of man pages we will not install
NO_INSTALL_MANS = $(DST_MAN_DIR)FileCheck.1

# The set of man pages that we will install
INSTALL_MANS = $(filter-out $(NO_INSTALL_MANS), $(MAN))

.SUFFIXES:
.SUFFIXES: .html .pod .1 .ps

$(DST_HTML_DIR)%.html: %.pod $(DST_HTML_DIR)/.dir
	pod2html --css=manpage.css --htmlroot=. \
	  --podpath=. --noindex --infile=$< --outfile=$@ --title=$*

$(DST_MAN_DIR)%.1: %.pod $(DST_MAN_DIR)/.dir
	pod2man --release=CVS --center="LLVM Command Guide" $< $@

$(DST_PS_DIR)%.ps: $(DST_MAN_DIR)%.1 $(DST_PS_DIR)/.dir
	groff -Tps -man $< > $@


html: $(HTML)
man: $(MAN)
ps: $(PS)

EXTRA_DIST := $(POD) index.html

clean-local::
	$(Verb) $(RM) -f pod2htm*.*~~ $(HTML) $(MAN) $(PS)

HTML_DIR := $(PROJ_docsdir)/html/CommandGuide
MAN_DIR  := $(PROJ_mandir)/man1
PS_DIR   := $(PROJ_docsdir)/ps

install-local:: $(HTML) $(INSTALL_MANS) $(PS)
	$(Echo) Installing HTML CommandGuide Documentation
	$(Verb) $(MKDIR) $(HTML_DIR)
	$(Verb) $(DataInstall) $(HTML) $(HTML_DIR)
	$(Verb) $(DataInstall) $(PROJ_SRC_DIR)/index.html $(HTML_DIR)
	$(Verb) $(DataInstall) $(PROJ_SRC_DIR)/manpage.css $(HTML_DIR)
	$(Echo) Installing MAN CommandGuide Documentation
	$(Verb) $(MKDIR) $(MAN_DIR)
	$(Verb) $(DataInstall) $(INSTALL_MANS) $(MAN_DIR)
	$(Echo) Installing PS CommandGuide Documentation
	$(Verb) $(MKDIR) $(PS_DIR)
	$(Verb) $(DataInstall) $(PS) $(PS_DIR)

uninstall-local::
	$(Echo) Uninstalling CommandGuide Documentation
	$(Verb) $(RM) -rf $(HTML_DIR) $(MAN_DIR) $(PS_DIR)

printvars::
	$(Echo) "POD            : " '$(POD)'
	$(Echo) "HTML           : " '$(HTML)'
