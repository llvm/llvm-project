##===- docs/tutorial/Makefile ------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL := ../..
include $(LEVEL)/Makefile.common

HTML       := $(wildcard $(PROJ_SRC_DIR)/*.html)
PNG        := $(wildcard $(PROJ_SRC_DIR)/*.png)
EXTRA_DIST := $(HTML) index.html
HTML_DIR   := $(DESTDIR)$(PROJ_docsdir)/html/tutorial

install-local:: $(HTML)
	$(Echo) Installing HTML Tutorial Documentation
	$(Verb) $(MKDIR) $(HTML_DIR)
	$(Verb) $(DataInstall) $(HTML) $(HTML_DIR)
	$(Verb) $(DataInstall) $(PNG) $(HTML_DIR)
	$(Verb) $(DataInstall) $(PROJ_SRC_DIR)/index.html $(HTML_DIR)

uninstall-local::
	$(Echo) Uninstalling Tutorial Documentation
	$(Verb) $(RM) -rf $(HTML_DIR)

printvars::
	$(Echo) "HTML           : " '$(HTML)'
