##===- tools/gccld/Makefile --------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..

include $(LEVEL)/Makefile.common

install-local:: $(PROJ_bindir)/gccld
	
$(PROJ_bindir)/gccld : gccld.sh Makefile
	$(Echo) Installing gccld shell script.
	$(Verb) sed "s#@TOOLDIR@#$(PROJ_bindir)#" $< > $@
	$(Verb) chmod 0755 $@

all-local:: $(ToolDir)/gccld
	
$(ToolDir)/gccld : gccld.sh Makefile
	$(Echo) Making $(ToolDir)/gccld shell script.
	$(Verb) sed "s#@TOOLDIR@#$(ToolDir)#" $< > $@
	$(Verb) chmod 0755 $@

clean-local::
	$(Verb)$(RM) -f $(ToolDir)/gccld
