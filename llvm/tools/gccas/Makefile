##===- tools/gccas/Makefile --------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../..

include $(LEVEL)/Makefile.common

install-local:: $(PROJ_bindir)/gccas
	
$(PROJ_bindir)/gccas : gccas.sh Makefile
	$(Echo) Installing gccas shell script.
	$(Verb) sed "s#@TOOLDIR@#$(PROJ_bindir)#" $< > $@
	$(Verb) chmod 0755 $@

all-local:: $(ToolDir)/gccas
	
$(ToolDir)/gccas : gccas.sh Makefile
	$(Echo) Making $(ToolDir)/gccas shell script.
	$(Verb) sed "s#@TOOLDIR@#$(ToolDir)#" $< > $@
	$(Verb) chmod 0755 $@

clean-local::
	$(Verb)$(RM) -f $(ToolDir)/gccas
