##===- utils/Makefile --------------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ..
PARALLEL_DIRS := FileCheck FileUpdate TableGen PerfectShuffle \
	      count fpcmp llvm-lit not unittest yaml2obj

EXTRA_DIST := check-each-file codegen-diff countloc.sh \
              DSAclean.py DSAextract.py emacs findsym.pl GenLibDeps.pl \
	      getsrcs.sh llvmdo llvmgrep llvm-native-gcc \
	      llvm-native-gxx makellvm profile.pl vim

include $(LEVEL)/Makefile.common
