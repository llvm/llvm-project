##===- runtime/libdummy/Makefile ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../..
BYTECODE_LIBRARY=1
BUILD_ARCHIVE=1
DONT_BUILD_RELINKED=1
LIBRARYNAME=z
CPPFLAGS += -DNO_DUMMY_DECL=1

Source = adler32.c compress.c crc32.c gzio.c uncompr.c deflate.c trees.c \
         zutil.c inflate.c infback.c inftrees.c inffast.c

include $(LEVEL)/Makefile.common
