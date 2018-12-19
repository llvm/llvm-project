#===-- macos.inc ---------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.TXT for details.
#
#===----------------------------------------------------------------------===##

DYN_LDFLAGS += -Wl,-rpath,$(TBBROOT)/lib
