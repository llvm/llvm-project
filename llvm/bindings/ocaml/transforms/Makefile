##===- bindings/ocaml/transforms/Makefile ------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL := ../../..
DIRS = scalar ipo

ocamldoc:
	$(Verb) for i in $(DIRS) ; do \
		$(MAKE) -C $$i ocamldoc; \
	done

include $(LEVEL)/Makefile.common
