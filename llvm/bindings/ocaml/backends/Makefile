##===- bindings/ocaml/backends/Makefile --------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
#
# This is the master makefile for backend-specific bindings. It works by
# creating a stub makefile for each configured target, e.g. Makefile.ARM, and
# invoking it to compile the corresponding library, e.g. Llvm_ARM.
#
# This scheme allows to keep changes to Makefile.ocaml minimal.
#
##===----------------------------------------------------------------------===##

LEVEL := ../../..
ExtraMakefiles = $(PROJ_OBJ_DIR)/Makefile.common

include $(LEVEL)/Makefile.config
include $(LEVEL)/Makefile.common

all-local:: all-backends
clean-local:: clean-backends
install-local:: install-backends
uninstall-local:: uninstall-backends

stubs:
	$(Verb) for i in $(TARGETS_TO_BUILD); do \
		$(ECHO) "TARGET := $$i"           >  Makefile.$$i; \
		$(ECHO) "include Makefile.common" >> Makefile.$$i; \
	done

all-backends: stubs
	$(Verb) for i in $(TARGETS_TO_BUILD); do \
		$(MAKE) -f Makefile.$$i all; \
	done

clean-backends: stubs
	$(Verb) for i in $(TARGETS_TO_BUILD); do \
		$(MAKE) -f Makefile.$$i clean; \
		$(RM) -f Makefile.$$i; \
	done

install-backends: stubs
	$(Verb) for i in $(TARGETS_TO_BUILD); do \
		$(MAKE) -f Makefile.$$i install; \
	done

uninstall-backends: stubs
	$(Verb) for i in $(TARGETS_TO_BUILD); do \
		$(MAKE) -f Makefile.$$i uninstall; \
	done

ocamldoc: stubs
	$(Verb) for i in $(TARGETS_TO_BUILD); do \
		$(MAKE) -f Makefile.$$i ocamldoc; \
	done

.PHONY: all-backends clean-backends install-backends uninstall-backends ocamldoc
