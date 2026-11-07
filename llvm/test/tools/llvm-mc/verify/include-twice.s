## Directives in a file that is included twice are checked separately for each
## inclusion.

# REQUIRES: x86-registered-target
# RUN: rm -rf %t && split-file %s %t && cd %t
## FIXME: Both inclusions share one list of directives, so the second
## diagnostic finds the directive already used.
# RUN: not llvm-mc --triple=x86_64 --verify main.s 2>&1 \
# RUN:   | FileCheck %s --implicit-check-not=error:
# CHECK: inc.s:1:1: error: unexpected error: unknown directive

#--- main.s
.include "inc.s"
.include "inc.s"

#--- inc.s
.foo_directive # expected-error {{unknown directive}}
