## A directive in an included file (inc.s) is checked even if no diagnostics
## are reported in that file.

# REQUIRES: x86-registered-target
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: not llvm-mc --triple=x86_64 --verify main.s 2>&1 \
# RUN:   | FileCheck inc.s --implicit-check-not=error:

#--- main.s
.include "inc.s"

#--- inc.s
nop
# expected-error@-1 {{this will never happen}}
# CHECK: inc.s:2:3: error: expected error "this will never happen" was not produced
