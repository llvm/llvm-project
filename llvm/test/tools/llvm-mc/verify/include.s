## A directive in an included file (inc.s) is checked even if no diagnostics
## are reported in that file.

# REQUIRES: x86-registered-target
# RUN: rm -rf %t && split-file %s %t && cd %t
## FIXME: inc.s is never scanned since it has no diagnostics.
# RUN: llvm-mc --triple=x86_64 --verify main.s

#--- main.s
.include "inc.s"

#--- inc.s
nop
# expected-error@-1 {{this will never happen}}
