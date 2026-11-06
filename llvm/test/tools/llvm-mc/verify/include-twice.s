## Directives in a file that is included twice are checked separately for each
## inclusion.

# REQUIRES: x86-registered-target
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc --triple=x86_64 --verify main.s

#--- main.s
.include "inc.s"
.include "inc.s"

#--- inc.s
.foo_directive # expected-error {{unknown directive}}
