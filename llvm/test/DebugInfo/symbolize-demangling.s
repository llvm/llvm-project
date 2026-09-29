# REQUIRES: x86-registered-target

# RUN: llvm-mc --filetype=obj --triple=x86_64-pc-linux %s -o %t.o -g

# RUN: llvm-symbolizer --obj=%t.o 0 1 2 3 | FileCheck %s

# CHECK:       f()
# CHECK-NEXT:  symbolize-demangling.s:24
# CHECK-EMPTY:
# CHECK-NEXT:  {{^g$}}
# CHECK-NEXT:  symbolize-demangling.s:26
# CHECK-EMPTY:
# CHECK-NEXT:  {{^baz$}}
# CHECK-NEXT:  symbolize-demangling.s:28
# CHECK-EMPTY:
# CHECK-NEXT:  {{^x::y$}}
# CHECK-NEXT:  symbolize-demangling.s:30

.type _Z1fv,@function
.type g,@function
.type baz,@function
.type _RNvC1x1y,@function
_Z1fv:
  nop
g:
  nop
baz:
  nop
_RNvC1x1y:
  nop
