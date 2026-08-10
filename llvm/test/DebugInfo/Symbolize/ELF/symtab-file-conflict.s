# REQUIRES: x86-registered-target
## If the filename from the preceding STT_FILE does not match .debug_line,
## STT_FILE wins. Compilers should not emit such bogus information.
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-symbolizer --obj=%t 0 | FileCheck %s

# CHECK:       foo
# CHECK-NEXT:  1.c:0:0

.file "1.c"
.file 0 "/tmp" "0.c"
foo:
  .loc 0 1 0
  nop
