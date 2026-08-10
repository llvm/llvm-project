# REQUIRES: aarch64-registered-target
## Test that we can copy LC_LINKER_OPTIMIZATION_HINT.

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: llvm-objdump --macho --link-opt-hints - < %t.o > %tloh.txt
# RUN: FileCheck --input-file=%tloh.txt %s

# CHECK:      Linker optimiztion hints (8 total bytes)
# CHECK-NEXT:   identifier 7 AdrpAdd

# RUN: llvm-objcopy %t.o %t.copy.o
# RUN: llvm-objdump --macho --link-opt-hints - < %t.copy.o | diff %tloh.txt -

.text
.align 2
_test:
L1:
  adrp  x0, _foo@PAGE
L2:
  add   x0, x0, _foo@PAGEOFF
.loh AdrpAdd L1, L2

.data
_foo:
  .long 0
