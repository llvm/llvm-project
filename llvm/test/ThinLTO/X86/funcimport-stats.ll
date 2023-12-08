; REQUIRES: asserts
; RUN: opt -module-summary %p/funcimport.ll -o %t.bc
; RUN: opt -module-summary %p/Inputs/funcimport.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t3.bc %t.bc %t2.bc

; RUN: llvm-lto -thinlto-action=import %t2.bc -thinlto-index=%t3.bc -o /dev/null -debug-only=function-import -stats > %t4 2>&1
; RUN: cat %t4 | grep 'Is importing global' | count 4
; RUN: cat %t4 | grep 'Is importing function' | count 8
; RUN: cat %t4 | grep 'Is importing aliasee' | count 1
; RUN: cat %t4 | FileCheck %s

; CHECK:      - [[NUM_FUNCS:[0-9]+]] functions imported from
; CHECK-NEXT: - [[NUM_VARS:[0-9]+]] global vars imported from

; CHECK:      [[NUM_FUNCS]] function-import - Number of functions imported in backend
; CHECK-NEXT: [[NUM_FUNCS]] function-import - Number of functions thin link decided to import
; CHECK-NEXT: [[NUM_VARS]] function-import - Number of global variables imported in backend
; CHECK-NEXT: [[NUM_VARS]] function-import - Number of global variables thin link decided to import
; CHECK-NEXT: 1 function-import - Number of modules imported from
; CHECK-NEXT: [[NUM_VARS]] module-summary-index - Number of live global variables marked read only
; CHECK-NEXT: 1 module-summary-index - Number of live global variables marked write only
