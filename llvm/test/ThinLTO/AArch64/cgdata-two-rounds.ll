; This test verifies whether we can outline a singleton instance (i.e., an instance that does not repeat)
; by running two codegen rounds.

; RUN: split-file %s %t

; Verify each outlining instance is singleton with the global outlining for thinlto.
; They will be identical, which can be folded by the linker with ICF.
; RUN: opt -module-summary %t/thin-one.ll -o %t/thin-one.bc
; RUN: opt -module-summary %t/thin-two.ll -o %t/thin-two.bc
; RUN: llvm-lto2 run %t/thin-one.bc %t/thin-two.bc -o %t/thinlto \
; RUN:  -r %t/thin-one.bc,_f3,px -r %t/thin-one.bc,_g,x \
; RUN:  -r %t/thin-two.bc,_f1,px -r %t/thin-two.bc,_f2,px -r %t/thin-two.bc,_g,x \
; RUN:  -codegen-data-thinlto-two-rounds

; thin-one.ll will have one outlining instance (matched in the global outlined hash tree)
; RUN: llvm-objdump -d %t/thinlto.1 | FileCheck %s --check-prefix=THINLTO-1
; THINLTO-1: _OUTLINED_FUNCTION{{.*}}>:
; THINLTO-1-NEXT:  mov
; THINLTO-1-NEXT:  mov
; THINLTO-1-NEXT:  b

; thin-two.ll will have two outlining instances (matched in the global outlined hash tree)
; RUN: llvm-objdump -d %t/thinlto.2 | FileCheck %s --check-prefix=THINLTO-2
; THINLTO-2: _OUTLINED_FUNCTION{{.*}}>:
; THINLTO-2-NEXT:  mov
; THINLTO-2-NEXT:  mov
; THINLTO-2-NEXT:  b
; THINLTO-2: _OUTLINED_FUNCTION{{.*}}>:
; THINLTO-2-NEXT:  mov
; THINLTO-2-NEXT:  mov
; THINLTO-2-NEXT:  b

; Now add a lto module to the above thinlto modules.
; Verify the lto module is optimized independent of the global outlining for thinlto.
; RUN: opt %t/lto.ll -o %t/lto.bc
; RUN: llvm-lto2 run %t/thin-one.bc %t/thin-two.bc %t/lto.bc -o %t/out \
; RUN:  -r %t/thin-one.bc,_f3,px -r %t/thin-one.bc,_g,x \
; RUN:  -r %t/thin-two.bc,_f1,px -r %t/thin-two.bc,_f2,px -r %t/thin-two.bc,_g,x \
; RUN:  -r %t/lto.bc,_f4,px -r %t/lto.bc,_f5,px -r %t/lto.bc,_f6,px -r %t/lto.bc,_g,x \
; RUN:  -codegen-data-thinlto-two-rounds

; lto.ll will have one outlining instance within the lto module itself (no global outlining).
; RUN: llvm-objdump -d %t/out.0 | FileCheck %s --check-prefix=LTO-0
; LTO-0: _OUTLINED_FUNCTION{{.*}}>:
; LTO-0-NEXT:  mov
; LTO-0-NEXT:  b

; thin-one.ll will have one outlining instance (matched in the global outlined hash tree)
; RUN: llvm-objdump -d %t/out.1 | FileCheck %s --check-prefix=THINLTO-1

; thin-two.ll will have two outlining instances (matched in the global outlined hash tree)
; RUN: llvm-objdump -d %t/out.2 | FileCheck %s --check-prefix=THINLTO-2

;--- thin-one.ll
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-darwin"

declare i32 @g(i32, i32, i32)
define i32 @f3() minsize {
  %1 = call i32 @g(i32 30, i32 1, i32 2);
 ret i32 %1
}

;--- thin-two.ll
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-darwin"

declare i32 @g(i32, i32, i32)
define i32 @f1() minsize {
  %1 = call i32 @g(i32 10, i32 1, i32 2);
  ret i32 %1
}
define i32 @f2() minsize {
  %1 = call i32 @g(i32 20, i32 1, i32 2);
  ret i32 %1
}

;--- lto.ll
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-darwin"

declare i32 @g(i32, i32, i32)
define i32 @f4() minsize {
  %1 = call i32 @g(i32 10, i32 30, i32 2);
  ret i32 %1
}
define i32 @f5() minsize {
  %1 = call i32 @g(i32 20, i32 40, i32 2);
  ret i32 %1
}
define i32 @f6() minsize {
  %1 = call i32 @g(i32 50, i32 60, i32 2);
  ret i32 %1
}
