; This test is similar to cgdata-read-double-outline.ll, but it is executed with LTO (Link Time Optimization).
; It demonstrates how identical instruction sequences are handled during global outlining.
; Currently, we do not attempt to reuse an outlined function for identical sequences.
; Instead, each instruction sequence that appears in the global outlined hash tree
; is outlined into its own unique function.

; RUN: split-file %s %t

; We first create the cgdata file from a local outline instance in local-two.ll
; RUN: opt -module-summary %t/local-two.ll -o %t/write.bc
; RUN: llvm-lto2 run %t/write.bc -o %t/write \
; RUN:  -r %t/write.bc,_f1,px -r %t/write.bc,_f2,px -r %t/write.bc,_g,p \
; RUN:  -codegen-data-generate=true
; RUN: llvm-cgdata --merge %t/write.1 -o %t_cgdata
; RUN: llvm-cgdata --show %t_cgdata | FileCheck %s --check-prefix=SHOW

; SHOW: Outlined hash tree:
; SHOW-NEXT:  Total Node Count: 4
; SHOW-NEXT:  Terminal Node Count: 1
; SHOW-NEXT:  Depth: 3

; Now, we execute either ThinLTO or LTO by reading the cgdata for local-two-another.ll.
; With ThinLTO, similar to the no-LTO scenario shown in cgdata-read-double-outline.ll,
; it optimistically outlines each instruction sequence that matches against
; the global outlined hash tree. Since each matching sequence is considered a candidate,
; we expect to generate two unique outlined functions that will be folded
; by the linker at a later stage.
; However, with LTO, we do not utilize the cgdata, but instead fall back to the default
; outliner mode. This results in a single outlined function that is
; shared across two call-sites.

; Run ThinLTO
; RUN: opt -module-summary %t/local-two-another.ll -o %t/thinlto.bc
; RUN: llvm-lto2 run %t/thinlto.bc -o %t/thinlto \
; RUN:  -r %t/thinlto.bc,_f3,px -r %t/thinlto.bc,_f4,px -r %t/thinlto.bc,_g,p \
; RUN:  -codegen-data-use-path=%t_cgdata
; RUN: llvm-objdump -d %t/thinlto.1 | FileCheck %s

; CHECK: _OUTLINED_FUNCTION_{{.*}}:
; CHECK-NEXT:  mov
; CHECK-NEXT:  mov
; CHECK-NEXT:  b
; CHECK: _OUTLINED_FUNCTION_{{.*}}:
; CHECK-NEXT:  mov
; CHECK-NEXT:  mov
; CHECK-NEXT:  b

; Run ThinLTO while disabling the global outliner.
; We have a single outlined case with the default outliner.
; RUN: llvm-lto2 run %t/thinlto.bc -o %t/thinlto-disable \
; RUN:  -r %t/thinlto.bc,_f3,px -r %t/thinlto.bc,_f4,px -r %t/thinlto.bc,_g,p \
; RUN:  -enable-machine-outliner \
; RUN:  -codegen-data-use-path=%t_cgdata \
; RUN:  -disable-global-outlining
; RUN: llvm-objdump -d %t/thinlto-disable.1 | FileCheck %s --check-prefix=DISABLE

; DISABLE: _OUTLINED_FUNCTION_{{.*}}:
; DISABLE-NEXT:  mov
; DISABLE-NEXT:  mov
; DISABLE-NEXT:  b
; DISABLE-NOT: _OUTLINED_FUNCTION_{{.*}}:

; Run LTO, which effectively disables the global outliner.
; RUN: opt %t/local-two-another.ll -o %t/lto.bc
; RUN: llvm-lto2 run %t/lto.bc -o %t/lto \
; RUN:  -r %t/lto.bc,_f3,px -r %t/lto.bc,_f4,px -r %t/lto.bc,_g,p \
; RUN:  -enable-machine-outliner \
; RUN:  -codegen-data-use-path=%t_cgdata
; RUN: llvm-objdump -d %t/lto.0 | FileCheck %s --check-prefix=DISABLE

;--- local-two.ll
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

;--- local-two-another.ll
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-darwin"

declare i32 @g(i32, i32, i32)
define i32 @f3() minsize {
  %1 = call i32 @g(i32 30, i32 1, i32 2);
  ret i32 %1
}
define i32 @f4() minsize {
  %1 = call i32 @g(i32 40, i32 1, i32 2);
  ret i32 %1
}
