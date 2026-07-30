; This test demonstrates how identical instruction sequences are handled during global outlining.
; Currently, we do not attempt to share an outlined function for identical sequences.
; Instead, each instruction sequence that matches against the global outlined hash tree
; is outlined into its own unique function.

; RUN: split-file %s %t

; First, we generate the cgdata file from a local outline instance present in local-two.ll.
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-generate=true -filetype=obj %t/local-two.ll -o %t_write
; RUN: llvm-cgdata --merge %t_write -o %t_cgdata
; RUN: llvm-cgdata --show %t_cgdata | FileCheck %s --check-prefix=SHOW

; SHOW: Outlined hash tree:
; SHOW-NEXT:  Total Node Count: 4
; SHOW-NEXT:  Terminal Node Count: 1
; SHOW-NEXT:  Depth: 3

; Now, we read the cgdata for local-two-another.ll and proceed to optimistically outline
; each instruction sequence that matches against the global outlined hash tree.
; Since each matching sequence is considered a candidate, we expect to generate two
; unique outlined functions. These functions, although unique, will be identical in code,
; and thus, will be folded by the linker.

; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-use-path=%t_cgdata -filetype=obj %t/local-two-another.ll -o %t_read
; RUN: llvm-objdump -d %t_read | FileCheck %s

; CHECK: _OUTLINED_FUNCTION_{{.*}}:
; CHECK-NEXT:  mov
; CHECK-NEXT:  mov
; CHECK-NEXT:  b

; CHECK: _OUTLINED_FUNCTION_{{.*}}:
; CHECK-NEXT:  mov
; CHECK-NEXT:  mov
; CHECK-NEXT:  b

;--- local-two.ll
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
declare i32 @g(i32, i32, i32)
define i32 @f3() minsize {
  %1 = call i32 @g(i32 30, i32 1, i32 2);
  ret i32 %1
}
define i32 @f4() minsize {
  %1 = call i32 @g(i32 40, i32 1, i32 2);
  ret i32 %1
}
