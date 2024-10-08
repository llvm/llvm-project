; This test verifies whether we can outline a singleton instance (i.e., an instance that does not repeat)
; using codegen data that has been read from a previous codegen run.

; RUN: split-file %s %t

; First, we generate the cgdata file from a local outline instance present in local-two.ll.
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-generate=true -filetype=obj %t/local-two.ll -o %t_write
; RUN: llvm-cgdata --merge %t_write -o %t_cgdata
; RUN: llvm-cgdata --show %t_cgdata | FileCheck %s --check-prefix=SHOW

; SHOW: Outlined hash tree:
; SHOW-NEXT:  Total Node Count: 4
; SHOW-NEXT:  Terminal Node Count: 1
; SHOW-NEXT:  Depth: 3

; Now, we read the cgdata in the machine outliner, enabling us to optimistically
; outline a singleton instance in local-one.ll that matches against the cgdata.
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-use-path=%t_cgdata -filetype=obj %t/local-one.ll -o %t_read
; RUN: llvm-objdump -d %t_read | FileCheck %s

; CHECK: _OUTLINED_FUNCTION
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

;--- local-one.ll
declare i32 @g(i32, i32, i32)
define i32 @f3() minsize {
  %1 = call i32 @g(i32 30, i32 1, i32 2);
 ret i32 %1
}
