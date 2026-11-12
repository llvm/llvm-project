; This test verifies whether we can outline a singleton instance (i.e., an instance that does not repeat)
; using codegen data that has been read from a previous codegen run.
; When multiple matches occur, we prioritize the candidates using the global frequency.

; RUN: split-file %s %t

; First, we generate the cgdata file from local outline instances present in write1.ll and write2.ll
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-generate=true -filetype=obj %t/write1.ll -o %t_write1
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-generate=true -filetype=obj %t/write2.ll -o %t_write2
; RUN: llvm-cgdata --merge %t_write1 %t_write2 -o %t_cgdata
; RUN: llvm-cgdata --show %t_cgdata | FileCheck %s --check-prefix=SHOW

; SHOW: Outlined hash tree:
; SHOW-NEXT:  Total Node Count: 8
; SHOW-NEXT:  Terminal Node Count: 2
; SHOW-NEXT:  Depth: 4

; Now, we read the cgdata in the machine outliner, enabling us to optimistically
; outline a singleton instance in read.ll that matches against the cgdata.
; There are two matches -- (1) (mov #1, mov #2, mov #3, b) and (2) (mov #2, mov #3, b).
; Even though sequence (1) is longer than sequence (2), the latter is outlined because it occurs more frequently in the outlined hash tree.

; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-use-path=%t_cgdata -filetype=obj %t/read.ll -o %t_read
; RUN: llvm-objdump -d %t_read | FileCheck %s

; CHECK: _OUTLINED_FUNCTION
; CHECK-NEXT:  mov
; CHECK-NEXT:  mov
; CHECK-NEXT:  b

;--- write1.ll
; The sequence (mov #2, mov #3, b) are repeated 4 times.
declare i32 @g(i32, i32, i32)
define i32 @f1() minsize {
  %1 = call i32 @g(i32 10, i32 50, i32 2, i32 3);
  ret i32 %1
}
define i32 @f2() minsize {
  %1 = call i32 @g(i32 20, i32 60, i32 2, i32 3);
  ret i32 %1
}
define i32 @f3() minsize {
  %1 = call i32 @g(i32 30, i32 70, i32 2, i32 3);
  ret i32 %1
}
define i32 @f4() minsize {
  %1 = call i32 @g(i32 40, i32 80, i32 2, i32 3);
  ret i32 %1
}

;--- write2.ll
; The sequence (mov #1, mov #2, mov #3, b) are repeated 2 times.
declare i32 @g(i32, i32, i32)
define i32 @f6() minsize {
  %1 = call i32 @g(i32 10, i32 1, i32 2, i32 3);
  ret i32 %1
}
define i32 @f7() minsize {
  %1 = call i32 @g(i32 20, i32 1, i32 2, i32 3);
  ret i32 %1
}

;--- read.ll
declare i32 @g(i32, i32, i32)
define i32 @f3() minsize {
  %1 = call i32 @g(i32 30, i32 1, i32 2, i32 3);
  ret i32 %1
}
