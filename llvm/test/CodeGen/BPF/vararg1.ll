; RUN: not llc -mtriple=bpf < %s 2> %t1
; RUN: FileCheck %s < %t1
; CHECK: error: <unknown>:0:0: in function foo void (i32, ...): variadic functions are not supported

; Function Attrs: nounwind readnone uwtable
define void @foo(i32 %a, ...) #0 {
entry:
  ret void
}
