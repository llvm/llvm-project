; RUN: not llc -mtriple=bpf < %s 2> %t1
; RUN: FileCheck %s < %t1
; CHECK: error: <unknown>:0:0: in function bar i32 (i32, i32, i32, i32, i32, i32): stack arguments are not supported

; Function Attrs: nounwind readnone uwtable
define i32 @bar(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f) #0 {
entry:
  ret i32 1
}

; Function Attrs: nounwind readnone uwtable
define i32 @foo(i32 %a, i32 %b, i32 %c) #0 {
entry:
  ret i32 1
}
