; RUN: llc -march=bpfel -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s

; CHECK: if r2 s> r1 goto
; CHECK: call
; CHECK: exit

declare void @a()
declare void @b()

define void @foo(i32 %a) {
%b = icmp sgt i32 %a, -1
br i1 %b, label %x, label %y
x:
call void @a()
ret void
y:
call void @b()
ret void
}
