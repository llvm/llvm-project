; RUN: llc -march=hexagon -filetype=obj -ifcvt-limit=0 -o - %s | llvm-objdump --no-print-imm-hex -d - | FileCheck %s

; CHECK: p0 = cmp.gt(r0,#-1); if (!p0.new) jump:nt

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
