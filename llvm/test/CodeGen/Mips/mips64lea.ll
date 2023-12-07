; RUN: llc -march=mips64el -mcpu=mips4 < %s | FileCheck %s
; RUN: llc -march=mips64el -mcpu=mips64 < %s | FileCheck %s

define void @foo3() nounwind {
entry:
; CHECK: daddiu ${{[0-9]+}}, $sp
  %a = alloca i32, align 4
  call void @foo1(ptr %a) nounwind
  ret void
}

declare void @foo1(ptr)

