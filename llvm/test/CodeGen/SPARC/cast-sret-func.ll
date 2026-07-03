; RUN: llc < %s -mtriple=sparc

; CHECK: call func
; CHECK: st %i0, [%sp+64]
; CHECK: unimp 8

%struct = type { i32, i32 }

define void @test() nounwind {
entry:
  %tmp = alloca %struct, align 4
  call void @func
    (ptr nonnull sret(%struct) %tmp)
  ret void
}

declare void @func()
