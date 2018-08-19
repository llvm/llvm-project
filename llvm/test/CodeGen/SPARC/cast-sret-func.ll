; RUN: llc < %s -march=sparc

; CHECK: call func
; CHECK: st %i0, [%sp+64]
; CHECK: unimp 8

%struct = type { i32, i32 }

define void @test() nounwind {
entry:
  %tmp = alloca %struct, align 4
  call void bitcast (void ()* @func to void (%struct*)*)
    (%struct* nonnull sret %tmp)
  ret void
}

declare void @func()
