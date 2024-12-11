; RUN: llc < %s -march=mips -mcpu=mips32r5 -mattr=+fp64,+msa | FileCheck %s

; Test that fexup[rl].w don't crash LLVM during type legalization.

@g = local_unnamed_addr global <8 x half> <half 0xH5BF8, half 0xH5BF8, half 0xH5BF8, half 0xH5BF8, half 0xH73C0, half 0xH73C0, half 0xH73C0, half 0xH73C0>, align 16
@i = local_unnamed_addr global <4 x float> zeroinitializer, align 16
@j = local_unnamed_addr global <4 x float> zeroinitializer, align 16

define i32 @test() local_unnamed_addr {
entry:
  %0 = load <8 x half>, ptr @g, align 16
  %1 = tail call <4 x float> @llvm.mips.fexupl.w(<8 x half> %0)
  store <4 x float> %1, ptr @i, align 16
; CHECK: ld.h $w[[W0:[0-9]+]], 0(${{[0-9]+}})
; CHECK: fexupl.w $w[[W1:[0-9]+]], $w[[W0]]
; CHECK: st.w $w[[W1]], 0(${{[0-9]+}})

  %2 = tail call <4 x float> @llvm.mips.fexupr.w(<8 x half> %0)
  store <4 x float> %2, ptr @j, align 16

; CHECK: fexupr.w $w[[W2:[0-9]+]], $w[[W0]]
; CHECK: st.w $w[[W2]], 0(${{[0-9]+}})

  ret i32 0
}

declare <4 x float> @llvm.mips.fexupl.w(<8 x half>)
declare <4 x float> @llvm.mips.fexupr.w(<8 x half>)
