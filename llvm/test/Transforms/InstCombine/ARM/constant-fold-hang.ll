; RUN: opt -instcombine < %s

; Function Attrs: nounwind readnone ssp
define void @mulByZero(<4 x i16> %x) #0 {
entry:
  %a = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %x, <4 x i16> zeroinitializer) #2
  ret void
}

; Function Attrs: nounwind readnone
declare <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16>, <4 x i16>) #1

attributes #0 = { nounwind readnone ssp }
attributes #1 = { nounwind readnone }
