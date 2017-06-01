; RUN: opt -o /dev/null -structurizecfg %s

; The following function caused an infinite loop inside the structurizer's
; rebuildSSA routine, where we were iterating over an instruction's uses while
; modifying the use list, without taking care to do this safely.

target triple = "amdgcn--"

declare <4 x float> @llvm.SI.vs.load.input(<16 x i8>, i32, i32) #0
declare <4 x float> @llvm.amdgcn.image.load.v4f32.v2i32.v8i32(<2 x i32>, <8 x i32>, i32, i1, i1, i1, i1) #1
declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float) #2

define amdgpu_vs void @wrapper(i32 inreg, i32) {
main_body:
  %2 = add i32 %1, %0
  %3 = call <4 x float> @llvm.SI.vs.load.input(<16 x i8> undef, i32 0, i32 %2)
  %4 = extractelement <4 x float> %3, i32 1
  %5 = fptosi float %4 to i32
  %6 = insertelement <2 x i32> undef, i32 %5, i32 1
  br label %loop11.i

loop11.i:                                         ; preds = %endif46.i, %main_body
  %7 = phi i32 [ 0, %main_body ], [ %15, %endif46.i ]
  %8 = icmp sgt i32 %7, 999
  br i1 %8, label %main.exit, label %if16.i

if16.i:                                           ; preds = %loop11.i
  %9 = call <4 x float> @llvm.amdgcn.image.load.v4f32.v2i32.v8i32(<2 x i32> %6, <8 x i32> undef, i32 15, i1 true, i1 false, i1 false, i1 false)
  %10 = extractelement <4 x float> %9, i32 0
  %11 = fcmp ult float 0.000000e+00, %10
  br i1 %11, label %if28.i, label %endif46.i

if28.i:                                           ; preds = %if16.i
  %12 = bitcast float %10 to i32
  %13 = shl i32 %12, 16
  %14 = bitcast i32 %13 to float
  br label %main.exit

endif46.i:                                        ; preds = %if16.i
  %15 = add i32 %7, 1
  br label %loop11.i

main.exit:                                        ; preds = %if28.i, %loop11.i
  %16 = phi float [ %14, %if28.i ], [ 0x36F0800000000000, %loop11.i ]
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 32, i32 0, float %16, float 0.000000e+00, float 0.000000e+00, float 0x36A0000000000000)
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind readonly }
attributes #2 = { nounwind }
