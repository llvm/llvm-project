; RUN: opt -o /dev/null -structurizecfg %s

; The following function caused an infinite loop inside the structurizer's
; rebuildSSA routine, where we were iterating over an instruction's uses while
; modifying the use list, without taking care to do this safely.

target triple = "amdgcn--"

define amdgpu_vs void @wrapper(i32 inreg %arg, i32 %arg1) {
main_body:
  %tmp = add i32 %arg1, %arg
  %tmp2 = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> undef, i32 %tmp, i32 0, i32 0, i32 0)
  %tmp3 = extractelement <4 x float> %tmp2, i32 1
  %tmp4 = fptosi float %tmp3 to i32
  %tmp5 = insertelement <2 x i32> undef, i32 %tmp4, i32 1
  br label %loop11.i

loop11.i:                                         ; preds = %endif46.i, %main_body
  %tmp6 = phi i32 [ 0, %main_body ], [ %tmp14, %endif46.i ]
  %tmp7 = icmp sgt i32 %tmp6, 999
  br i1 %tmp7, label %main.exit, label %if16.i

if16.i:                                           ; preds = %loop11.i
  %tmp8 = call <4 x float> @llvm.amdgcn.image.load.v4f32.v2i32.v8i32(<2 x i32> %tmp5, <8 x i32> undef, i32 15, i1 true, i1 false, i1 false, i1 false)
  %tmp9 = extractelement <4 x float> %tmp8, i32 0
  %tmp10 = fcmp ult float 0.000000e+00, %tmp9
  br i1 %tmp10, label %if28.i, label %endif46.i

if28.i:                                           ; preds = %if16.i
  %tmp11 = bitcast float %tmp9 to i32
  %tmp12 = shl i32 %tmp11, 16
  %tmp13 = bitcast i32 %tmp12 to float
  br label %main.exit

endif46.i:                                        ; preds = %if16.i
  %tmp14 = add i32 %tmp6, 1
  br label %loop11.i

main.exit:                                        ; preds = %if28.i, %loop11.i
  %tmp15 = phi float [ %tmp13, %if28.i ], [ 0x36F0800000000000, %loop11.i ]
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float %tmp15, float 0.000000e+00, float 0.000000e+00, float 0x36A0000000000000, i1 false, i1 false) #0
  ret void
}

; Function Attrs: nounwind
declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #0

declare <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32>, i32, i32, i32, i32 immarg) #2
declare <4 x float> @llvm.amdgcn.image.load.v4f32.v2i32.v8i32(<2 x i32>, <8 x i32>, i32, i1, i1, i1, i1) #2

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind readonly }
