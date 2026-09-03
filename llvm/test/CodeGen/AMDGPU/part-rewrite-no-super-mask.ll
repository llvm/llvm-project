; RUN: llc -mtriple=amdgpu -mcpu=gfx1201 -debug-only=rewrite-partial-reg-uses -filetype=null %s 2>&1 | FileCheck %s

; CHECK: Try to rewrite partial reg {{%.*}}:SGPR_1024
; CHECK-NEXT:  Shift 32, reg align 32
; CHECK-NEXT:  sub4_sub5_sub6:SGPR_96 -> sub3_sub4_sub5  No improvement achieved

define amdgpu_cs void @_amdgpu_cs_main(i32 %LocalInvocationId) #0 {
entry:
  %aval1 = alloca [24 x i32], align 4, addrspace(5)
  %i = call <2 x i32> @llvm.amdgcn.s.buffer.load.v2i32(<4 x i32> zeroinitializer, i32 208, i32 0)
  %i1 = extractelement <2 x i32> %i, i64 0
  %i2 = insertelement <3 x i32> zeroinitializer, i32 %i1, i64 0
  %i3 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> zeroinitializer, i32 216, i32 0)
  %i4 = insertelement <3 x i32> %i2, i32 %i3, i64 1
  %i5 = getelementptr i8, ptr addrspace(5) %aval1, i32 16
  store <3 x i32> %i4, ptr addrspace(5) %i5, align 4
  br label %loop0.breakc0

loop0.breakc0:                                    ; preds = %loop0.breakc0, %entry
  %aval2 = phi float [ 0.000000e+00, %entry ], [ %aval3, %loop0.breakc0 ]
  store <3 x i32> zeroinitializer, ptr addrspace(5) %aval1, align 4
  %idx1 = shl i32 %LocalInvocationId, 4
  %i6 = getelementptr i8, ptr addrspace(5) %aval1, i32 %idx1
  %i7 = load <3 x float>, ptr addrspace(5) %i6, align 4
  %i8 = extractelement <3 x float> %i7, i64 0
  %i9 = fcmp ogt float %i8, 0.000000e+00
  %i10 = fadd float %aval2, 0.000000e+00
  %aval3 = select i1 %i9, float %i10, float %aval2
  br label %loop0.breakc0
}

declare <2 x i32> @llvm.amdgcn.s.buffer.load.v2i32(<4 x i32>, i32, i32 immarg)
declare i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32>, i32, i32 immarg)

attributes #0 = { denormal_fpenv(float: preservesign) }
