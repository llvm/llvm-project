; RUN: llc -march=amdgcn -mcpu=gfx1201 -debug-only=rewrite-partial-reg-uses -filetype=null %s 2>&1 | FileCheck %s

; CHECK: Try to rewrite partial reg {{%.*}}:SGPR_1024
; CHECK-NEXT:  Shift 32, reg align 32
; CHECK-NEXT:  sub4_sub5_sub6:SGPR_96 -> sub3_sub4_sub5  No improvement achieved

define amdgpu_cs void @_amdgpu_cs_main(i32 %LocalInvocationId) #0 {
entry:
  %dx.v32.x01 = alloca [24 x i32], align 4, addrspace(5)
  %0 = call <2 x i32> @llvm.amdgcn.s.buffer.load.v2i32(<4 x i32> zeroinitializer, i32 208, i32 0)
  %1 = extractelement <2 x i32> %0, i64 0
  %2 = insertelement <3 x i32> zeroinitializer, i32 %1, i64 0
  %3 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> zeroinitializer, i32 216, i32 0)
  %4 = insertelement <3 x i32> %2, i32 %3, i64 1
  %5 = getelementptr i8, ptr addrspace(5) %dx.v32.x01, i32 16
  store <3 x i32> %4, ptr addrspace(5) %5, align 4
  br label %loop0.breakc0

loop0.breakc0:                                    ; preds = %loop0.breakc0, %entry
  %dx.v32.r1.09 = phi float [ 0.000000e+00, %entry ], [ %dx.v32.r27.0, %loop0.breakc0 ]
  store <3 x i32> zeroinitializer, ptr addrspace(5) %dx.v32.x01, align 4
  %.idx1024 = shl i32 %LocalInvocationId, 4
  %6 = getelementptr i8, ptr addrspace(5) %dx.v32.x01, i32 %.idx1024
  %7 = load <3 x float>, ptr addrspace(5) %6, align 4
  %8 = extractelement <3 x float> %7, i64 0
  %9 = fcmp ogt float %8, 0.000000e+00
  %10 = fadd float %dx.v32.r1.09, 0.000000e+00
  %dx.v32.r27.0 = select i1 %9, float %10, float %dx.v32.r1.09
  br label %loop0.breakc0
}

declare <2 x i32> @llvm.amdgcn.s.buffer.load.v2i32(<4 x i32>, i32, i32 immarg)
declare i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32>, i32, i32 immarg)

attributes #0 = { denormal_fpenv(float: preservesign) }
