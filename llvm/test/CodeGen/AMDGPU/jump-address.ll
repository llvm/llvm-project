;RUN: llc < %s -mtriple=r600 -mcpu=redwood | FileCheck %s

; CHECK: JUMP @6
; CHECK: EXPORT
; CHECK-NOT: EXPORT

define amdgpu_ps void @main() {
main_body:
  %0 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 1)
  %1 = extractelement <4 x float> %0, i32 0
  %2 = bitcast float %1 to i32
  %3 = icmp eq i32 %2, 0
  %4 = sext i1 %3 to i32
  %5 = bitcast i32 %4 to float
  %6 = bitcast float %5 to i32
  %7 = icmp ne i32 %6, 0
  br i1 %7, label %ENDIF, label %ELSE

ELSE:                                             ; preds = %main_body
  %8 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 1)
  %9 = extractelement <4 x float> %8, i32 0
  %10 = bitcast float %9 to i32
  %11 = icmp eq i32 %10, 1
  %12 = sext i1 %11 to i32
  %13 = bitcast i32 %12 to float
  %14 = bitcast float %13 to i32
  %15 = icmp ne i32 %14, 0
  br i1 %15, label %IF13, label %ENDIF

ENDIF:                                            ; preds = %IF13, %ELSE, %main_body
  %temp.0 = phi float [ 0xFFF8000000000000, %main_body ], [ 0.000000e+00, %ELSE ], [ 0.000000e+00, %IF13 ]
  %temp1.0 = phi float [ 0.000000e+00, %main_body ], [ %23, %IF13 ], [ 0.000000e+00, %ELSE ]
  %temp2.0 = phi float [ 1.000000e+00, %main_body ], [ 0.000000e+00, %ELSE ], [ 0.000000e+00, %IF13 ]
  %temp3.0 = phi float [ 5.000000e-01, %main_body ], [ 0.000000e+00, %ELSE ], [ 0.000000e+00, %IF13 ]
  %16 = insertelement <4 x float> undef, float %temp.0, i32 0
  %17 = insertelement <4 x float> %16, float %temp1.0, i32 1
  %18 = insertelement <4 x float> %17, float %temp2.0, i32 2
  %19 = insertelement <4 x float> %18, float %temp3.0, i32 3
  call void @llvm.r600.store.swizzle(<4 x float> %19, i32 0, i32 0)
  ret void

IF13:                                             ; preds = %ELSE
  %20 = load <4 x float>, ptr addrspace(8) null
  %21 = extractelement <4 x float> %20, i32 0
  %22 = fsub float -0.000000e+00, %21
  %23 = fadd float 0x3FF8000000000000, %22
  br label %ENDIF
}

declare void @llvm.r600.store.swizzle(<4 x float>, i32, i32)
