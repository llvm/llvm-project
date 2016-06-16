; RUN: llc -march=amdgcn -mcpu=tonga -post-RA-scheduler=0 < %s | FileCheck %s

; CHECK: NumVgprs: 64
define void @main([9 x <16 x i8>] addrspace(2)* byval, [17 x <16 x i8>] addrspace(2)* byval, [17 x <8 x i32>] addrspace(2)* byval, [16 x <8 x i32>] addrspace(2)* byval, [16 x <4 x i32>] addrspace(2)* byval, <3 x i32> inreg, <3 x i32> inreg, <3 x i32>) #0 {
main_body:
  %8 = getelementptr [16 x <4 x i32>], [16 x <4 x i32>] addrspace(2)* %4, i64 0, i64 8
  %9 = load <4 x i32>, <4 x i32> addrspace(2)* %8, align 16, !tbaa !0
  %10 = extractelement <3 x i32> %7, i32 0
  %11 = extractelement <3 x i32> %7, i32 1
  %12 = mul i32 %10, %11
  %bc = bitcast <3 x i32> %7 to <3 x float>
  %13 = extractelement <3 x float> %bc, i32 1
  %14 = insertelement <512 x float> undef, float %13, i32 %12
  call void @llvm.amdgcn.s.barrier()
  %15 = extractelement <3 x i32> %6, i32 0
  %16 = extractelement <3 x i32> %7, i32 0
  %17 = shl i32 %15, 5
  %18 = add i32 %17, %16
  %19 = shl i32 %18, 4
  %20 = extractelement <3 x i32> %7, i32 1
  %21 = shl i32 %20, 2
  %22 = sext i32 %21 to i64
  %23 = getelementptr i8, i8 addrspace(3)* null, i64 %22
  %24 = bitcast i8 addrspace(3)* %23 to i32 addrspace(3)*
  %25 = load i32, i32 addrspace(3)* %24, align 4
  %26 = extractelement <512 x float> %14, i32 %25
  %27 = insertelement <4 x float> undef, float %26, i32 0
  call void @llvm.amdgcn.buffer.store.format.v4f32(<4 x float> %27, <4 x i32> %9, i32 0, i32 %19, i1 false, i1 false)
  ret void
}

declare void @llvm.amdgcn.s.barrier() #1

declare void @llvm.amdgcn.buffer.store.format.v4f32(<4 x float>, <4 x i32>, i32, i32, i1, i1) #2

attributes #0 = { "amdgpu-max-work-group-size"="1024" }
attributes #1 = { convergent nounwind }
attributes #2 = { nounwind }

!0 = !{!1, !1, i64 0, i32 1}
!1 = !{!"const", null}
