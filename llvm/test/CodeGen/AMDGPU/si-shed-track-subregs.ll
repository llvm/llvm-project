; This test does not check anything. Just ensure no crash.
; RUN: llc -O2 -mtriple amdgcn--amdhsa --misched=si -mattr=si-scheduler -mcpu=fiji -filetype=asm < %s

declare i32 @llvm.amdgcn.workitem.id.x() #4

declare i32 @llvm.amdgcn.workitem.id.y() #4

define amdgpu_kernel void @"test"(float addrspace(1)* nocapture,
 [4 x [4 x float]] addrspace(3) *,
 [4 x [4 x float]] addrspace(3) *,
 [4 x [4 x float]] addrspace(3) *,
 [4 x [4 x float]] addrspace(3) *
) {

  %st_addr = getelementptr float, float addrspace(1)* %0, i64 10
  %id_x = tail call i32 @llvm.amdgcn.workitem.id.x() #4
  %id_y = tail call i32 @llvm.amdgcn.workitem.id.y() #4

  %6  = getelementptr [4 x [4 x float]], [4 x [4 x float]] addrspace(3)* %1, i32 0, i32 %id_y, i32 1234
  %7  = getelementptr [4 x [4 x float]], [4 x [4 x float]] addrspace(3)* %2, i32 0, i32 0, i32 %id_x
  %8  = getelementptr [4 x [4 x float]], [4 x [4 x float]] addrspace(3)* %3, i32 0, i32 %id_y, i32 0
  %9  = getelementptr [4 x [4 x float]], [4 x [4 x float]] addrspace(3)* %4, i32 0, i32 0, i32 %id_x
  %10 = getelementptr [4 x [4 x float]], [4 x [4 x float]] addrspace(3)* %1, i32 0, i32 %id_y, i32 1294
  %11 = getelementptr [4 x [4 x float]], [4 x [4 x float]] addrspace(3)* %2, i32 0, i32 1, i32 %id_x
  %12 = getelementptr [4 x [4 x float]], [4 x [4 x float]] addrspace(3)* %3, i32 0, i32 %id_y, i32 1
  %13 = getelementptr [4 x [4 x float]], [4 x [4 x float]] addrspace(3)* %4, i32 0, i32 1, i32 %id_x


  %14 = load float, float addrspace(3)* %6
  %15 = load float, float addrspace(3)* %7
  %mul3 = fmul float %14, %15
  %add1 = fadd float 2.0, %mul3
  %16 = load float, float addrspace(3)* %8
  %17 = load float, float addrspace(3)* %9
  %mul4 = fmul float %16, %17
  %sub2 = fsub float %add1, %mul4
  %18 = load float, float addrspace(3)* %10
  %19 = load float, float addrspace(3)* %11
  %mul5 = fmul float %18, %19
  %sub3 = fsub float %sub2, %mul5
  %20 = load float, float addrspace(3)* %12
  %21 = load float, float addrspace(3)* %13
  %mul6 = fmul float %20, %21
  %sub4 = fsub float %sub3, %mul6
  store float %sub4, float addrspace(1)* %st_addr
  ret void
}

attributes #4 = { nounwind readnone }
