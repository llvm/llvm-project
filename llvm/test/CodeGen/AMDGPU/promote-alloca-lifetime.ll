; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -amdgpu-promote-alloca %s | FileCheck -check-prefix=OPT %s

declare void @llvm.lifetime.start(i64, i8* nocapture) #0
declare void @llvm.lifetime.end(i64, i8* nocapture) #0

; OPT-LABEL: @use_lifetime_promotable_lds(
; OPT-NOT: alloca i32
; OPT-NOT: llvm.lifetime
; OPT: store i32 %tmp3, i32 addrspace(3)*
define amdgpu_kernel void @use_lifetime_promotable_lds(i32 addrspace(1)* %arg) #2 {
bb:
  %tmp = alloca i32, align 4
  %tmp1 = bitcast i32* %tmp to i8*
  call void @llvm.lifetime.start(i64 4, i8* %tmp1)
  %tmp2 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 1
  %tmp3 = load i32, i32 addrspace(1)* %tmp2
  store i32 %tmp3, i32* %tmp
  call void @llvm.lifetime.end(i64 4, i8* %tmp1)
  ret void
}

attributes #0 = { argmemonly nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
