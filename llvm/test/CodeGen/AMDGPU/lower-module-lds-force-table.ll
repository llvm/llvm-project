; RUN: opt -S -mtriple=amdgcn-- -mcpu=gfx942 -amdgpu-lower-module-lds < %s 2>&1 | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -mcpu=gfx942 -passes=amdgpu-lower-module-lds < %s 2>&1 | FileCheck %s

; This test has the following kernels with following GV access pattern
; EN32 kernels
; EN32_compress_wrapperIhm - GV's 1, 2, 3, 4, 5, 6, 7
; EN32_compress_wrapperItm - GV's 8, 9, 10, 11, 12, 13, 7
; EN32_compress_wrapperIjm - GV's 15, 16, 17, 18, 19, 20, 7
; EN32_compress_wrapperImm - GV's 21, 22, 23, 24, 25, 26, 27, 7
; EN64 kernels
; EN64_compress_wrapperIhm - GV's 1, 2, 3, 4, 5, 6, 7
; EN64_compress_wrapperItm - GV's 8, 9, 10, 11, 12, 13, 7
; EN64_compress_wrapperIjm - GV's 15, 16, 17, 18, 19, 20, 7
; EN64_compress_wrapperImm - GV's 21, 22, 23, 24, 25, 26, 27, 7

; CHECK: define amdgpu_kernel void @EN32_compress_wrapperIhm() #0
; CHECK: define amdgpu_kernel void @EN32_compress_wrapperItm() #2
; CHECK: define amdgpu_kernel void @EN32_compress_wrapperIjm() #3
; CHECK: define amdgpu_kernel void @EN32_compress_wrapperImm() #4
; CHECK: define amdgpu_kernel void @EN64_compress_wrapperIhm() #0
; CHECK: define amdgpu_kernel void @EN64_compress_wrapperItm() #2
; CHECK: define amdgpu_kernel void @EN64_compress_wrapperIjm() #3
; CHECK: define amdgpu_kernel void @EN64_compress_wrapperImm() #4

; CHECK: attributes #0 = { "amdgpu-lds-size"="25760" "target-cpu"="gfx942" }
; CHECK: attributes #2 = { "amdgpu-lds-size"="17560" "target-cpu"="gfx942" }
; CHECK: attributes #3 = { "amdgpu-lds-size"="13464" "target-cpu"="gfx942" }
; CHECK: attributes #4 = { "amdgpu-lds-size"="11424" "target-cpu"="gfx942" }

%RawStorage1 = type { [1056 x i8] }
%RawStorage2 = type { [4 x i8] }
%RawStorage3 = type { [16 x i8] }

@one = addrspace(3) global [1026 x i32] poison
@two = addrspace(3) global [1026 x i32] poison
@three = external addrspace(3) global [2048 x i32]
@four = addrspace(3) global [2050 x i32] poison
@five = addrspace(3) global [16 x i32] poison
@six = external addrspace(3) global %RawStorage1
@seven = addrspace(3) global %RawStorage2 poison
@eight = addrspace(3) global [1026 x i32] poison
@nine = addrspace(3) global [1026 x i32] poison
@ten = external addrspace(3) global [1024 x i32]
@eleven = addrspace(3) global [1026 x i32] poison
@twelve = external addrspace(3) global [16 x i32]
@thirteen = external addrspace(3) global %RawStorage1
@fourteen = external addrspace(3) global [1 x i32]
@fifteen = addrspace(3) global [1026 x i32] poison
@sixteen = addrspace(3) global [1026 x i32] poison
@seventeen = external addrspace(3) global [512 x i32]
@eighteen = addrspace(3) global [514 x i32] poison
@nineteen = external addrspace(3) global [16 x i32]
@twenty = external addrspace(3) global %RawStorage1
@twentyone = external addrspace(3) global [514 x i64]
@twentytwo = external addrspace(3) global [514 x i64]
@twentythree = external addrspace(3) global [256 x i32]
@twentyfour = external addrspace(3) global [258 x i32]
@twentyfive = external addrspace(3) global [16 x i32]
@twentysix = external addrspace(3) global %RawStorage1
@twentyseven = external addrspace(3) global %RawStorage3

define amdgpu_kernel void @EN32_compress_wrapperIhm() {
entry:
  %0 = call i32 @Ihm_one()
  ret void
}

define i32 @Ihm_one() {
entry:
  %0 = call i32 @Ihm_chunk()
  ret i32 %0
}

define i32 @Ihm_chunk() {
entry:
  %0 = call i32 @Ihm_CascadedOpts()
  ret i32 %0
}

define i32 @Ihm_CascadedOpts() {
entry:
  store ptr addrspacecast (ptr addrspace(3) @one to ptr), ptr null, align 8
  store ptr addrspacecast (ptr addrspace(3) @two to ptr), ptr null, align 8
  %add.ptr = getelementptr i32, ptr getelementptr inbounds (i32, ptr addrspacecast (ptr addrspace(3) @five to ptr), i64 1), i64 0
  call void @Ihm_PS1_PT1_PS4_S7()
  %call69 = call i32 @foo(ptr addrspacecast (ptr addrspace(3) @three to ptr), ptr addrspacecast (ptr addrspace(3) @four to ptr))
  ret i32 %call69
}

define void @Ihm_PS1_PT1_PS4_S7() {
entry:
  store ptr addrspacecast (ptr addrspace(3) @six to ptr), ptr null, align 8
  ret void
}

define i32 @foo(ptr %input, ptr %temp_storage) {
entry:
  call void @Itm_PjPS4()
  ret i32 0
}

define void @Itm_PjPS4() {
entry:
  call void @Itm_PS1_Pj()
  ret void
}

define void @Itm_PS1_Pj() {
entry:
  store ptr addrspacecast (ptr addrspace(3) @seven to ptr), ptr null, align 8
  ret void
}

define amdgpu_kernel void @EN32_compress_wrapperItm() {
entry:
  %0 = call i32 @Itm_one()
  ret void
}

define i32 @Itm_one() {
entry:
  %0 = call i32 @Itm_chunk()
  ret i32 %0
}

define i32 @Itm_chunk() {
entry:
  %0 = call i32 @Itm_CascadedOpts()
  ret i32 %0
}

define i32 @Itm_CascadedOpts() {
entry:
  store ptr addrspacecast (ptr addrspace(3) @eight to ptr), ptr null, align 8
  store ptr addrspacecast (ptr addrspace(3) @nine to ptr), ptr null, align 8
  %add.ptr = getelementptr i32, ptr getelementptr inbounds (i32, ptr addrspacecast (ptr addrspace(3) @twelve to ptr), i64 1), i64 0
  call void @Itm_PS1_PT1_PS4_S7()
  %call69 = call i32 @foo(ptr addrspacecast (ptr addrspace(3) @ten to ptr), ptr addrspacecast (ptr addrspace(3) @eleven to ptr))
  ret i32 %call69
}

define void @Itm_PS1_PT1_PS4_S7() {
entry:
  store ptr addrspacecast (ptr addrspace(3) @thirteen to ptr), ptr null, align 8
  ret void
}

define amdgpu_kernel void @EN32_compress_wrapperIjm() {
entry:
  %arrayidx = getelementptr [1 x i32], ptr addrspacecast (ptr addrspace(3) @fourteen to ptr), i64 0, i64 0
  %0 = call i32 @Ijm_one()
  ret void
}

define i32 @Ijm_one() {
entry:
  %0 = call i32 @Ijm_chunk()
  ret i32 %0
}

define i32 @Ijm_chunk() {
entry:
  %0 = call i32 @Ijm_CascadedOpts()
  ret i32 %0
}

define i32 @Ijm_CascadedOpts() {
entry:
  store ptr addrspacecast (ptr addrspace(3) @fifteen to ptr), ptr null, align 8
  store ptr addrspacecast (ptr addrspace(3) @sixteen to ptr), ptr null, align 8
  %add.ptr = getelementptr i32, ptr getelementptr inbounds (i32, ptr addrspacecast (ptr addrspace(3) @nineteen to ptr), i64 1), i64 0
  call void @Ijm_PS1_PT1_PS4_S7()
  %call69 = call i32 @foo(ptr addrspacecast (ptr addrspace(3) @seventeen to ptr), ptr addrspacecast (ptr addrspace(3) @eighteen to ptr))
  ret i32 %call69
}

define void @Ijm_PS1_PT1_PS4_S7() {
entry:
  store ptr addrspacecast (ptr addrspace(3) @twenty to ptr), ptr null, align 8
  ret void
}

define amdgpu_kernel void @EN32_compress_wrapperImm() {
entry:
  %0 = call i32 @Imm_one()
  ret void
}

define i32 @Imm_one() {
entry:
  %0 = call i32 @Imm_chunk()
  ret i32 %0
}

define i32 @Imm_chunk() {
entry:
  %0 = call i32 @Imm_CascadedOpts()
  ret i32 %0
}

define i32 @Imm_CascadedOpts() {
entry:
  store ptr addrspacecast (ptr addrspace(3) @twentyone to ptr), ptr null, align 8
  store ptr addrspacecast (ptr addrspace(3) @twentytwo to ptr), ptr null, align 8
  %add.ptr = getelementptr i32, ptr getelementptr inbounds (i32, ptr addrspacecast (ptr addrspace(3) @twentyfive to ptr), i64 1), i64 0
  br i1 false, label %for.body65, label %for.end102

for.body65:
  call void @Imm_PS1_PT1_PS4_S7()
  %call69 = call i32 @foo(ptr addrspacecast (ptr addrspace(3) @twentythree to ptr), ptr addrspacecast (ptr addrspace(3) @twentyfour to ptr))
  ret i32 %call69

for.end102:
  %call106 = call i32 @Imm_PjPKjPS5_S6_b()
  ret i32 0
}

define void @Imm_PS1_PT1_PS4_S7() {
entry:
  store ptr addrspacecast (ptr addrspace(3) @twentysix to ptr), ptr null, align 8
  ret void
}

define i32 @Imm_PjPKjPS5_S6_b() {
entry:
  call void @Imm_PjPS4()
  ret i32 0
}

define void @Imm_PjPS4() {
entry:
  call void @Imm_PS1_Pj()
  ret void
}

define void @Imm_PS1_Pj() {
entry:
  store ptr addrspacecast (ptr addrspace(3) @twentyseven to ptr), ptr null, align 8
  ret void
}

define amdgpu_kernel void @EN64_compress_wrapperIhm() {
entry:
  %0 = call i32 @Ihm_one()
  ret void
}

define amdgpu_kernel void @EN64_compress_wrapperItm() {
entry:
  %0 = call i32 @Itm_one()
  ret void
}

define amdgpu_kernel void @EN64_compress_wrapperIjm() {
entry:
  %0 = call i32 @Ijm_one()
  ret void
}

define amdgpu_kernel void @EN64_compress_wrapperImm() {
entry:
  %0 = call i32 @Imm_one()
  ret void
}
