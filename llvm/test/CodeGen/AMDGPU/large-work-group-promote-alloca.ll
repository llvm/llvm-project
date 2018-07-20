; RUN: opt -S -mtriple=amdgcn-unknown-unknown -amdgpu-promote-alloca -disable-promote-alloca-to-vector < %s | FileCheck --check-prefix=SI --check-prefix=ALL %s
; RUN: opt -S -mcpu=tonga -mtriple=amdgcn-unknown-unknown -amdgpu-promote-alloca -disable-promote-alloca-to-vector < %s | FileCheck --check-prefix=CI --check-prefix=ALL %s

; SI-NOT: @promote_alloca_size_63.stack = internal unnamed_addr addrspace(3) global [63 x [5 x i32]] undef, align 4
; CI: @promote_alloca_size_63.stack = internal unnamed_addr addrspace(3) global [63 x [5 x i32]] undef, align 4

define amdgpu_kernel void @promote_alloca_size_63(i32 addrspace(1)* nocapture %out, i32 addrspace(1)* nocapture %in) #0 {
entry:
  %stack = alloca [5 x i32], align 4
  %0 = load i32, i32 addrspace(1)* %in, align 4
  %arrayidx1 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 %0
  store i32 4, i32* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 1
  %1 = load i32, i32 addrspace(1)* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 %1
  store i32 5, i32* %arrayidx3, align 4
  %arrayidx10 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 0
  %2 = load i32, i32* %arrayidx10, align 4
  store i32 %2, i32 addrspace(1)* %out, align 4
  %arrayidx12 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 1
  %3 = load i32, i32* %arrayidx12
  %arrayidx13 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 1
  store i32 %3, i32 addrspace(1)* %arrayidx13
  ret void
}

; ALL: @promote_alloca_size_256.stack = internal unnamed_addr addrspace(3) global [256 x [5 x i32]] undef, align 4

define amdgpu_kernel void @promote_alloca_size_256(i32 addrspace(1)* nocapture %out, i32 addrspace(1)* nocapture %in) #1 {
entry:
  %stack = alloca [5 x i32], align 4
  %0 = load i32, i32 addrspace(1)* %in, align 4
  %arrayidx1 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 %0
  store i32 4, i32* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 1
  %1 = load i32, i32 addrspace(1)* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 %1
  store i32 5, i32* %arrayidx3, align 4
  %arrayidx10 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 0
  %2 = load i32, i32* %arrayidx10, align 4
  store i32 %2, i32 addrspace(1)* %out, align 4
  %arrayidx12 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 1
  %3 = load i32, i32* %arrayidx12
  %arrayidx13 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 1
  store i32 %3, i32 addrspace(1)* %arrayidx13
  ret void
}

; ALL: @promote_alloca_size_1600.stack = internal unnamed_addr addrspace(3) global [1600 x [5 x i32]] undef, align 4

define amdgpu_kernel void @promote_alloca_size_1600(i32 addrspace(1)* nocapture %out, i32 addrspace(1)* nocapture %in) #2 {
entry:
  %stack = alloca [5 x i32], align 4
  %0 = load i32, i32 addrspace(1)* %in, align 4
  %arrayidx1 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 %0
  store i32 4, i32* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 1
  %1 = load i32, i32 addrspace(1)* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 %1
  store i32 5, i32* %arrayidx3, align 4
  %arrayidx10 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 0
  %2 = load i32, i32* %arrayidx10, align 4
  store i32 %2, i32 addrspace(1)* %out, align 4
  %arrayidx12 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 1
  %3 = load i32, i32* %arrayidx12
  %arrayidx13 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 1
  store i32 %3, i32 addrspace(1)* %arrayidx13
  ret void
}

; ALL-LABEL: @occupancy_0(
; CI-NOT: alloca [5 x i32]
; SI: alloca [5 x i32]
define amdgpu_kernel void @occupancy_0(i32 addrspace(1)* nocapture %out, i32 addrspace(1)* nocapture %in) #3 {
entry:
  %stack = alloca [5 x i32], align 4
  %0 = load i32, i32 addrspace(1)* %in, align 4
  %arrayidx1 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 %0
  store i32 4, i32* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 1
  %1 = load i32, i32 addrspace(1)* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 %1
  store i32 5, i32* %arrayidx3, align 4
  %arrayidx10 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 0
  %2 = load i32, i32* %arrayidx10, align 4
  store i32 %2, i32 addrspace(1)* %out, align 4
  %arrayidx12 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 1
  %3 = load i32, i32* %arrayidx12
  %arrayidx13 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 1
  store i32 %3, i32 addrspace(1)* %arrayidx13
  ret void
}

; ALL-LABEL: @occupancy_max(
; CI-NOT: alloca [5 x i32]
; SI: alloca [5 x i32]
define amdgpu_kernel void @occupancy_max(i32 addrspace(1)* nocapture %out, i32 addrspace(1)* nocapture %in) #4 {
entry:
  %stack = alloca [5 x i32], align 4
  %0 = load i32, i32 addrspace(1)* %in, align 4
  %arrayidx1 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 %0
  store i32 4, i32* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 1
  %1 = load i32, i32 addrspace(1)* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 %1
  store i32 5, i32* %arrayidx3, align 4
  %arrayidx10 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 0
  %2 = load i32, i32* %arrayidx10, align 4
  store i32 %2, i32 addrspace(1)* %out, align 4
  %arrayidx12 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 1
  %3 = load i32, i32* %arrayidx12
  %arrayidx13 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 1
  store i32 %3, i32 addrspace(1)* %arrayidx13
  ret void
}

; SI-LABEL: @occupancy_6(
; CI-LABEL: @occupancy_6(
; SI: alloca
; CI-NOT: alloca
define amdgpu_kernel void @occupancy_6(i8 addrspace(1)* nocapture %out, i8 addrspace(1)* nocapture %in) #5 {
entry:
  %stack = alloca [42 x i8], align 4
  %tmp = load i8, i8 addrspace(1)* %in, align 1
  %tmp4 = sext i8 %tmp to i64
  %arrayidx1 = getelementptr inbounds [42 x i8], [42 x i8]* %stack, i64 0, i64 %tmp4
  store i8 4, i8* %arrayidx1, align 1
  %arrayidx2 = getelementptr inbounds i8, i8 addrspace(1)* %in, i64 1
  %tmp1 = load i8, i8 addrspace(1)* %arrayidx2, align 1
  %tmp5 = sext i8 %tmp1 to i64
  %arrayidx3 = getelementptr inbounds [42 x i8], [42 x i8]* %stack, i64 0, i64 %tmp5
  store i8 5, i8* %arrayidx3, align 1
  %arrayidx10 = getelementptr inbounds [42 x i8], [42 x i8]* %stack, i64 0, i64 0
  %tmp2 = load i8, i8* %arrayidx10, align 1
  store i8 %tmp2, i8 addrspace(1)* %out, align 1
  %arrayidx12 = getelementptr inbounds [42 x i8], [42 x i8]* %stack, i64 0, i64 1
  %tmp3 = load i8, i8* %arrayidx12, align 1
  %arrayidx13 = getelementptr inbounds i8, i8 addrspace(1)* %out, i64 1
  store i8 %tmp3, i8 addrspace(1)* %arrayidx13, align 1
  ret void
}

; ALL-LABEL: @occupancy_6_over(
; ALL: alloca [43 x i8]
define amdgpu_kernel void @occupancy_6_over(i8 addrspace(1)* nocapture %out, i8 addrspace(1)* nocapture %in) #5 {
entry:
  %stack = alloca [43 x i8], align 4
  %tmp = load i8, i8 addrspace(1)* %in, align 1
  %tmp4 = sext i8 %tmp to i64
  %arrayidx1 = getelementptr inbounds [43 x i8], [43 x i8]* %stack, i64 0, i64 %tmp4
  store i8 4, i8* %arrayidx1, align 1
  %arrayidx2 = getelementptr inbounds i8, i8 addrspace(1)* %in, i64 1
  %tmp1 = load i8, i8 addrspace(1)* %arrayidx2, align 1
  %tmp5 = sext i8 %tmp1 to i64
  %arrayidx3 = getelementptr inbounds [43 x i8], [43 x i8]* %stack, i64 0, i64 %tmp5
  store i8 5, i8* %arrayidx3, align 1
  %arrayidx10 = getelementptr inbounds [43 x i8], [43 x i8]* %stack, i64 0, i64 0
  %tmp2 = load i8, i8* %arrayidx10, align 1
  store i8 %tmp2, i8 addrspace(1)* %out, align 1
  %arrayidx12 = getelementptr inbounds [43 x i8], [43 x i8]* %stack, i64 0, i64 1
  %tmp3 = load i8, i8* %arrayidx12, align 1
  %arrayidx13 = getelementptr inbounds i8, i8 addrspace(1)* %out, i64 1
  store i8 %tmp3, i8 addrspace(1)* %arrayidx13, align 1
  ret void
}

; SI-LABEL: @occupancy_8(
; CI-LABEL: @occupancy_8(
; SI: alloca
; CI-NOT: alloca
define amdgpu_kernel void @occupancy_8(i8 addrspace(1)* nocapture %out, i8 addrspace(1)* nocapture %in) #6 {
entry:
  %stack = alloca [32 x i8], align 4
  %tmp = load i8, i8 addrspace(1)* %in, align 1
  %tmp4 = sext i8 %tmp to i64
  %arrayidx1 = getelementptr inbounds [32 x i8], [32 x i8]* %stack, i64 0, i64 %tmp4
  store i8 4, i8* %arrayidx1, align 1
  %arrayidx2 = getelementptr inbounds i8, i8 addrspace(1)* %in, i64 1
  %tmp1 = load i8, i8 addrspace(1)* %arrayidx2, align 1
  %tmp5 = sext i8 %tmp1 to i64
  %arrayidx3 = getelementptr inbounds [32 x i8], [32 x i8]* %stack, i64 0, i64 %tmp5
  store i8 5, i8* %arrayidx3, align 1
  %arrayidx10 = getelementptr inbounds [32 x i8], [32 x i8]* %stack, i64 0, i64 0
  %tmp2 = load i8, i8* %arrayidx10, align 1
  store i8 %tmp2, i8 addrspace(1)* %out, align 1
  %arrayidx12 = getelementptr inbounds [32 x i8], [32 x i8]* %stack, i64 0, i64 1
  %tmp3 = load i8, i8* %arrayidx12, align 1
  %arrayidx13 = getelementptr inbounds i8, i8 addrspace(1)* %out, i64 1
  store i8 %tmp3, i8 addrspace(1)* %arrayidx13, align 1
  ret void
}

; ALL-LABEL: @occupancy_8_over(
; ALL: alloca [33 x i8]
define amdgpu_kernel void @occupancy_8_over(i8 addrspace(1)* nocapture %out, i8 addrspace(1)* nocapture %in) #6 {
entry:
  %stack = alloca [33 x i8], align 4
  %tmp = load i8, i8 addrspace(1)* %in, align 1
  %tmp4 = sext i8 %tmp to i64
  %arrayidx1 = getelementptr inbounds [33 x i8], [33 x i8]* %stack, i64 0, i64 %tmp4
  store i8 4, i8* %arrayidx1, align 1
  %arrayidx2 = getelementptr inbounds i8, i8 addrspace(1)* %in, i64 1
  %tmp1 = load i8, i8 addrspace(1)* %arrayidx2, align 1
  %tmp5 = sext i8 %tmp1 to i64
  %arrayidx3 = getelementptr inbounds [33 x i8], [33 x i8]* %stack, i64 0, i64 %tmp5
  store i8 5, i8* %arrayidx3, align 1
  %arrayidx10 = getelementptr inbounds [33 x i8], [33 x i8]* %stack, i64 0, i64 0
  %tmp2 = load i8, i8* %arrayidx10, align 1
  store i8 %tmp2, i8 addrspace(1)* %out, align 1
  %arrayidx12 = getelementptr inbounds [33 x i8], [33 x i8]* %stack, i64 0, i64 1
  %tmp3 = load i8, i8* %arrayidx12, align 1
  %arrayidx13 = getelementptr inbounds i8, i8 addrspace(1)* %out, i64 1
  store i8 %tmp3, i8 addrspace(1)* %arrayidx13, align 1
  ret void
}

; SI-LABEL: @occupancy_9(
; CI-LABEL: @occupancy_9(
; SI: alloca
; CI-NOT: alloca
define amdgpu_kernel void @occupancy_9(i8 addrspace(1)* nocapture %out, i8 addrspace(1)* nocapture %in) #7 {
entry:
  %stack = alloca [28 x i8], align 4
  %tmp = load i8, i8 addrspace(1)* %in, align 1
  %tmp4 = sext i8 %tmp to i64
  %arrayidx1 = getelementptr inbounds [28 x i8], [28 x i8]* %stack, i64 0, i64 %tmp4
  store i8 4, i8* %arrayidx1, align 1
  %arrayidx2 = getelementptr inbounds i8, i8 addrspace(1)* %in, i64 1
  %tmp1 = load i8, i8 addrspace(1)* %arrayidx2, align 1
  %tmp5 = sext i8 %tmp1 to i64
  %arrayidx3 = getelementptr inbounds [28 x i8], [28 x i8]* %stack, i64 0, i64 %tmp5
  store i8 5, i8* %arrayidx3, align 1
  %arrayidx10 = getelementptr inbounds [28 x i8], [28 x i8]* %stack, i64 0, i64 0
  %tmp2 = load i8, i8* %arrayidx10, align 1
  store i8 %tmp2, i8 addrspace(1)* %out, align 1
  %arrayidx12 = getelementptr inbounds [28 x i8], [28 x i8]* %stack, i64 0, i64 1
  %tmp3 = load i8, i8* %arrayidx12, align 1
  %arrayidx13 = getelementptr inbounds i8, i8 addrspace(1)* %out, i64 1
  store i8 %tmp3, i8 addrspace(1)* %arrayidx13, align 1
  ret void
}

; ALL-LABEL: @occupancy_9_over(
; ALL: alloca [29 x i8]
define amdgpu_kernel void @occupancy_9_over(i8 addrspace(1)* nocapture %out, i8 addrspace(1)* nocapture %in) #7 {
entry:
  %stack = alloca [29 x i8], align 4
  %tmp = load i8, i8 addrspace(1)* %in, align 1
  %tmp4 = sext i8 %tmp to i64
  %arrayidx1 = getelementptr inbounds [29 x i8], [29 x i8]* %stack, i64 0, i64 %tmp4
  store i8 4, i8* %arrayidx1, align 1
  %arrayidx2 = getelementptr inbounds i8, i8 addrspace(1)* %in, i64 1
  %tmp1 = load i8, i8 addrspace(1)* %arrayidx2, align 1
  %tmp5 = sext i8 %tmp1 to i64
  %arrayidx3 = getelementptr inbounds [29 x i8], [29 x i8]* %stack, i64 0, i64 %tmp5
  store i8 5, i8* %arrayidx3, align 1
  %arrayidx10 = getelementptr inbounds [29 x i8], [29 x i8]* %stack, i64 0, i64 0
  %tmp2 = load i8, i8* %arrayidx10, align 1
  store i8 %tmp2, i8 addrspace(1)* %out, align 1
  %arrayidx12 = getelementptr inbounds [29 x i8], [29 x i8]* %stack, i64 0, i64 1
  %tmp3 = load i8, i8* %arrayidx12, align 1
  %arrayidx13 = getelementptr inbounds i8, i8 addrspace(1)* %out, i64 1
  store i8 %tmp3, i8 addrspace(1)* %arrayidx13, align 1
  ret void
}

attributes #0 = { nounwind "amdgpu-max-work-group-size"="63" }
attributes #1 = { nounwind "amdgpu-waves-per-eu"="1,3" "amdgpu-flat-work-group-size"="256,256" }
attributes #2 = { nounwind "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="1600,1600" }
attributes #3 = { nounwind "amdgpu-waves-per-eu"="1,10" }
attributes #4 = { nounwind "amdgpu-waves-per-eu"="1,10" }
attributes #5 = { nounwind "amdgpu-waves-per-eu"="1,6" "amdgpu-flat-work-group-size"="64,64" }
attributes #6 = { nounwind "amdgpu-waves-per-eu"="1,8" "amdgpu-flat-work-group-size"="64,64" }
attributes #7 = { nounwind "amdgpu-waves-per-eu"="1,9" "amdgpu-flat-work-group-size"="64,64" }
