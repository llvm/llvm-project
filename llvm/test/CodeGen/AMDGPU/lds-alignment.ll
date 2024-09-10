; RUN: llc -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck -check-prefix=HSA %s

@lds.align16.0 = internal unnamed_addr addrspace(3) global [38 x i8] undef, align 16
@lds.align16.1 = internal unnamed_addr addrspace(3) global [38 x i8] undef, align 16

@lds.align8.0 = internal unnamed_addr addrspace(3) global [38 x i8] undef, align 8
@lds.align32.0 = internal unnamed_addr addrspace(3) global [38 x i8] undef, align 32

@lds.missing.align.0 = internal unnamed_addr addrspace(3) global [39 x i32] undef
@lds.missing.align.1 = internal unnamed_addr addrspace(3) global [7 x i64] undef

declare void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) nocapture, ptr addrspace(1) nocapture readonly, i32, i1) #0
declare void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) nocapture, ptr addrspace(3) nocapture readonly, i32, i1) #0


; HSA-LABEL: {{^}}test_no_round_size_1:
; HSA: .amdhsa_group_segment_fixed_size 38
define amdgpu_kernel void @test_no_round_size_1(ptr addrspace(1) %out, ptr addrspace(1) %in) #1 {
  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 4 @lds.align16.0, ptr addrspace(1) align 4 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 4 %out, ptr addrspace(3) align 4 @lds.align16.0, i32 38, i1 false)
  ret void
}

; There are two objects, so one requires padding to be correctly
; aligned after the other.

; (38 -> 48) + 38 = 92

; I don't think it is necessary to add padding after since if there
; were to be a dynamically sized LDS kernel arg, the runtime should
; add the alignment padding if necessary alignment padding if needed.

; HSA-LABEL: {{^}}test_round_size_2:
; HSA: .amdhsa_group_segment_fixed_size 86
define amdgpu_kernel void @test_round_size_2(ptr addrspace(1) %out, ptr addrspace(1) %in) #1 {
  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 4 @lds.align16.0, ptr addrspace(1) align 4 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 4 %out, ptr addrspace(3) align 4 @lds.align16.0, i32 38, i1 false)

  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 4 @lds.align16.1, ptr addrspace(1) align 4 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 4 %out, ptr addrspace(3) align 4 @lds.align16.1, i32 38, i1 false)

  ret void
}

; 38 + (10 pad) + 38  (= 86)
; HSA-LABEL: {{^}}test_round_size_2_align_8:
; HSA: .amdhsa_group_segment_fixed_size 86
define amdgpu_kernel void @test_round_size_2_align_8(ptr addrspace(1) %out, ptr addrspace(1) %in) #1 {
  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 8 @lds.align16.0, ptr addrspace(1) align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 8 %out, ptr addrspace(3) align 8 @lds.align16.0, i32 38, i1 false)

  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 8 @lds.align8.0, ptr addrspace(1) align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 8 %out, ptr addrspace(3) align 8 @lds.align8.0, i32 38, i1 false)

  ret void
}

; HSA-LABEL: {{^}}test_round_local_lds_and_arg:
; HSA: .amdhsa_group_segment_fixed_size 38
define amdgpu_kernel void @test_round_local_lds_and_arg(ptr addrspace(1) %out, ptr addrspace(1) %in, ptr addrspace(3) %lds.arg) #1 {
  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 4 @lds.align16.0, ptr addrspace(1) align 4 %in, i32 38, i1 false)

  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 4 %out, ptr addrspace(3) align 4 @lds.align16.0, i32 38, i1 false)
  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 4 %lds.arg, ptr addrspace(1) align 4 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 4 %out, ptr addrspace(3) align 4 %lds.arg, i32 38, i1 false)
  ret void
}

; HSA-LABEL: {{^}}test_round_lds_arg:
; HSA: .amdhsa_group_segment_fixed_size 0
define amdgpu_kernel void @test_round_lds_arg(ptr addrspace(1) %out, ptr addrspace(1) %in, ptr addrspace(3) %lds.arg) #1 {
  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 4 %lds.arg, ptr addrspace(1) align 4 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 4 %out, ptr addrspace(3) align 4 %lds.arg, i32 38, i1 false)
  ret void
}

; FIXME: Parameter alignment not considered
; HSA-LABEL: {{^}}test_high_align_lds_arg:
; HSA: .amdhsa_group_segment_fixed_size 0
define amdgpu_kernel void @test_high_align_lds_arg(ptr addrspace(1) %out, ptr addrspace(1) %in, ptr addrspace(3) align 64 %lds.arg) #1 {
  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 64 %lds.arg, ptr addrspace(1) align 64 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 64 %out, ptr addrspace(3) align 64 %lds.arg, i32 38, i1 false)
  ret void
}

; (39 * 4) + (4 pad) + (7 * 8) = 216
; HSA-LABEL: {{^}}test_missing_alignment_size_2_order0:
; HSA: .amdhsa_group_segment_fixed_size 216
define amdgpu_kernel void @test_missing_alignment_size_2_order0(ptr addrspace(1) %out, ptr addrspace(1) %in) #1 {
  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 4 @lds.missing.align.0, ptr addrspace(1) align 4 %in, i32 160, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 4 %out, ptr addrspace(3) align 4 @lds.missing.align.0, i32 160, i1 false)

  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 8 @lds.missing.align.1, ptr addrspace(1) align 8 %in, i32 56, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 8 %out, ptr addrspace(3) align 8 @lds.missing.align.1, i32 56, i1 false)

  ret void
}

; (39 * 4) + (4 pad) + (7 * 8) = 216
; HSA-LABEL: {{^}}test_missing_alignment_size_2_order1:
; HSA: .amdhsa_group_segment_fixed_size 216
define amdgpu_kernel void @test_missing_alignment_size_2_order1(ptr addrspace(1) %out, ptr addrspace(1) %in) #1 {
  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 8 @lds.missing.align.1, ptr addrspace(1) align 8 %in, i32 56, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 8 %out, ptr addrspace(3) align 8 @lds.missing.align.1, i32 56, i1 false)

  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 4 @lds.missing.align.0, ptr addrspace(1) align 4 %in, i32 160, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 4 %out, ptr addrspace(3) align 4 @lds.missing.align.0, i32 160, i1 false)

  ret void
}

; align 32, 16, 16
; 38 + (10 pad) + 38 + (10 pad) + 38  ( = 134)
; HSA-LABEL: {{^}}test_round_size_3_order0:
; HSA: .amdhsa_group_segment_fixed_size 134
define amdgpu_kernel void @test_round_size_3_order0(ptr addrspace(1) %out, ptr addrspace(1) %in) #1 {
  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 8 @lds.align32.0, ptr addrspace(1) align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 8 %out, ptr addrspace(3) align 8 @lds.align32.0, i32 38, i1 false)

  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 8 @lds.align16.0, ptr addrspace(1) align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 8 %out, ptr addrspace(3) align 8 @lds.align16.0, i32 38, i1 false)

  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 8 @lds.align8.0, ptr addrspace(1) align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 8 %out, ptr addrspace(3) align 8 @lds.align8.0, i32 38, i1 false)

  ret void
}

; align 32, 16, 16
; 38 (+ 10 pad) + 38 + (10 pad) + 38 ( = 134)
; HSA-LABEL: {{^}}test_round_size_3_order1:
; HSA: .amdhsa_group_segment_fixed_size 134
define amdgpu_kernel void @test_round_size_3_order1(ptr addrspace(1) %out, ptr addrspace(1) %in) #1 {
  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 8 @lds.align32.0, ptr addrspace(1) align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 8 %out, ptr addrspace(3) align 8 @lds.align32.0, i32 38, i1 false)

  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 8 @lds.align8.0, ptr addrspace(1) align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 8 %out, ptr addrspace(3) align 8 @lds.align8.0, i32 38, i1 false)

  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 8 @lds.align16.0, ptr addrspace(1) align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 8 %out, ptr addrspace(3) align 8 @lds.align16.0, i32 38, i1 false)

  ret void
}

; align 32, 16, 16
; 38 + (10 pad) + 38 + (10 pad) + 38  ( = 126)
; HSA-LABEL: {{^}}test_round_size_3_order2:
; HSA: .amdhsa_group_segment_fixed_size 134
define amdgpu_kernel void @test_round_size_3_order2(ptr addrspace(1) %out, ptr addrspace(1) %in) #1 {
  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 8 @lds.align16.0, ptr addrspace(1) align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 8 %out, ptr addrspace(3) align 8 @lds.align16.0, i32 38, i1 false)

  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 8 @lds.align32.0, ptr addrspace(1) align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 8 %out, ptr addrspace(3) align 8 @lds.align32.0, i32 38, i1 false)

  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 8 @lds.align8.0, ptr addrspace(1) align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 8 %out, ptr addrspace(3) align 8 @lds.align8.0, i32 38, i1 false)

  ret void
}

; align 32, 16, 16
; 38 + (10 pad) + 38 + (10 pad) + 38 ( = 134)
; HSA-LABEL: {{^}}test_round_size_3_order3:
; HSA: .amdhsa_group_segment_fixed_size 134
define amdgpu_kernel void @test_round_size_3_order3(ptr addrspace(1) %out, ptr addrspace(1) %in) #1 {
  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 8 @lds.align16.0, ptr addrspace(1) align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 8 %out, ptr addrspace(3) align 8 @lds.align16.0, i32 38, i1 false)

  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 8 @lds.align8.0, ptr addrspace(1) align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 8 %out, ptr addrspace(3) align 8 @lds.align8.0, i32 38, i1 false)

  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 8 @lds.align32.0, ptr addrspace(1) align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 8 %out, ptr addrspace(3) align 8 @lds.align32.0, i32 38, i1 false)

  ret void
}

; align 32, 16, 16
; 38 + (10 pad) + 38 + (10 pad) + 38  (= 134)
; HSA-LABEL: {{^}}test_round_size_3_order4:
; HSA: .amdhsa_group_segment_fixed_size 134
define amdgpu_kernel void @test_round_size_3_order4(ptr addrspace(1) %out, ptr addrspace(1) %in) #1 {
  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 8 @lds.align8.0, ptr addrspace(1) align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 8 %out, ptr addrspace(3) align 8 @lds.align8.0, i32 38, i1 false)

  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 8 @lds.align32.0, ptr addrspace(1) align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 8 %out, ptr addrspace(3) align 8 @lds.align32.0, i32 38, i1 false)

  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 8 @lds.align16.0, ptr addrspace(1) align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 8 %out, ptr addrspace(3) align 8 @lds.align16.0, i32 38, i1 false)

  ret void
}

; align 32, 16, 16
; 38 + (10 pad) + 38 + (10 pad) + 38  (= 134)
; HSA-LABEL: {{^}}test_round_size_3_order5:
; HSA: .amdhsa_group_segment_fixed_size 134
define amdgpu_kernel void @test_round_size_3_order5(ptr addrspace(1) %out, ptr addrspace(1) %in) #1 {
  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 8 @lds.align8.0, ptr addrspace(1) align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 8 %out, ptr addrspace(3) align 8 @lds.align8.0, i32 38, i1 false)

  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 8 @lds.align16.0, ptr addrspace(1) align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 8 %out, ptr addrspace(3) align 8 @lds.align16.0, i32 38, i1 false)

  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 8 @lds.align32.0, ptr addrspace(1) align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 8 %out, ptr addrspace(3) align 8 @lds.align32.0, i32 38, i1 false)

  ret void
}

attributes #0 = { argmemonly nounwind }
attributes #1 = { nounwind }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
