; Modified from: https://github.com/KhronosGroup/SPIRV-LLVM-Translator/test/extensions/INTEL/SPV_INTEL_subgroups/cl_intel_sub_groups.ll

;Source:
;void __kernel test(float2 x, uint c,
;                   read_only image2d_t image_in,
;                   write_only image2d_t image_out,
;                   int2 coord,
;                   __local uint* p,
;                   __local ushort* sp,
;                   __local uchar* cp,
;                   __local ulong* lp) {
;
;    uint2 ui2 = intel_sub_group_block_read2(image_in, coord);
;    intel_sub_group_block_write2(image_out, coord, ui2);
;    ui2 = intel_sub_group_block_read2(p);
;    intel_sub_group_block_write2(p, ui2);
;
;    ushort2 us2 = intel_sub_group_block_read_us2(image_in, coord);
;    intel_sub_group_block_write_us2(image_out, coord, us2);
;    us2 = intel_sub_group_block_read_us2(sp);
;    intel_sub_group_block_write_us2(sp, us2);
;
;    uchar2 uc2 = intel_sub_group_block_read_uc2(image_in, coord);
;    intel_sub_group_block_write_uc2(image_out, coord, uc2);
;    uc2 = intel_sub_group_block_read_uc2(cp);
;    intel_sub_group_block_write_uc2(cp, uc2);
;
;    ulong2 ul2 = intel_sub_group_block_read_ul2(image_in, coord);
;    intel_sub_group_block_write_ul2(image_out, coord, ul2);
;    ul2 = intel_sub_group_block_read_ul2(lp);
;    intel_sub_group_block_write_ul2(lp, ul2);
;}

; RUN: not llc -O0 -mtriple=spirv32-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_subgroups %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_subgroups %s -o - -filetype=obj | spirv-val %}

; CHECK-ERROR: LLVM ERROR: intel_sub_group_block_read2: the builtin requires the following SPIR-V extension: SPV_INTEL_subgroups

; CHECK-DAG: Capability SubgroupBufferBlockIOINTEL
; CHECK-DAG: Capability SubgroupImageBlockIOINTEL
; CHECK: Extension "SPV_INTEL_subgroups"

; CHECK-SPIRV-LABEL: Function
; CHECK-SPIRV-LABEL: Label

; CHECK: SubgroupImageBlockReadINTEL
; CHECK: SubgroupImageBlockWriteINTEL
; CHECK: SubgroupBlockReadINTEL
; CHECK: SubgroupBlockWriteINTEL

; CHECK: SubgroupImageBlockReadINTEL
; CHECK: SubgroupImageBlockWriteINTEL
; CHECK: SubgroupBlockReadINTEL
; CHECK: SubgroupBlockWriteINTEL

; CHECK: SubgroupImageBlockReadINTEL
; CHECK: SubgroupImageBlockWriteINTEL
; CHECK: SubgroupBlockReadINTEL
; CHECK: SubgroupBlockWriteINTEL

; CHECK: SubgroupImageBlockReadINTEL
; CHECK: SubgroupImageBlockWriteINTEL
; CHECK: SubgroupBlockReadINTEL
; CHECK: SubgroupBlockWriteINTEL

; CHECK-SPIRV-LABEL: Return

%opencl.image2d_ro_t = type opaque
%opencl.image2d_wo_t = type opaque

; Function Attrs: convergent nounwind
define spir_kernel void @test(<2 x float> %x, i32 %c, ptr addrspace(3) %image_in, ptr addrspace(3) %image_out, <2 x i32> %coord, ptr addrspace(3) %p, ptr addrspace(3) %sp, ptr addrspace(3) %cp, ptr addrspace(3) %lp) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 !kernel_arg_name !6 {
entry:
  %call4 = tail call spir_func <2 x i32> @_Z27intel_sub_group_block_read214ocl_image2d_roDv2_i(ptr addrspace(3) %image_in, <2 x i32> %coord)
  tail call spir_func void @_Z28intel_sub_group_block_write214ocl_image2d_woDv2_iDv2_j(ptr addrspace(3) %image_out, <2 x i32> %coord, <2 x i32> %call4)
  %call5 = tail call spir_func <2 x i32> @_Z27intel_sub_group_block_read2PU3AS1Kj(ptr addrspace(3) %p)
  tail call spir_func void @_Z28intel_sub_group_block_write2PU3AS1jDv2_j(ptr addrspace(3) %p, <2 x i32> %call5)

  %call6 = tail call spir_func <2 x i16> @_Z30intel_sub_group_block_read_us214ocl_image2d_roDv2_i(ptr addrspace(3) %image_in, <2 x i32> %coord)
  tail call spir_func void @_Z31intel_sub_group_block_write_us214ocl_image2d_woDv2_iDv2_t(ptr addrspace(3) %image_out, <2 x i32> %coord, <2 x i16> %call6)
  %call7 = tail call spir_func <2 x i16> @_Z30intel_sub_group_block_read_us2PU3AS1Kt(ptr addrspace(3) %sp)
  tail call spir_func void @_Z31intel_sub_group_block_write_us2PU3AS1tDv2_t(ptr addrspace(3) %sp, <2 x i16> %call7)

  %call8 = tail call spir_func <2 x i8> @_Z30intel_sub_group_block_read_uc214ocl_image2d_roDv2_i(ptr addrspace(3) %image_in, <2 x i32> %coord)
  tail call spir_func void @_Z31intel_sub_group_block_write_uc214ocl_image2d_woDv2_iDv2_h(ptr addrspace(3) %image_out, <2 x i32> %coord, <2 x i8> %call8)
  %call9 = tail call spir_func <2 x i8> @_Z30intel_sub_group_block_read_uc2PU3AS1Kh(ptr addrspace(3) %cp)
  tail call spir_func void @_Z31intel_sub_group_block_write_uc2PU3AS1hDv2_h(ptr addrspace(3) %cp, <2 x i8> %call9)

  %call10 = tail call spir_func <2 x i64> @_Z30intel_sub_group_block_read_ul214ocl_image2d_roDv2_i(ptr addrspace(3) %image_in, <2 x i32> %coord)
  tail call spir_func void @_Z31intel_sub_group_block_write_ul214ocl_image2d_woDv2_iDv2_m(ptr addrspace(3) %image_out, <2 x i32> %coord, <2 x i64> %call10)
  %call11 = tail call spir_func <2 x i64> @_Z30intel_sub_group_block_read_ul2PU3AS1Km(ptr addrspace(3) %lp)
  tail call spir_func void @_Z31intel_sub_group_block_write_ul2PU3AS1mDv2_m(ptr addrspace(3) %lp, <2 x i64> %call11)

  ret void
}

; Function Attrs: convergent
declare spir_func <2 x i32> @_Z27intel_sub_group_block_read214ocl_image2d_roDv2_i(ptr addrspace(3), <2 x i32>)

; Function Attrs: convergent
declare spir_func void @_Z28intel_sub_group_block_write214ocl_image2d_woDv2_iDv2_j(ptr addrspace(3), <2 x i32>, <2 x i32>)

; Function Attrs: convergent
declare spir_func <2 x i32> @_Z27intel_sub_group_block_read2PU3AS1Kj(ptr addrspace(3))

; Function Attrs: convergent
declare spir_func void @_Z28intel_sub_group_block_write2PU3AS1jDv2_j(ptr addrspace(3), <2 x i32>)

; Function Attrs: convergent
declare spir_func <2 x i16> @_Z30intel_sub_group_block_read_us214ocl_image2d_roDv2_i(ptr addrspace(3), <2 x i32>)

; Function Attrs: convergent
declare spir_func void @_Z31intel_sub_group_block_write_us214ocl_image2d_woDv2_iDv2_t(ptr addrspace(3), <2 x i32>, <2 x i16>)

; Function Attrs: convergent
declare spir_func <2 x i16> @_Z30intel_sub_group_block_read_us2PU3AS1Kt(ptr addrspace(3))

; Function Attrs: convergent
declare spir_func void @_Z31intel_sub_group_block_write_us2PU3AS1tDv2_t(ptr addrspace(3), <2 x i16>)

; Function Attrs: convergent
declare spir_func <2 x i8> @_Z30intel_sub_group_block_read_uc214ocl_image2d_roDv2_i(ptr addrspace(3), <2 x i32>)

; Function Attrs: convergent
declare spir_func void @_Z31intel_sub_group_block_write_uc214ocl_image2d_woDv2_iDv2_h(ptr addrspace(3), <2 x i32>, <2 x i8>)

; Function Attrs: convergent
declare spir_func <2 x i8> @_Z30intel_sub_group_block_read_uc2PU3AS1Kh(ptr addrspace(3))

; Function Attrs: convergent
declare spir_func void @_Z31intel_sub_group_block_write_uc2PU3AS1hDv2_h(ptr addrspace(3), <2 x i8>)

; Function Attrs: convergent
declare spir_func <2 x i64> @_Z30intel_sub_group_block_read_ul214ocl_image2d_roDv2_i(ptr addrspace(3), <2 x i32>)

; Function Attrs: convergent
declare spir_func void @_Z31intel_sub_group_block_write_ul214ocl_image2d_woDv2_iDv2_m(ptr addrspace(3), <2 x i32>, <2 x i64>)

; Function Attrs: convergent
declare spir_func <2 x i64> @_Z30intel_sub_group_block_read_ul2PU3AS1Km(ptr addrspace(3))

; Function Attrs: convergent
declare spir_func void @_Z31intel_sub_group_block_write_ul2PU3AS1mDv2_m(ptr addrspace(3), <2 x i64>)

!opencl.ocl.version = !{!0}
!opencl.spir.version = !{!0}

!0 = !{i32 1, i32 2}
!1 = !{i32 0, i32 0, i32 1, i32 1, i32 0, i32 1, i32 1, i32 1, i32 1}
!2 = !{!"none", !"none", !"read_only", !"write_only", !"none", !"none", !"none", !"none", !"none"}
!3 = !{!"float2", !"uint", !"image2d_t", !"image2d_t", !"int2", !"uint*", !"ushort*", !"uchar*", !"ulong*"}
!4 = !{!"float __attribute__((ext_vector_type(2)))", !"uint", !"image2d_t", !"image2d_t", !"int __attribute__((ext_vector_type(2)))", !"uint*", !"ushort*", !"uchar*", !"ulong*"}
!5 = !{!"", !"", !"", !"", !"", !"", !"", !"", !""}
!6 = !{!"x", !"c", !"image_in", !"image_out", !"coord", !"p", !"sp", !"cp", !"lp"}
