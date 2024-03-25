; Modified from: https://github.com/KhronosGroup/SPIRV-LLVM-Translator/test/extensions/INTEL/SPV_INTEL_subgroups/cl_intel_sub_groups.ll

;Source:
;void __kernel test(float2 x, uint c,
;                   read_only image2d_t image_in,
;                   write_only image2d_t image_out,
;                   int2 coord,
;                   __global uint* p,
;                   __global ushort* sp,
;                   __global uchar* cp,
;                   __global ulong* lp) {
;    intel_sub_group_shuffle(x, c);
;    intel_sub_group_shuffle_down(x, x, c);
;    intel_sub_group_shuffle_up(x, x, c);
;    intel_sub_group_shuffle_xor(x, c);
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

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-extensions=SPV_INTEL_subgroups %s -o - | FileCheck %s

; CHECK-ERROR: LLVM ERROR: intel_sub_group_shuffle: the builtin requires the following SPIR-V extension: SPV_INTEL_subgroups

; CHECK-DAG: Capability SubgroupShuffleINTEL
; CHECK-DAG: Capability SubgroupBufferBlockIOINTEL
; CHECK-DAG: Capability SubgroupImageBlockIOINTEL
; CHECK: Extension "SPV_INTEL_subgroups"

; CHECK-SPIRV-LABEL: Function
; CHECK-SPIRV-LABEL: Label

; CHECK: SubgroupShuffleINTEL
; CHECK: SubgroupShuffleDownINTEL
; CHECK: SubgroupShuffleUpINTEL
; CHECK: SubgroupShuffleXorINTEL

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

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

%opencl.image2d_ro_t = type opaque
%opencl.image2d_wo_t = type opaque

; Function Attrs: convergent nounwind
define spir_kernel void @test(<2 x float> %x, i32 %c, ptr addrspace(1) %image_in, ptr addrspace(1) %image_out, <2 x i32> %coord, ptr addrspace(1) %p, ptr addrspace(1) %sp, ptr addrspace(1) %cp, ptr addrspace(1) %lp) local_unnamed_addr #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 !kernel_arg_name !6 {
entry:
  %call = tail call spir_func <2 x float> @_Z23intel_sub_group_shuffleDv2_fj(<2 x float> %x, i32 %c) #2
  %call1 = tail call spir_func <2 x float> @_Z28intel_sub_group_shuffle_downDv2_fS_j(<2 x float> %x, <2 x float> %x, i32 %c) #2
  %call2 = tail call spir_func <2 x float> @_Z26intel_sub_group_shuffle_upDv2_fS_j(<2 x float> %x, <2 x float> %x, i32 %c) #2
  %call3 = tail call spir_func <2 x float> @_Z27intel_sub_group_shuffle_xorDv2_fj(<2 x float> %x, i32 %c) #2

  %call4 = tail call spir_func <2 x i32> @_Z27intel_sub_group_block_read214ocl_image2d_roDv2_i(ptr addrspace(1) %image_in, <2 x i32> %coord) #2
  tail call spir_func void @_Z28intel_sub_group_block_write214ocl_image2d_woDv2_iDv2_j(ptr addrspace(1) %image_out, <2 x i32> %coord, <2 x i32> %call4) #2
  %call5 = tail call spir_func <2 x i32> @_Z27intel_sub_group_block_read2PU3AS1Kj(ptr addrspace(1) %p) #2
  tail call spir_func void @_Z28intel_sub_group_block_write2PU3AS1jDv2_j(ptr addrspace(1) %p, <2 x i32> %call5) #2

  %call6 = tail call spir_func <2 x i16> @_Z30intel_sub_group_block_read_us214ocl_image2d_roDv2_i(ptr addrspace(1) %image_in, <2 x i32> %coord) #2
  tail call spir_func void @_Z31intel_sub_group_block_write_us214ocl_image2d_woDv2_iDv2_t(ptr addrspace(1) %image_out, <2 x i32> %coord, <2 x i16> %call6) #2
  %call7 = tail call spir_func <2 x i16> @_Z30intel_sub_group_block_read_us2PU3AS1Kt(ptr addrspace(1) %sp) #2
  tail call spir_func void @_Z31intel_sub_group_block_write_us2PU3AS1tDv2_t(ptr addrspace(1) %sp, <2 x i16> %call7) #2

  %call8 = tail call spir_func <2 x i8> @_Z30intel_sub_group_block_read_uc214ocl_image2d_roDv2_i(ptr addrspace(1) %image_in, <2 x i32> %coord) #2
  tail call spir_func void @_Z31intel_sub_group_block_write_uc214ocl_image2d_woDv2_iDv2_h(ptr addrspace(1) %image_out, <2 x i32> %coord, <2 x i8> %call8) #2
  %call9 = tail call spir_func <2 x i8> @_Z30intel_sub_group_block_read_uc2PU3AS1Kh(ptr addrspace(1) %cp) #2
  tail call spir_func void @_Z31intel_sub_group_block_write_uc2PU3AS1hDv2_h(ptr addrspace(1) %cp, <2 x i8> %call9) #2

  %call10 = tail call spir_func <2 x i64> @_Z30intel_sub_group_block_read_ul214ocl_image2d_roDv2_i(ptr addrspace(1) %image_in, <2 x i32> %coord) #2
  tail call spir_func void @_Z31intel_sub_group_block_write_ul214ocl_image2d_woDv2_iDv2_m(ptr addrspace(1) %image_out, <2 x i32> %coord, <2 x i64> %call10) #2
  %call11 = tail call spir_func <2 x i64> @_Z30intel_sub_group_block_read_ul2PU3AS1Km(ptr addrspace(1) %lp) #2
  tail call spir_func void @_Z31intel_sub_group_block_write_ul2PU3AS1mDv2_m(ptr addrspace(1) %lp, <2 x i64> %call11) #2

  ret void
}

; Function Attrs: convergent
declare spir_func <2 x float> @_Z23intel_sub_group_shuffleDv2_fj(<2 x float>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func <2 x float> @_Z28intel_sub_group_shuffle_downDv2_fS_j(<2 x float>, <2 x float>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func <2 x float> @_Z26intel_sub_group_shuffle_upDv2_fS_j(<2 x float>, <2 x float>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func <2 x float> @_Z27intel_sub_group_shuffle_xorDv2_fj(<2 x float>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func <2 x i32> @_Z27intel_sub_group_block_read214ocl_image2d_roDv2_i(ptr addrspace(1), <2 x i32>) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func void @_Z28intel_sub_group_block_write214ocl_image2d_woDv2_iDv2_j(ptr addrspace(1), <2 x i32>, <2 x i32>) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func <2 x i32> @_Z27intel_sub_group_block_read2PU3AS1Kj(ptr addrspace(1)) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func void @_Z28intel_sub_group_block_write2PU3AS1jDv2_j(ptr addrspace(1), <2 x i32>) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func <2 x i16> @_Z30intel_sub_group_block_read_us214ocl_image2d_roDv2_i(ptr addrspace(1), <2 x i32>) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func void @_Z31intel_sub_group_block_write_us214ocl_image2d_woDv2_iDv2_t(ptr addrspace(1), <2 x i32>, <2 x i16>) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func <2 x i16> @_Z30intel_sub_group_block_read_us2PU3AS1Kt(ptr addrspace(1)) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func void @_Z31intel_sub_group_block_write_us2PU3AS1tDv2_t(ptr addrspace(1), <2 x i16>) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func <2 x i8> @_Z30intel_sub_group_block_read_uc214ocl_image2d_roDv2_i(ptr addrspace(1), <2 x i32>) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func void @_Z31intel_sub_group_block_write_uc214ocl_image2d_woDv2_iDv2_h(ptr addrspace(1), <2 x i32>, <2 x i8>) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func <2 x i8> @_Z30intel_sub_group_block_read_uc2PU3AS1Kh(ptr addrspace(1)) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func void @_Z31intel_sub_group_block_write_uc2PU3AS1hDv2_h(ptr addrspace(1), <2 x i8>) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func <2 x i64> @_Z30intel_sub_group_block_read_ul214ocl_image2d_roDv2_i(ptr addrspace(1), <2 x i32>) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func void @_Z31intel_sub_group_block_write_ul214ocl_image2d_woDv2_iDv2_m(ptr addrspace(1), <2 x i32>, <2 x i64>) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func <2 x i64> @_Z30intel_sub_group_block_read_ul2PU3AS1Km(ptr addrspace(1)) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func void @_Z31intel_sub_group_block_write_ul2PU3AS1mDv2_m(ptr addrspace(1), <2 x i64>) local_unnamed_addr #1

attributes #0 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind }

!opencl.ocl.version = !{!0}
!opencl.spir.version = !{!0}

!0 = !{i32 1, i32 2}
!1 = !{i32 0, i32 0, i32 1, i32 1, i32 0, i32 1, i32 1, i32 1, i32 1}
!2 = !{!"none", !"none", !"read_only", !"write_only", !"none", !"none", !"none", !"none", !"none"}
!3 = !{!"float2", !"uint", !"image2d_t", !"image2d_t", !"int2", !"uint*", !"ushort*", !"uchar*", !"ulong*"}
!4 = !{!"float __attribute__((ext_vector_type(2)))", !"uint", !"image2d_t", !"image2d_t", !"int __attribute__((ext_vector_type(2)))", !"uint*", !"ushort*", !"uchar*", !"ulong*"}
!5 = !{!"", !"", !"", !"", !"", !"", !"", !"", !""}
!6 = !{!"x", !"c", !"image_in", !"image_out", !"coord", !"p", !"sp", !"cp", !"lp"}
