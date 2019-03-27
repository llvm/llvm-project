;Source:
;void __kernel test(float2 x, uint c,
;                   read_only image2d_t image_in,
;                   write_only image2d_t image_out,
;                   int2 coord,
;                   __global uint* p,
;                   __global ushort* sp) {
;    intel_sub_group_shuffle(x, c);
;    intel_sub_group_shuffle_down(x, x, c);
;    intel_sub_group_shuffle_up(x, x, c);
;    intel_sub_group_shuffle_xor(x, c);
;
;    uint2 ui2 = intel_sub_group_block_read2(image_in, coord);
;    intel_sub_group_block_write2(p, ui2);
;    intel_sub_group_block_write2(image_out, coord, ui2);
;
;    ushort2 us2 = intel_sub_group_block_read_us2(sp);
;    intel_sub_group_block_write_us2(sp, us2);
;    intel_sub_group_block_write_us2(image_out, coord, us2);
;}

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o - -spirv-text | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Capability SubgroupShuffleINTEL
; CHECK-SPIRV: Capability SubgroupBufferBlockIOINTEL
; CHECK-SPIRV: Capability SubgroupImageBlockIOINTEL
; CHECK-SPIRV: Extension "cl_intel_subgroups"
; CHECK-SPIRV: Extension "cl_intel_subgroups_short"

; CHECK-SPIRV: SubgroupShuffleINTEL
; CHECK-SPIRV: SubgroupShuffleDownINTEL
; CHECK-SPIRV: SubgroupShuffleUpINTEL
; CHECK-SPIRV: SubgroupShuffleXorINTEL

; CHECK-SPIRV: SubgroupImageBlockReadINTEL
; CHECK-SPIRV: SubgroupBlockWriteINTEL
; CHECK-SPIRV: SubgroupImageBlockWriteINTEL

; CHECK-SPIRV: SubgroupBlockReadINTEL
; CHECK-SPIRV: SubgroupBlockWriteINTEL
; CHECK-SPIRV: SubgroupImageBlockWriteINTEL

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

%opencl.image2d_ro_t = type opaque
%opencl.image2d_wo_t = type opaque

; Function Attrs: nounwind
define spir_kernel void @test(<2 x float> %x, i32 %c, %opencl.image2d_ro_t addrspace(1)* %image_in, %opencl.image2d_wo_t addrspace(1)* %image_out, <2 x i32> %coord, i32 addrspace(1)* %p, i16 addrspace(1)* %sp) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %call = tail call spir_func <2 x float> @_Z23intel_sub_group_shuffleDv2_fj(<2 x float> %x, i32 %c) #3
  %call1 = tail call spir_func <2 x float> @_Z28intel_sub_group_shuffle_downDv2_fDv2_fj(<2 x float> %x, <2 x float> %x, i32 %c) #3
  %call2 = tail call spir_func <2 x float> @_Z26intel_sub_group_shuffle_upDv2_fDv2_fj(<2 x float> %x, <2 x float> %x, i32 %c) #3
  %call3 = tail call spir_func <2 x float> @_Z27intel_sub_group_shuffle_xorDv2_fj(<2 x float> %x, i32 %c) #3
; CHECK-LLVM: call spir_func <2 x float> @_Z23intel_sub_group_shuffle{{.*}}(<2 x float> %x, i32 %c)
; CHECK-LLVM: call spir_func <2 x float> @_Z28intel_sub_group_shuffle_down{{.*}}(<2 x float> %x, <2 x float> %x, i32 %c)
; CHECK-LLVM: call spir_func <2 x float> @_Z26intel_sub_group_shuffle_up{{.*}}(<2 x float> %x, <2 x float> %x, i32 %c)
; CHECK-LLVM: call spir_func <2 x float> @_Z27intel_sub_group_shuffle_xor{{.*}}(<2 x float> %x, i32 %c)

  %call4 = tail call spir_func <2 x i32> @_Z27intel_sub_group_block_read214ocl_image2d_roDv2_i(%opencl.image2d_ro_t addrspace(1)* %image_in, <2 x i32> %coord) #4
  tail call spir_func void @_Z28intel_sub_group_block_write2PU3AS1jDv2_j(i32 addrspace(1)* %p, <2 x i32> %call4) #3
  tail call spir_func void @_Z28intel_sub_group_block_write214ocl_image2d_woDv2_iDv2_j(%opencl.image2d_wo_t addrspace(1)* %image_out, <2 x i32> %coord, <2 x i32> %call4) #3
; CHECK-LLVM: call spir_func <2 x i32> @_Z27intel_sub_group_block_read2{{.*}}(%opencl.image2d_ro_t addrspace(1)* %image_in, <2 x i32> %coord)
; CHECK-LLVM: call spir_func void @_Z28intel_sub_group_block_write2{{.*}}(i32 addrspace(1)* %p, <2 x i32> %call4)
; CHECK-LLVM: call spir_func void @_Z28intel_sub_group_block_write2{{.*}}(%opencl.image2d_wo_t addrspace(1)* %image_out, <2 x i32> %coord, <2 x i32> %call4)

  %call5 = tail call spir_func <2 x i16> @_Z30intel_sub_group_block_read_us2PKU3AS1t(i16 addrspace(1)* %sp) #4
  tail call spir_func void @_Z31intel_sub_group_block_write_us2PU3AS1tDv2_t(i16 addrspace(1)* %sp, <2 x i16> %call5) #3
  tail call spir_func void @_Z31intel_sub_group_block_write_us214ocl_image2d_woDv2_iDv2_t(%opencl.image2d_wo_t addrspace(1)* %image_out, <2 x i32> %coord, <2 x i16> %call5) #3
; CHECK-LLVM: call spir_func <2 x i16> @_Z30intel_sub_group_block_read_us2{{.*}}(i16 addrspace(1)* %sp)
; CHECK-LLVM: call spir_func void @_Z31intel_sub_group_block_write_us2{{.*}}(i16 addrspace(1)* %sp, <2 x i16> %call5)
; CHECK-LLVM: call spir_func void @_Z31intel_sub_group_block_write_us2{{.*}}(%opencl.image2d_wo_t addrspace(1)* %image_out, <2 x i32> %coord, <2 x i16> %call5)
  ret void
}

declare spir_func <2 x float> @_Z23intel_sub_group_shuffleDv2_fj(<2 x float>, i32) #1

declare spir_func <2 x float> @_Z28intel_sub_group_shuffle_downDv2_fDv2_fj(<2 x float>, <2 x float>, i32) #1

declare spir_func <2 x float> @_Z26intel_sub_group_shuffle_upDv2_fDv2_fj(<2 x float>, <2 x float>, i32) #1

declare spir_func <2 x float> @_Z27intel_sub_group_shuffle_xorDv2_fj(<2 x float>, i32) #1

; Function Attrs: nounwind readonly
declare spir_func <2 x i32> @_Z27intel_sub_group_block_read214ocl_image2d_roDv2_i(%opencl.image2d_ro_t addrspace(1)*, <2 x i32>) #2

declare spir_func void @_Z28intel_sub_group_block_write2PU3AS1jDv2_j(i32 addrspace(1)*, <2 x i32>) #1

declare spir_func void @_Z28intel_sub_group_block_write214ocl_image2d_woDv2_iDv2_j(%opencl.image2d_wo_t addrspace(1)*, <2 x i32>, <2 x i32>) #1

; Function Attrs: nounwind readonly
declare spir_func <2 x i16> @_Z30intel_sub_group_block_read_us2PKU3AS1t(i16 addrspace(1)*) #2

declare spir_func void @_Z31intel_sub_group_block_write_us2PU3AS1tDv2_t(i16 addrspace(1)*, <2 x i16>) #1

declare spir_func void @_Z31intel_sub_group_block_write_us214ocl_image2d_woDv2_iDv2_t(%opencl.image2d_wo_t addrspace(1)*, <2 x i32>, <2 x i16>) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readonly "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }
attributes #4 = { nounwind readonly }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!9}
!opencl.compiler.options = !{!9}

!1 = !{i32 0, i32 0, i32 1, i32 1, i32 0, i32 1, i32 1}
!2 = !{!"none", !"none", !"read_only", !"write_only", !"none", !"none", !"none"}
!3 = !{!"float2", !"uint", !"__read_only image2d_t", !"__write_only image2d_t", !"int2", !"uint*", !"ushort*"}
!4 = !{!"float2", !"uint", !"__read_only image2d_t", !"__write_only image2d_t", !"int2", !"uint*", !"ushort*"}
!5 = !{!"", !"", !"", !"", !"", !"", !""}
!6 = !{i32 1, i32 2}
!7 = !{i32 2, i32 0}
!8 = !{!"cl_intel_subgroups", !"cl_intel_subgroups_short"}
!9 = !{}
