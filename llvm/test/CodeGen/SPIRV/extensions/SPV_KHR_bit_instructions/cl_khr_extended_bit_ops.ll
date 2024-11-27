; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s --spirv-ext=+SPV_KHR_bit_instructions -o - | FileCheck %s --check-prefix=CHECK-EXTENSION
; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-NO-EXTENSION
; ModuleID = 'cl_khr_extended_bit_ops.cl.tmp.bc'
source_filename = "cl_khr_extended_bit_ops.cl"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir-unknown-unknown"

; CHECK-EXTENSION: Capability BitInstructions
; CHECK-EXTENSION: Extension "SPV_KHR_bit_instructions"
; CHECK-NO-EXTENSION: LLVM ERROR: bitfield_insert: the builtin requires the following SPIR-V extension: SPV_KHR_bit_instructions

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase:]] = OpFunctionParameter %[[#]] 
; CHECK-EXTENSION: %[[#insertinsert:]] = OpFunctionParameter %[[#]] 
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase]] %[[#insertinsert]] 
; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(argmem: write)
define dso_local spir_kernel void @testInsert(<2 x i32> noundef %b, <2 x i32> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 8 %res) local_unnamed_addr #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !4 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 !kernel_arg_host_accessible !7 !kernel_arg_pipe_depth !8 !kernel_arg_pipe_io !6 !kernel_arg_buffer_location !6 {
entry:
  %call = tail call spir_func <2 x i32> @_Z15bitfield_insertDv2_iS_jj(<2 x i32> noundef %b, <2 x i32> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <2 x i32> %call, ptr addrspace(1) %res, align 8, !tbaa !9
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func <2 x i32> @_Z15bitfield_insertDv2_iS_jj(<2 x i32> noundef, <2 x i32> noundef, i32 noundef, i32 noundef) local_unnamed_addr #1

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]] 
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]] 
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(argmem: write)
define dso_local spir_kernel void @testExtractS(i16 noundef signext %b, i16 noundef zeroext %bu, ptr addrspace(1) nocapture noundef writeonly align 2 %res) local_unnamed_addr #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !12 !kernel_arg_base_type !12 !kernel_arg_type_qual !6 !kernel_arg_host_accessible !7 !kernel_arg_pipe_depth !8 !kernel_arg_pipe_io !6 !kernel_arg_buffer_location !6 {
entry:
  %call = tail call spir_func signext i16 @_Z23bitfield_extract_signedsjj(i16 noundef signext %b, i32 noundef 5, i32 noundef 4) #2
  %call1 = tail call spir_func signext i16 @_Z23bitfield_extract_signedtjj(i16 noundef zeroext %bu, i32 noundef 5, i32 noundef 4) #2
  %add = add i16 %call1, %call
  store i16 %add, ptr addrspace(1) %res, align 2, !tbaa !13
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func signext i16 @_Z23bitfield_extract_signedsjj(i16 noundef signext, i32 noundef, i32 noundef) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func signext i16 @_Z23bitfield_extract_signedtjj(i16 noundef zeroext, i32 noundef, i32 noundef) local_unnamed_addr #1

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]] 
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]] 
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(argmem: write)
define dso_local spir_kernel void @testExtractU(<8 x i8> noundef %b, <8 x i8> noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 8 %res) local_unnamed_addr #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !15 !kernel_arg_base_type !16 !kernel_arg_type_qual !6 !kernel_arg_host_accessible !7 !kernel_arg_pipe_depth !8 !kernel_arg_pipe_io !6 !kernel_arg_buffer_location !6 {
entry:
  %call = tail call spir_func <8 x i8> @_Z25bitfield_extract_unsignedDv8_cjj(<8 x i8> noundef %b, i32 noundef 3, i32 noundef 4) #2
  %call1 = tail call spir_func <8 x i8> @_Z25bitfield_extract_unsignedDv8_hjj(<8 x i8> noundef %bu, i32 noundef 3, i32 noundef 4) #2
  %add = add <8 x i8> %call1, %call
  store <8 x i8> %add, ptr addrspace(1) %res, align 8, !tbaa !9
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func <8 x i8> @_Z25bitfield_extract_unsignedDv8_cjj(<8 x i8> noundef, i32 noundef, i32 noundef) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func <8 x i8> @_Z25bitfield_extract_unsignedDv8_hjj(<8 x i8> noundef, i32 noundef, i32 noundef) local_unnamed_addr #1

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]] 
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(argmem: write)
define dso_local spir_kernel void @testBitReverse(<4 x i64> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 32 %res) local_unnamed_addr #0 !kernel_arg_addr_space !17 !kernel_arg_access_qual !18 !kernel_arg_type !19 !kernel_arg_base_type !20 !kernel_arg_type_qual !21 !kernel_arg_host_accessible !22 !kernel_arg_pipe_depth !23 !kernel_arg_pipe_io !21 !kernel_arg_buffer_location !21 {
entry:
  %call = tail call spir_func <4 x i64> @_Z11bit_reverseDv4_m(<4 x i64> noundef %b) #2
  store <4 x i64> %call, ptr addrspace(1) %res, align 32, !tbaa !9
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func <4 x i64> @_Z11bit_reverseDv4_m(<4 x i64> noundef) local_unnamed_addr #1

attributes #0 = { convergent mustprogress nofree norecurse nounwind willreturn memory(argmem: write) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" }
attributes #1 = { convergent mustprogress nofree nounwind willreturn memory(none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nounwind willreturn memory(none) }

!opencl.ocl.version = !{!0}
!opencl.spir.version = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 2, i32 0}
!1 = !{!"Intel(R) oneAPI DPC++/C++ Compiler 2025.1.0 (2025.x.0.YYYYMMDD)"}
!2 = !{i32 0, i32 0, i32 1}
!3 = !{!"none", !"none", !"none"}
!4 = !{!"int2", !"int2", !"int2*"}
!5 = !{!"int __attribute__((ext_vector_type(2)))", !"int __attribute__((ext_vector_type(2)))", !"int __attribute__((ext_vector_type(2)))*"}
!6 = !{!"", !"", !""}
!7 = !{i1 false, i1 false, i1 false}
!8 = !{i32 0, i32 0, i32 0}
!9 = !{!10, !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = !{!"short", !"ushort", !"short*"}
!13 = !{!14, !14, i64 0}
!14 = !{!"short", !10, i64 0}
!15 = !{!"char8", !"uchar8", !"uchar8*"}
!16 = !{!"char __attribute__((ext_vector_type(8)))", !"uchar __attribute__((ext_vector_type(8)))", !"uchar __attribute__((ext_vector_type(8)))*"}
!17 = !{i32 0, i32 1}
!18 = !{!"none", !"none"}
!19 = !{!"ulong4", !"ulong4*"}
!20 = !{!"ulong __attribute__((ext_vector_type(4)))", !"ulong __attribute__((ext_vector_type(4)))*"}
!21 = !{!"", !""}
!22 = !{i1 false, i1 false}
!23 = !{i32 0, i32 0}