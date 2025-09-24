; RUN: llc -mtriple=amdgcn-amd-amdhsa < %s

%"struct.rocprim::ROCPRIM_400000_NS::detail::raw_storage" = type { [1056 x i8] }
%"struct.rocprim::ROCPRIM_400000_NS::detail::raw_storage.17" = type { [4 x i8] }
%"struct.rocprim::ROCPRIM_400000_NS::detail::raw_storage.43" = type { [16 x i8] }

@_ZZN7hipcomp30do_cascaded_compression_kernelIhmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE24shared_element_storage_0 = addrspace(3) global [1026 x i32] undef
@_ZZN7hipcomp30do_cascaded_compression_kernelIhmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE24shared_element_storage_1 = addrspace(3) global [1026 x i32] undef
@_ZZN7hipcomp30do_cascaded_compression_kernelIhmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE19shared_count_buffer = external addrspace(3) global [2048 x i32]
@_ZZN7hipcomp30do_cascaded_compression_kernelIhmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE17shared_tmp_buffer = addrspace(3) global [2050 x i32] undef
@_ZZN7hipcomp30do_cascaded_compression_kernelIhmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE14chunk_metadata = addrspace(3) global [16 x i32] undef
@_ZZN7hipcomp18block_rle_compressIhmtLi128EEEvPKT_T0_PS1_PT1_PS4_S7_E12temp_storage = external addrspace(3) global %"struct.rocprim::ROCPRIM_400000_NS::detail::raw_storage"
@_ZZN7hipcomp16get_for_bitwidthItmLi128EEEvPKT_T0_PS1_PjE12temp_storage = addrspace(3) global %"struct.rocprim::ROCPRIM_400000_NS::detail::raw_storage.17" undef
@_ZZN7hipcomp30do_cascaded_compression_kernelItmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE24shared_element_storage_0 = addrspace(3) global [1026 x i32] undef
@_ZZN7hipcomp30do_cascaded_compression_kernelItmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE24shared_element_storage_1 = addrspace(3) global [1026 x i32] undef
@_ZZN7hipcomp30do_cascaded_compression_kernelItmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE19shared_count_buffer = external addrspace(3) global [1024 x i32]
@_ZZN7hipcomp30do_cascaded_compression_kernelItmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE17shared_tmp_buffer = addrspace(3) global [1026 x i32] undef
@_ZZN7hipcomp30do_cascaded_compression_kernelItmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE14chunk_metadata = external addrspace(3) global [16 x i32]
@_ZZN7hipcomp18block_rle_compressItmtLi128EEEvPKT_T0_PS1_PT1_PS4_S7_E12temp_storage = external addrspace(3) global %"struct.rocprim::ROCPRIM_400000_NS::detail::raw_storage"
@_ZZ23HlifCompressBatchKernelILi32EN7hipcomp25cascaded_compress_wrapperIjmLi128ELi4096EEERK28hipcompBatchedCascadedOpts_tLi1EENSt9enable_ifIXsr3std10is_base_ofI21hlif_compress_wrapperT0_EE5valueEvE4typeE12CompressArgsT1_E13output_status = external addrspace(3) global [1 x i32]
@_ZZN7hipcomp30do_cascaded_compression_kernelIjmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE24shared_element_storage_0 = addrspace(3) global [1026 x i32] undef
@_ZZN7hipcomp30do_cascaded_compression_kernelIjmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE24shared_element_storage_1 = addrspace(3) global [1026 x i32] undef
@_ZZN7hipcomp30do_cascaded_compression_kernelIjmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE19shared_count_buffer = external addrspace(3) global [512 x i32]
@_ZZN7hipcomp30do_cascaded_compression_kernelIjmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE17shared_tmp_buffer = addrspace(3) global [514 x i32] undef
@_ZZN7hipcomp30do_cascaded_compression_kernelIjmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE14chunk_metadata = external addrspace(3) global [16 x i32]
@_ZZN7hipcomp18block_rle_compressIjmtLi128EEEvPKT_T0_PS1_PT1_PS4_S7_E12temp_storage = external addrspace(3) global %"struct.rocprim::ROCPRIM_400000_NS::detail::raw_storage"
@_ZZN7hipcomp30do_cascaded_compression_kernelImmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE24shared_element_storage_0 = external addrspace(3) global [514 x i64]
@_ZZN7hipcomp30do_cascaded_compression_kernelImmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE24shared_element_storage_1 = external addrspace(3) global [514 x i64]
@_ZZN7hipcomp30do_cascaded_compression_kernelImmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE19shared_count_buffer = external addrspace(3) global [256 x i32]
@_ZZN7hipcomp30do_cascaded_compression_kernelImmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE17shared_tmp_buffer = external addrspace(3) global [258 x i32]
@_ZZN7hipcomp30do_cascaded_compression_kernelImmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE14chunk_metadata = external addrspace(3) global [16 x i32]
@_ZZN7hipcomp18block_rle_compressImmtLi128EEEvPKT_T0_PS1_PT1_PS4_S7_E12temp_storage = external addrspace(3) global %"struct.rocprim::ROCPRIM_400000_NS::detail::raw_storage"
@_ZZN7hipcomp16get_for_bitwidthImmLi128EEEvPKT_T0_PS1_PjE12temp_storage = external addrspace(3) global %"struct.rocprim::ROCPRIM_400000_NS::detail::raw_storage.43"

define amdgpu_kernel void @_Z23HlifCompressBatchKernelILi32EN7hipcomp25cascaded_compress_wrapperIhmLi128ELi4096EEERK28hipcompBatchedCascadedOpts_tLi1EENSt9enable_ifIXsr3std10is_base_ofI21hlif_compress_wrapperT0_EE5valueEvE4typeE12CompressArgsT1_() {
entry:
  %0 = call i32 @_Z17HlifCompressBatchILi1ERN7hipcomp25cascaded_compress_wrapperIhmLi128ELi4096EEERN18cooperative_groups12thread_blockEEvRK12CompressArgsOT0_OT1_()
  ret void
}

define i32 @_Z17HlifCompressBatchILi1ERN7hipcomp25cascaded_compress_wrapperIhmLi128ELi4096EEERN18cooperative_groups12thread_blockEEvRK12CompressArgsOT0_OT1_() {
entry:
  %0 = call i32 @_ZN7hipcomp25cascaded_compress_wrapperIhmLi128ELi4096EE14compress_chunkEPhPKhmmPm()
  ret i32 %0
}

define i32 @_ZN7hipcomp25cascaded_compress_wrapperIhmLi128ELi4096EE14compress_chunkEPhPKhmmPm() {
entry:
  %0 = call i32 @_ZN7hipcomp30do_cascaded_compression_kernelIhmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_t()
  ret i32 %0
}

define i32 @_ZN7hipcomp30do_cascaded_compression_kernelIhmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_t() {
entry:
  store ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp30do_cascaded_compression_kernelIhmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE24shared_element_storage_0 to ptr), ptr null, align 8
  store ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp30do_cascaded_compression_kernelIhmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE24shared_element_storage_1 to ptr), ptr null, align 8
  %add.ptr = getelementptr i32, ptr getelementptr inbounds (i32, ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp30do_cascaded_compression_kernelIhmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE14chunk_metadata to ptr), i64 1), i64 0
  call void @_ZN7hipcomp18block_rle_compressIhmtLi128EEEvPKT_T0_PS1_PT1_PS4_S7_()
  %call69 = call i32 @_ZN7hipcomp11block_writeItmLi128EEENS_13BlockIOStatusEPKT_T0_PjPKjPS5_S6_b(ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp30do_cascaded_compression_kernelIhmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE19shared_count_buffer to ptr), ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp30do_cascaded_compression_kernelIhmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE17shared_tmp_buffer to ptr))
  ret i32 %call69
}

define void @_ZN7hipcomp18block_rle_compressIhmtLi128EEEvPKT_T0_PS1_PT1_PS4_S7_() {
entry:
  store ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp18block_rle_compressIhmtLi128EEEvPKT_T0_PS1_PT1_PS4_S7_E12temp_storage to ptr), ptr null, align 8
  ret void
}

define i32 @_ZN7hipcomp11block_writeItmLi128EEENS_13BlockIOStatusEPKT_T0_PjPKjPS5_S6_b(ptr %input, ptr %temp_storage) {
entry:
  call void @_ZN7hipcomp13block_bitpackItmLi128EEEvPKT_T0_PjPS4_()
  ret i32 0
}

define void @_ZN7hipcomp13block_bitpackItmLi128EEEvPKT_T0_PjPS4_() {
entry:
  call void @_ZN7hipcomp16get_for_bitwidthItmLi128EEEvPKT_T0_PS1_Pj()
  ret void
}

define void @_ZN7hipcomp16get_for_bitwidthItmLi128EEEvPKT_T0_PS1_Pj() {
entry:
  store ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp16get_for_bitwidthItmLi128EEEvPKT_T0_PS1_PjE12temp_storage to ptr), ptr null, align 8
  ret void
}

define amdgpu_kernel void @_Z23HlifCompressBatchKernelILi32EN7hipcomp25cascaded_compress_wrapperItmLi128ELi4096EEERK28hipcompBatchedCascadedOpts_tLi1EENSt9enable_ifIXsr3std10is_base_ofI21hlif_compress_wrapperT0_EE5valueEvE4typeE12CompressArgsT1_() {
entry:
  %0 = call i32 @_Z17HlifCompressBatchILi1ERN7hipcomp25cascaded_compress_wrapperItmLi128ELi4096EEERN18cooperative_groups12thread_blockEEvRK12CompressArgsOT0_OT1_()
  ret void
}

define i32 @_Z17HlifCompressBatchILi1ERN7hipcomp25cascaded_compress_wrapperItmLi128ELi4096EEERN18cooperative_groups12thread_blockEEvRK12CompressArgsOT0_OT1_() {
entry:
  %0 = call i32 @_ZN7hipcomp25cascaded_compress_wrapperItmLi128ELi4096EE14compress_chunkEPhPKhmmPm()
  ret i32 %0
}

define i32 @_ZN7hipcomp25cascaded_compress_wrapperItmLi128ELi4096EE14compress_chunkEPhPKhmmPm() {
entry:
  %0 = call i32 @_ZN7hipcomp30do_cascaded_compression_kernelItmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_t()
  ret i32 %0
}

define i32 @_ZN7hipcomp30do_cascaded_compression_kernelItmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_t() {
entry:
  store ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp30do_cascaded_compression_kernelItmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE24shared_element_storage_0 to ptr), ptr null, align 8
  store ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp30do_cascaded_compression_kernelItmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE24shared_element_storage_1 to ptr), ptr null, align 8
  %add.ptr = getelementptr i32, ptr getelementptr inbounds (i32, ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp30do_cascaded_compression_kernelItmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE14chunk_metadata to ptr), i64 1), i64 0
  call void @_ZN7hipcomp18block_rle_compressItmtLi128EEEvPKT_T0_PS1_PT1_PS4_S7_()
  %call69 = call i32 @_ZN7hipcomp11block_writeItmLi128EEENS_13BlockIOStatusEPKT_T0_PjPKjPS5_S6_b(ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp30do_cascaded_compression_kernelItmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE19shared_count_buffer to ptr), ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp30do_cascaded_compression_kernelItmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE17shared_tmp_buffer to ptr))
  ret i32 %call69
}

define void @_ZN7hipcomp18block_rle_compressItmtLi128EEEvPKT_T0_PS1_PT1_PS4_S7_() {
entry:
  store ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp18block_rle_compressItmtLi128EEEvPKT_T0_PS1_PT1_PS4_S7_E12temp_storage to ptr), ptr null, align 8
  ret void
}

define amdgpu_kernel void @_Z23HlifCompressBatchKernelILi32EN7hipcomp25cascaded_compress_wrapperIjmLi128ELi4096EEERK28hipcompBatchedCascadedOpts_tLi1EENSt9enable_ifIXsr3std10is_base_ofI21hlif_compress_wrapperT0_EE5valueEvE4typeE12CompressArgsT1_() {
entry:
  %arrayidx = getelementptr [1 x i32], ptr addrspacecast (ptr addrspace(3) @_ZZ23HlifCompressBatchKernelILi32EN7hipcomp25cascaded_compress_wrapperIjmLi128ELi4096EEERK28hipcompBatchedCascadedOpts_tLi1EENSt9enable_ifIXsr3std10is_base_ofI21hlif_compress_wrapperT0_EE5valueEvE4typeE12CompressArgsT1_E13output_status to ptr), i64 0, i64 0
  %0 = call i32 @_Z17HlifCompressBatchILi1ERN7hipcomp25cascaded_compress_wrapperIjmLi128ELi4096EEERN18cooperative_groups12thread_blockEEvRK12CompressArgsOT0_OT1_()
  ret void
}

define i32 @_Z17HlifCompressBatchILi1ERN7hipcomp25cascaded_compress_wrapperIjmLi128ELi4096EEERN18cooperative_groups12thread_blockEEvRK12CompressArgsOT0_OT1_() {
entry:
  %0 = call i32 @_ZN7hipcomp25cascaded_compress_wrapperIjmLi128ELi4096EE14compress_chunkEPhPKhmmPm()
  ret i32 %0
}

define i32 @_ZN7hipcomp25cascaded_compress_wrapperIjmLi128ELi4096EE14compress_chunkEPhPKhmmPm() {
entry:
  %0 = call i32 @_ZN7hipcomp30do_cascaded_compression_kernelIjmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_t()
  ret i32 %0
}

define i32 @_ZN7hipcomp30do_cascaded_compression_kernelIjmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_t() {
entry:
  store ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp30do_cascaded_compression_kernelIjmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE24shared_element_storage_0 to ptr), ptr null, align 8
  store ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp30do_cascaded_compression_kernelIjmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE24shared_element_storage_1 to ptr), ptr null, align 8
  %add.ptr = getelementptr i32, ptr getelementptr inbounds (i32, ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp30do_cascaded_compression_kernelIjmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE14chunk_metadata to ptr), i64 1), i64 0
  call void @_ZN7hipcomp18block_rle_compressIjmtLi128EEEvPKT_T0_PS1_PT1_PS4_S7_()
  %call69 = call i32 @_ZN7hipcomp11block_writeItmLi128EEENS_13BlockIOStatusEPKT_T0_PjPKjPS5_S6_b(ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp30do_cascaded_compression_kernelIjmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE19shared_count_buffer to ptr), ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp30do_cascaded_compression_kernelIjmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE17shared_tmp_buffer to ptr))
  ret i32 %call69
}

define void @_ZN7hipcomp18block_rle_compressIjmtLi128EEEvPKT_T0_PS1_PT1_PS4_S7_() {
entry:
  store ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp18block_rle_compressIjmtLi128EEEvPKT_T0_PS1_PT1_PS4_S7_E12temp_storage to ptr), ptr null, align 8
  ret void
}

define amdgpu_kernel void @_Z23HlifCompressBatchKernelILi32EN7hipcomp25cascaded_compress_wrapperImmLi128ELi4096EEERK28hipcompBatchedCascadedOpts_tLi1EENSt9enable_ifIXsr3std10is_base_ofI21hlif_compress_wrapperT0_EE5valueEvE4typeE12CompressArgsT1_() {
entry:
  %0 = call i32 @_Z17HlifCompressBatchILi1ERN7hipcomp25cascaded_compress_wrapperImmLi128ELi4096EEERN18cooperative_groups12thread_blockEEvRK12CompressArgsOT0_OT1_()
  ret void
}

define i32 @_Z17HlifCompressBatchILi1ERN7hipcomp25cascaded_compress_wrapperImmLi128ELi4096EEERN18cooperative_groups12thread_blockEEvRK12CompressArgsOT0_OT1_() {
entry:
  %0 = call i32 @_ZN7hipcomp25cascaded_compress_wrapperImmLi128ELi4096EE14compress_chunkEPhPKhmmPm()
  ret i32 %0
}

define i32 @_ZN7hipcomp25cascaded_compress_wrapperImmLi128ELi4096EE14compress_chunkEPhPKhmmPm() {
entry:
  %0 = call i32 @_ZN7hipcomp30do_cascaded_compression_kernelImmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_t()
  ret i32 %0
}

define i32 @_ZN7hipcomp30do_cascaded_compression_kernelImmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_t() {
entry:
  store ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp30do_cascaded_compression_kernelImmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE24shared_element_storage_0 to ptr), ptr null, align 8
  store ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp30do_cascaded_compression_kernelImmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE24shared_element_storage_1 to ptr), ptr null, align 8
  %add.ptr = getelementptr i32, ptr getelementptr inbounds (i32, ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp30do_cascaded_compression_kernelImmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE14chunk_metadata to ptr), i64 1), i64 0
  br i1 false, label %for.body65, label %for.end102

for.body65:                                       ; preds = %entry
  call void @_ZN7hipcomp18block_rle_compressImmtLi128EEEvPKT_T0_PS1_PT1_PS4_S7_()
  %call69 = call i32 @_ZN7hipcomp11block_writeItmLi128EEENS_13BlockIOStatusEPKT_T0_PjPKjPS5_S6_b(ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp30do_cascaded_compression_kernelImmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE19shared_count_buffer to ptr), ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp30do_cascaded_compression_kernelImmLi128ELi4096EEEviiiPKPKT_PKT0_PKPvPS6_28hipcompBatchedCascadedOpts_tE17shared_tmp_buffer to ptr))
  ret i32 %call69

for.end102:                                       ; preds = %entry
  %call106 = call i32 @_ZN7hipcomp11block_writeImmLi128EEENS_13BlockIOStatusEPKT_T0_PjPKjPS5_S6_b()
  ret i32 0
}

define void @_ZN7hipcomp18block_rle_compressImmtLi128EEEvPKT_T0_PS1_PT1_PS4_S7_() {
entry:
  store ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp18block_rle_compressImmtLi128EEEvPKT_T0_PS1_PT1_PS4_S7_E12temp_storage to ptr), ptr null, align 8
  ret void
}

define i32 @_ZN7hipcomp11block_writeImmLi128EEENS_13BlockIOStatusEPKT_T0_PjPKjPS5_S6_b() {
entry:
  call void @_ZN7hipcomp13block_bitpackImmLi128EEEvPKT_T0_PjPS4_()
  ret i32 0
}

define void @_ZN7hipcomp13block_bitpackImmLi128EEEvPKT_T0_PjPS4_() {
entry:
  call void @_ZN7hipcomp16get_for_bitwidthImmLi128EEEvPKT_T0_PS1_Pj()
  ret void
}

define void @_ZN7hipcomp16get_for_bitwidthImmLi128EEEvPKT_T0_PS1_Pj() {
entry:
  store ptr addrspacecast (ptr addrspace(3) @_ZZN7hipcomp16get_for_bitwidthImmLi128EEEvPKT_T0_PS1_PjE12temp_storage to ptr), ptr null, align 8
  ret void
}

define amdgpu_kernel void @_Z23HlifCompressBatchKernelILi64EN7hipcomp25cascaded_compress_wrapperIhmLi128ELi4096EEERK28hipcompBatchedCascadedOpts_tLi1EENSt9enable_ifIXsr3std10is_base_ofI21hlif_compress_wrapperT0_EE5valueEvE4typeE12CompressArgsT1_() {
entry:
  %0 = call i32 @_Z17HlifCompressBatchILi1ERN7hipcomp25cascaded_compress_wrapperIhmLi128ELi4096EEERN18cooperative_groups12thread_blockEEvRK12CompressArgsOT0_OT1_()
  ret void
}

define amdgpu_kernel void @_Z23HlifCompressBatchKernelILi64EN7hipcomp25cascaded_compress_wrapperItmLi128ELi4096EEERK28hipcompBatchedCascadedOpts_tLi1EENSt9enable_ifIXsr3std10is_base_ofI21hlif_compress_wrapperT0_EE5valueEvE4typeE12CompressArgsT1_() {
entry:
  %0 = call i32 @_Z17HlifCompressBatchILi1ERN7hipcomp25cascaded_compress_wrapperItmLi128ELi4096EEERN18cooperative_groups12thread_blockEEvRK12CompressArgsOT0_OT1_()
  ret void
}

define amdgpu_kernel void @_Z23HlifCompressBatchKernelILi64EN7hipcomp25cascaded_compress_wrapperIjmLi128ELi4096EEERK28hipcompBatchedCascadedOpts_tLi1EENSt9enable_ifIXsr3std10is_base_ofI21hlif_compress_wrapperT0_EE5valueEvE4typeE12CompressArgsT1_() {
entry:
  %0 = call i32 @_Z17HlifCompressBatchILi1ERN7hipcomp25cascaded_compress_wrapperIjmLi128ELi4096EEERN18cooperative_groups12thread_blockEEvRK12CompressArgsOT0_OT1_()
  ret void
}

define amdgpu_kernel void @_Z23HlifCompressBatchKernelILi64EN7hipcomp25cascaded_compress_wrapperImmLi128ELi4096EEERK28hipcompBatchedCascadedOpts_tLi1EENSt9enable_ifIXsr3std10is_base_ofI21hlif_compress_wrapperT0_EE5valueEvE4typeE12CompressArgsT1_() {
entry:
  %0 = call i32 @_Z17HlifCompressBatchILi1ERN7hipcomp25cascaded_compress_wrapperImmLi128ELi4096EEERN18cooperative_groups12thread_blockEEvRK12CompressArgsOT0_OT1_()
  ret void
}
