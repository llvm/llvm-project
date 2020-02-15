//===- IntrinsicsBPF.td - Defines BPF intrinsics -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the BPF-specific intrinsics.
//
//===----------------------------------------------------------------------===//

// Specialized loads from packet
let TargetPrefix = "bpf" in {  // All intrinsics start with "llvm.bpf."
  def int_bpf_load_byte : GCCBuiltin<"__builtin_bpf_load_byte">,
              Intrinsic<[llvm_i64_ty], [llvm_ptr_ty, llvm_i64_ty], [IntrReadMem]>;
  def int_bpf_load_half : GCCBuiltin<"__builtin_bpf_load_half">,
              Intrinsic<[llvm_i64_ty], [llvm_ptr_ty, llvm_i64_ty], [IntrReadMem]>;
  def int_bpf_load_word : GCCBuiltin<"__builtin_bpf_load_word">,
              Intrinsic<[llvm_i64_ty], [llvm_ptr_ty, llvm_i64_ty], [IntrReadMem]>;
  def int_bpf_pseudo : GCCBuiltin<"__builtin_bpf_pseudo">,
              Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty]>;
  def int_bpf_preserve_field_info : GCCBuiltin<"__builtin_bpf_preserve_field_info">,
              Intrinsic<[llvm_i32_ty], [llvm_anyptr_ty, llvm_i64_ty],
              [IntrNoMem, ImmArg<1>]>;
  def int_bpf_btf_type_id : GCCBuiltin<"__builtin_bpf_btf_type_id">,
              Intrinsic<[llvm_i32_ty], [llvm_any_ty, llvm_any_ty, llvm_i64_ty],
              [IntrNoMem]>;
}
