(*===-- llvm_target.ml - LLVM Ocaml Interface ------------------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

module Endian = struct
  type t =
  | Big
  | Little
end

module TargetData = struct
  type t

  external create : string -> t = "llvm_targetdata_create"
  external add : t -> [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
               = "llvm_targetdata_add"
  external as_string : t -> string = "llvm_targetdata_as_string"
  external invalidate_struct_layout : t -> Llvm.lltype -> unit
                                    = "llvm_targetdata_invalidate_struct_layout"
  external dispose : t -> unit = "llvm_targetdata_dispose"
end

external byte_order : TargetData.t -> Endian.t = "llvm_byte_order"
external pointer_size : TargetData.t -> int = "llvm_pointer_size"
external intptr_type : TargetData.t -> Llvm.lltype = "LLVMIntPtrType"
external size_in_bits : TargetData.t -> Llvm.lltype -> Int64.t
                      = "llvm_size_in_bits"
external store_size : TargetData.t -> Llvm.lltype -> Int64.t = "llvm_store_size"
external abi_size : TargetData.t -> Llvm.lltype -> Int64.t = "llvm_abi_size"
external abi_align : TargetData.t -> Llvm.lltype -> int = "llvm_abi_align"
external stack_align : TargetData.t -> Llvm.lltype -> int = "llvm_stack_align"
external preferred_align : TargetData.t -> Llvm.lltype -> int
                         = "llvm_preferred_align"
external preferred_align_of_global : TargetData.t -> Llvm.llvalue -> int
                                   = "llvm_preferred_align_of_global"
external element_at_offset : TargetData.t -> Llvm.lltype -> Int64.t -> int
                           = "llvm_element_at_offset"
external offset_of_element : TargetData.t -> Llvm.lltype -> int -> Int64.t
                           = "llvm_offset_of_element"
