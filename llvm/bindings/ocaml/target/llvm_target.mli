(*===-- llvm_target.mli - LLVM Ocaml Interface -----------------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

(** Target Information.

    This interface provides an ocaml API for LLVM target information,
    the classes in the Target library. *)

module Endian : sig
  type t =
  | Big
  | Little
end

module TargetData : sig
  type t

  (** [TargetData.create rep] parses the target data string representation [rep].
      See the constructor llvm::TargetData::TargetData. *)
  external create : string -> t = "llvm_targetdata_create"

  (** [add_target_data td pm] adds the target data [td] to the pass manager [pm].
      Does not take ownership of the target data.
      See the method llvm::PassManagerBase::add. *)
  external add : t -> [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
               = "llvm_targetdata_add"

  (** [as_string td] is the string representation of the target data [td].
      See the constructor llvm::TargetData::TargetData. *)
  external as_string : t -> string = "llvm_targetdata_as_string"

  (** Struct layouts are speculatively cached. If a TargetDataRef is alive when
      types are being refined and removed, this method must be called whenever a
      struct type is removed to avoid a dangling pointer in this cache.
      See the method llvm::TargetData::InvalidateStructLayoutInfo. *)
  external invalidate_struct_layout : t -> Llvm.lltype -> unit
                                    = "llvm_targetdata_invalidate_struct_layout"

  (** Deallocates a TargetData.
      See the destructor llvm::TargetData::~TargetData. *)
  external dispose : t -> unit = "llvm_targetdata_dispose"
end

(** Returns the byte order of a target, either LLVMBigEndian or
    LLVMLittleEndian.
    See the method llvm::TargetData::isLittleEndian. *)
external byte_order : TargetData.t -> Endian.t = "llvm_byte_order"

(** Returns the pointer size in bytes for a target.
    See the method llvm::TargetData::getPointerSize. *)
external pointer_size : TargetData.t -> int = "llvm_pointer_size"

(** Returns the integer type that is the same size as a pointer on a target.
    See the method llvm::TargetData::getIntPtrType. *)
external intptr_type : TargetData.t -> Llvm.lltype = "LLVMIntPtrType"

(** Computes the size of a type in bytes for a target.
    See the method llvm::TargetData::getTypeSizeInBits. *)
external size_in_bits : TargetData.t -> Llvm.lltype -> Int64.t
                      = "llvm_size_in_bits"

(** Computes the storage size of a type in bytes for a target.
    See the method llvm::TargetData::getTypeStoreSize. *)
external store_size : TargetData.t -> Llvm.lltype -> Int64.t = "llvm_store_size"

(** Computes the ABI size of a type in bytes for a target.
    See the method llvm::TargetData::getTypePaddedSize. *)
external abi_size : TargetData.t -> Llvm.lltype -> Int64.t = "llvm_abi_size"

(** Computes the ABI alignment of a type in bytes for a target.
    See the method llvm::TargetData::getTypeABISize. *)
external abi_align : TargetData.t -> Llvm.lltype -> int = "llvm_abi_align"

(** Computes the call frame alignment of a type in bytes for a target.
    See the method llvm::TargetData::getTypeABISize. *)
external stack_align : TargetData.t -> Llvm.lltype -> int = "llvm_stack_align"

(** Computes the preferred alignment of a type in bytes for a target.
    See the method llvm::TargetData::getTypeABISize. *)
external preferred_align : TargetData.t -> Llvm.lltype -> int
                         = "llvm_preferred_align"

(** Computes the preferred alignment of a global variable in bytes for a target.
    See the method llvm::TargetData::getPreferredAlignment. *)
external preferred_align_of_global : TargetData.t -> Llvm.llvalue -> int
                                   = "llvm_preferred_align_of_global"

(** Computes the structure element that contains the byte offset for a target.
    See the method llvm::StructLayout::getElementContainingOffset. *)
external element_at_offset : TargetData.t -> Llvm.lltype -> Int64.t -> int
                           = "llvm_element_at_offset"

(** Computes the byte offset of the indexed struct element for a target.
    See the method llvm::StructLayout::getElementContainingOffset. *)
external offset_of_element : TargetData.t -> Llvm.lltype -> int -> Int64.t
                           = "llvm_offset_of_element"
