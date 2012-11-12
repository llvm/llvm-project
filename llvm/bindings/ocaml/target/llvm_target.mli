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

module DataLayout : sig
  type t

  (** [DataLayout.create rep] parses the target data string representation [rep].
      See the constructor llvm::DataLayout::DataLayout. *)
  external create : string -> t = "llvm_targetdata_create"

  (** [add_target_data td pm] adds the target data [td] to the pass manager [pm].
      Does not take ownership of the target data.
      See the method llvm::PassManagerBase::add. *)
  external add : t -> [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
               = "llvm_targetdata_add"

  (** [as_string td] is the string representation of the target data [td].
      See the constructor llvm::DataLayout::DataLayout. *)
  external as_string : t -> string = "llvm_targetdata_as_string"

  (** Deallocates a DataLayout.
      See the destructor llvm::DataLayout::~DataLayout. *)
  external dispose : t -> unit = "llvm_targetdata_dispose"
end

(** Returns the byte order of a target, either LLVMBigEndian or
    LLVMLittleEndian.
    See the method llvm::DataLayout::isLittleEndian. *)
external byte_order : DataLayout.t -> Endian.t = "llvm_byte_order"

(** Returns the pointer size in bytes for a target.
    See the method llvm::DataLayout::getPointerSize. *)
external pointer_size : DataLayout.t -> int = "llvm_pointer_size"

(** Returns the integer type that is the same size as a pointer on a target.
    See the method llvm::DataLayout::getIntPtrType. *)
external intptr_type : DataLayout.t -> Llvm.lltype = "LLVMIntPtrType"

(** Computes the size of a type in bytes for a target.
    See the method llvm::DataLayout::getTypeSizeInBits. *)
external size_in_bits : DataLayout.t -> Llvm.lltype -> Int64.t
                      = "llvm_size_in_bits"

(** Computes the storage size of a type in bytes for a target.
    See the method llvm::DataLayout::getTypeStoreSize. *)
external store_size : DataLayout.t -> Llvm.lltype -> Int64.t = "llvm_store_size"

(** Computes the ABI size of a type in bytes for a target.
    See the method llvm::DataLayout::getTypeAllocSize. *)
external abi_size : DataLayout.t -> Llvm.lltype -> Int64.t = "llvm_abi_size"

(** Computes the ABI alignment of a type in bytes for a target.
    See the method llvm::DataLayout::getTypeABISize. *)
external abi_align : DataLayout.t -> Llvm.lltype -> int = "llvm_abi_align"

(** Computes the call frame alignment of a type in bytes for a target.
    See the method llvm::DataLayout::getTypeABISize. *)
external stack_align : DataLayout.t -> Llvm.lltype -> int = "llvm_stack_align"

(** Computes the preferred alignment of a type in bytes for a target.
    See the method llvm::DataLayout::getTypeABISize. *)
external preferred_align : DataLayout.t -> Llvm.lltype -> int
                         = "llvm_preferred_align"

(** Computes the preferred alignment of a global variable in bytes for a target.
    See the method llvm::DataLayout::getPreferredAlignment. *)
external preferred_align_of_global : DataLayout.t -> Llvm.llvalue -> int
                                   = "llvm_preferred_align_of_global"

(** Computes the structure element that contains the byte offset for a target.
    See the method llvm::StructLayout::getElementContainingOffset. *)
external element_at_offset : DataLayout.t -> Llvm.lltype -> Int64.t -> int
                           = "llvm_element_at_offset"

(** Computes the byte offset of the indexed struct element for a target.
    See the method llvm::StructLayout::getElementContainingOffset. *)
external offset_of_element : DataLayout.t -> Llvm.lltype -> int -> Int64.t
                           = "llvm_offset_of_element"
