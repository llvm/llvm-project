//===- IntrinsicsHexagon.td - Defines Hexagon intrinsics ---*- tablegen -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the Hexagon-specific intrinsics.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Definitions for all Hexagon intrinsics.
//
// All Hexagon intrinsics start with "llvm.hexagon.".
let TargetPrefix = "hexagon" in {
  /// Hexagon_Intrinsic - Base class for the majority of Hexagon intrinsics.
  class Hexagon_Intrinsic<string GCCIntSuffix, list<LLVMType> ret_types,
                              list<LLVMType> param_types,
                              list<IntrinsicProperty> properties>
    : GCCBuiltin<!strconcat("__builtin_", GCCIntSuffix)>,
      Intrinsic<ret_types, param_types, properties>;

  /// Hexagon_NonGCC_Intrinsic - Base class for bitcode convertible Hexagon
  /// intrinsics.
  class Hexagon_NonGCC_Intrinsic<list<LLVMType> ret_types,
                                 list<LLVMType> param_types,
                                 list<IntrinsicProperty> properties>
    : Intrinsic<ret_types, param_types, properties>;
}

class Hexagon_mem_memmemsi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_ptr_ty], [llvm_ptr_ty, llvm_ptr_ty,
                           llvm_i32_ty],
                          [IntrArgMemOnly]>;

class Hexagon_mem_memsisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty,
                           llvm_i32_ty],
                          [IntrWriteMem]>;

class Hexagon_mem_memdisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_ptr_ty], [llvm_ptr_ty, llvm_i64_ty,
                           llvm_i32_ty],
                          [IntrWriteMem]>;

class Hexagon_mem_memmemsisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_ptr_ty], [llvm_ptr_ty, llvm_ptr_ty,
                           llvm_i32_ty, llvm_i32_ty],
                          [IntrArgMemOnly, ImmArg<ArgIndex<3>>]>;

class Hexagon_mem_memsisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty,
                           llvm_i32_ty, llvm_i32_ty],
                          [IntrWriteMem, ImmArg<ArgIndex<3>>]>;

class Hexagon_mem_memdisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_ptr_ty], [llvm_ptr_ty, llvm_i64_ty,
                           llvm_i32_ty, llvm_i32_ty],
                          [IntrWriteMem, ImmArg<ArgIndex<3>>]>;

//
// BUILTIN_INFO_NONCONST(circ_ldd,PTR_ftype_PTRPTRSISI,4)
//
def int_hexagon_circ_ldd :
Hexagon_mem_memmemsisi_Intrinsic<"circ_ldd">;
//
// BUILTIN_INFO_NONCONST(circ_ldw,PTR_ftype_PTRPTRSISI,4)
//
def int_hexagon_circ_ldw :
Hexagon_mem_memmemsisi_Intrinsic<"circ_ldw">;
//
// BUILTIN_INFO_NONCONST(circ_ldh,PTR_ftype_PTRPTRSISI,4)
//
def int_hexagon_circ_ldh :
Hexagon_mem_memmemsisi_Intrinsic<"circ_ldh">;
//
// BUILTIN_INFO_NONCONST(circ_lduh,PTR_ftype_PTRPTRSISI,4)
//
def int_hexagon_circ_lduh :
Hexagon_mem_memmemsisi_Intrinsic<"circ_lduh">;
//
// BUILTIN_INFO_NONCONST(circ_ldb,PTR_ftype_PTRPTRSISI,4)
//
def int_hexagon_circ_ldb :
Hexagon_mem_memmemsisi_Intrinsic<"circ_ldb">;
//
// BUILTIN_INFO_NONCONST(circ_ldub,PTR_ftype_PTRPTRSISI,4)
//
def int_hexagon_circ_ldub :
Hexagon_mem_memmemsisi_Intrinsic<"circ_ldub">;

//
// BUILTIN_INFO_NONCONST(circ_std,PTR_ftype_PTRDISISI,4)
//
def int_hexagon_circ_std :
Hexagon_mem_memdisisi_Intrinsic<"circ_std">;
//
// BUILTIN_INFO_NONCONST(circ_stw,PTR_ftype_PTRSISISI,4)
//
def int_hexagon_circ_stw :
Hexagon_mem_memsisisi_Intrinsic<"circ_stw">;
//
// BUILTIN_INFO_NONCONST(circ_sth,PTR_ftype_PTRSISISI,4)
//
def int_hexagon_circ_sth :
Hexagon_mem_memsisisi_Intrinsic<"circ_sth">;
//
// BUILTIN_INFO_NONCONST(circ_sthhi,PTR_ftype_PTRSISISI,4)
//
def int_hexagon_circ_sthhi :
Hexagon_mem_memsisisi_Intrinsic<"circ_sthhi">;
//
// BUILTIN_INFO_NONCONST(circ_stb,PTR_ftype_PTRSISISI,4)
//
def int_hexagon_circ_stb :
Hexagon_mem_memsisisi_Intrinsic<"circ_stb">;

def int_hexagon_prefetch :
Hexagon_Intrinsic<"HEXAGON_prefetch", [], [llvm_ptr_ty], []>;

def llvm_ptr32_ty : LLVMPointerType<llvm_i32_ty>;
def llvm_ptr64_ty : LLVMPointerType<llvm_i64_ty>;

// Mark locked loads as read/write to prevent any accidental reordering.
def int_hexagon_L2_loadw_locked :
Hexagon_Intrinsic<"HEXAGON_L2_loadw_locked", [llvm_i32_ty], [llvm_ptr32_ty],
      [IntrArgMemOnly, NoCapture<ArgIndex<0>>]>;
def int_hexagon_L4_loadd_locked :
Hexagon_Intrinsic<"HEXAGON_L4_loadd_locked", [llvm_i64_ty], [llvm_ptr64_ty],
      [IntrArgMemOnly, NoCapture<ArgIndex<0>>]>;

def int_hexagon_S2_storew_locked :
Hexagon_Intrinsic<"HEXAGON_S2_storew_locked", [llvm_i32_ty],
      [llvm_ptr32_ty, llvm_i32_ty], [IntrArgMemOnly, NoCapture<ArgIndex<0>>]>;
def int_hexagon_S4_stored_locked :
Hexagon_Intrinsic<"HEXAGON_S4_stored_locked", [llvm_i32_ty],
      [llvm_ptr64_ty, llvm_i64_ty], [IntrArgMemOnly, NoCapture<ArgIndex<0>>]>;

def int_hexagon_vmemcpy : Hexagon_Intrinsic<"hexagon_vmemcpy",
    [], [llvm_ptr_ty, llvm_ptr_ty, llvm_i32_ty],
    [IntrArgMemOnly, NoCapture<ArgIndex<0>>, NoCapture<ArgIndex<1>>, WriteOnly<ArgIndex<0>>, ReadOnly<ArgIndex<1>>]>;

def int_hexagon_vmemset : Hexagon_Intrinsic<"hexagon_vmemset",
    [], [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrArgMemOnly, NoCapture<ArgIndex<0>>, WriteOnly<ArgIndex<0>>]>;

multiclass Hexagon_custom_circ_ld_Intrinsic<LLVMType ElTy> {
  def NAME#_pci : Hexagon_NonGCC_Intrinsic<
    [ElTy, llvm_ptr_ty],
    [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_ptr_ty],
    [IntrArgMemOnly, NoCapture<ArgIndex<3>>]>;
  def NAME#_pcr : Hexagon_NonGCC_Intrinsic<
    [ElTy, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty, llvm_ptr_ty],
    [IntrArgMemOnly, NoCapture<ArgIndex<2>>]>;
}

defm int_hexagon_L2_loadrub : Hexagon_custom_circ_ld_Intrinsic<llvm_i32_ty>;
defm int_hexagon_L2_loadrb : Hexagon_custom_circ_ld_Intrinsic<llvm_i32_ty>;
defm int_hexagon_L2_loadruh : Hexagon_custom_circ_ld_Intrinsic<llvm_i32_ty>;
defm int_hexagon_L2_loadrh : Hexagon_custom_circ_ld_Intrinsic<llvm_i32_ty>;
defm int_hexagon_L2_loadri : Hexagon_custom_circ_ld_Intrinsic<llvm_i32_ty>;
defm int_hexagon_L2_loadrd : Hexagon_custom_circ_ld_Intrinsic<llvm_i64_ty>;

multiclass Hexagon_custom_circ_st_Intrinsic<LLVMType ElTy> {
  def NAME#_pci : Hexagon_NonGCC_Intrinsic<
    [llvm_ptr_ty],
    [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty, ElTy, llvm_ptr_ty],
    [IntrArgMemOnly, NoCapture<ArgIndex<4>>]>;
  def NAME#_pcr : Hexagon_NonGCC_Intrinsic<
    [llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty, ElTy, llvm_ptr_ty],
    [IntrArgMemOnly, NoCapture<ArgIndex<3>>]>;
}

defm int_hexagon_S2_storerb : Hexagon_custom_circ_st_Intrinsic<llvm_i32_ty>;
defm int_hexagon_S2_storerh : Hexagon_custom_circ_st_Intrinsic<llvm_i32_ty>;
defm int_hexagon_S2_storerf : Hexagon_custom_circ_st_Intrinsic<llvm_i32_ty>;
defm int_hexagon_S2_storeri : Hexagon_custom_circ_st_Intrinsic<llvm_i32_ty>;
defm int_hexagon_S2_storerd : Hexagon_custom_circ_st_Intrinsic<llvm_i64_ty>;

// The front-end emits the intrinsic call with only two arguments. The third
// argument from the builtin is already used by front-end to write to memory
// by generating a store.
class Hexagon_custom_brev_ld_Intrinsic<LLVMType ElTy>
 : Hexagon_NonGCC_Intrinsic<
    [ElTy, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty],
    [IntrReadMem]>;

def int_hexagon_L2_loadrub_pbr : Hexagon_custom_brev_ld_Intrinsic<llvm_i32_ty>;
def int_hexagon_L2_loadrb_pbr : Hexagon_custom_brev_ld_Intrinsic<llvm_i32_ty>;
def int_hexagon_L2_loadruh_pbr : Hexagon_custom_brev_ld_Intrinsic<llvm_i32_ty>;
def int_hexagon_L2_loadrh_pbr : Hexagon_custom_brev_ld_Intrinsic<llvm_i32_ty>;
def int_hexagon_L2_loadri_pbr : Hexagon_custom_brev_ld_Intrinsic<llvm_i32_ty>;
def int_hexagon_L2_loadrd_pbr : Hexagon_custom_brev_ld_Intrinsic<llvm_i64_ty>;

def int_hexagon_S2_storerb_pbr : Hexagon_mem_memsisi_Intrinsic<"brev_stb">;
def int_hexagon_S2_storerh_pbr : Hexagon_mem_memsisi_Intrinsic<"brev_sth">;
def int_hexagon_S2_storerf_pbr : Hexagon_mem_memsisi_Intrinsic<"brev_sthhi">;
def int_hexagon_S2_storeri_pbr : Hexagon_mem_memsisi_Intrinsic<"brev_stw">;
def int_hexagon_S2_storerd_pbr : Hexagon_mem_memdisi_Intrinsic<"brev_std">;

// tag : V6_vrmpybub_rtt
class Hexagon_v32i32_v16i32i64_rtt_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
       [llvm_v32i32_ty], [llvm_v16i32_ty,llvm_i64_ty],
       [IntrNoMem]>;

// tag : V6_vrmpybub_rtt_128B
class Hexagon_v64i32_v32i32i64_rtt_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
       [llvm_v64i32_ty], [llvm_v32i32_ty,llvm_i64_ty],
       [IntrNoMem]>;

// tag : V6_vrmpybub_rtt_acc
class Hexagon_v32i32_v32i32v16i32i64_rtt_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
       [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_v16i32_ty,llvm_i64_ty],
       [IntrNoMem]>;

// tag : V6_vrmpybub_rtt_acc_128B
class Hexagon_v64i32_v64i32v32i32i64_rtt_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
       [llvm_v64i32_ty], [llvm_v64i32_ty,llvm_v32i32_ty,llvm_i64_ty],
       [IntrNoMem]>;

def int_hexagon_V6_vrmpybub_rtt :
Hexagon_v32i32_v16i32i64_rtt_Intrinsic<"HEXAGON_V6_vrmpybub_rtt">;

def int_hexagon_V6_vrmpybub_rtt_128B :
Hexagon_v64i32_v32i32i64_rtt_Intrinsic<"HEXAGON_V6_vrmpybub_rtt_128B">;

def int_hexagon_V6_vrmpybub_rtt_acc :
Hexagon_v32i32_v32i32v16i32i64_rtt_Intrinsic<"HEXAGON_V6_vrmpybub_rtt_acc">;

def int_hexagon_V6_vrmpybub_rtt_acc_128B :
Hexagon_v64i32_v64i32v32i32i64_rtt_Intrinsic<"HEXAGON_V6_vrmpybub_rtt_acc_128B">;

def int_hexagon_V6_vrmpyub_rtt :
Hexagon_v32i32_v16i32i64_rtt_Intrinsic<"HEXAGON_V6_vrmpyub_rtt">;

def int_hexagon_V6_vrmpyub_rtt_128B :
Hexagon_v64i32_v32i32i64_rtt_Intrinsic<"HEXAGON_V6_vrmpyub_rtt_128B">;

def int_hexagon_V6_vrmpyub_rtt_acc :
Hexagon_v32i32_v32i32v16i32i64_rtt_Intrinsic<"HEXAGON_V6_vrmpyub_rtt_acc">;

def int_hexagon_V6_vrmpyub_rtt_acc_128B :
Hexagon_v64i32_v64i32v32i32i64_rtt_Intrinsic<"HEXAGON_V6_vrmpyub_rtt_acc_128B">;

// HVX Vector predicate casts.
// These intrinsics do not emit (nor do they correspond to) any instructions,
// they are no-ops.

def int_hexagon_V6_pred_typecast :
Hexagon_NonGCC_Intrinsic<[llvm_anyvector_ty], [llvm_anyvector_ty], [IntrNoMem]>;

def int_hexagon_V6_pred_typecast_128B :
Hexagon_NonGCC_Intrinsic<[llvm_anyvector_ty], [llvm_anyvector_ty], [IntrNoMem]>;

// Masked vector stores
//

class Hexagon_custom_vms_Intrinsic
  : Hexagon_NonGCC_Intrinsic<
       [], [llvm_v64i1_ty,llvm_ptr_ty,llvm_v16i32_ty], [IntrWriteMem]>;

class Hexagon_custom_vms_Intrinsic_128B
  : Hexagon_NonGCC_Intrinsic<
       [], [llvm_v128i1_ty,llvm_ptr_ty,llvm_v32i32_ty], [IntrWriteMem]>;

def int_hexagon_V6_vmaskedstoreq: Hexagon_custom_vms_Intrinsic;
def int_hexagon_V6_vmaskedstorenq: Hexagon_custom_vms_Intrinsic;
def int_hexagon_V6_vmaskedstorentq: Hexagon_custom_vms_Intrinsic;
def int_hexagon_V6_vmaskedstorentnq: Hexagon_custom_vms_Intrinsic;

def int_hexagon_V6_vmaskedstoreq_128B: Hexagon_custom_vms_Intrinsic_128B;
def int_hexagon_V6_vmaskedstorenq_128B: Hexagon_custom_vms_Intrinsic_128B;
def int_hexagon_V6_vmaskedstorentq_128B: Hexagon_custom_vms_Intrinsic_128B;
def int_hexagon_V6_vmaskedstorentnq_128B: Hexagon_custom_vms_Intrinsic_128B;

include "llvm/IR/IntrinsicsHexagonDep.td"
