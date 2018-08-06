//===- IntrinsicsHexagon.td - Defines Hexagon intrinsics ---*- tablegen -*-===//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  /// Hexagon_Intrinsic - Base class for all Hexagon intrinsics.
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

//===----------------------------------------------------------------------===//
//
// DEF_FUNCTION_TYPE_1(QI_ftype_MEM,BT_BOOL,BT_PTR) ->
// Hexagon_qi_mem_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_mem_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_ptr_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(HI_ftype_SI,BT_I16,BT_INT) ->
// Hexagon_hi_si_Intrinsic<string GCCIntSuffix>
//
class Hexagon_hi_si_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i16_ty], [llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(SI_ftype_SI,BT_INT,BT_INT) ->
// Hexagon_si_si_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_si_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(DI_ftype_SI,BT_LONGLONG,BT_INT) ->
// Hexagon_di_si_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_si_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(SI_ftype_DI,BT_INT,BT_LONGLONG) ->
// Hexagon_si_di_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_di_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(DI_ftype_DI,BT_LONGLONG,BT_LONGLONG) ->
// Hexagon_di_di_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_di_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(QI_ftype_QI,BT_BOOL,BT_BOOL) ->
// Hexagon_qi_qi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_qi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(QI_ftype_SI,BT_BOOL,BT_INT) ->
// Hexagon_qi_si_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_si_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(DI_ftype_QI,BT_LONGLONG,BT_BOOL) ->
// Hexagon_di_qi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_qi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(SI_ftype_QI,BT_INT,BT_BOOL) ->
// Hexagon_si_qi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_qi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(QI_ftype_SISI,BT_BOOL,BT_INT,BT_INT) ->
// Hexagon_qi_sisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_sisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(void_ftype_SISI,BT_VOID,BT_INT,BT_INT) ->
// Hexagon_void_sisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_void_sisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_void_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(SI_ftype_SISI,BT_INT,BT_INT,BT_INT) ->
// Hexagon_si_sisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(USI_ftype_SISI,BT_UINT,BT_INT,BT_INT) ->
// Hexagon_usi_sisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_usi_sisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(DI_ftype_SISI,BT_LONGLONG,BT_INT,BT_INT) ->
// Hexagon_di_sisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_sisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(UDI_ftype_SISI,BT_ULONGLONG,BT_INT,BT_INT) ->
// Hexagon_udi_sisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_udi_sisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(DI_ftype_SIDI,BT_LONGLONG,BT_INT,BT_LONGLONG) ->
// Hexagon_di_sidi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_sidi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i32_ty, llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(DI_ftype_DISI,BT_LONGLONG,BT_LONGLONG,BT_INT) ->
// Hexagon_di_disi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_disi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(SI_ftype_SIDI,BT_INT,BT_INT,BT_LONGLONG) ->
// Hexagon_si_sidi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sidi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(SI_ftype_DIDI,BT_INT,BT_LONGLONG,BT_LONGLONG) ->
// Hexagon_si_didi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_didi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i64_ty, llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(DI_ftype_DIDI,BT_LONGLONG,BT_LONGLONG,BT_LONGLONG) ->
// Hexagon_di_didi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_didi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(UDI_ftype_DIDI,BT_ULONGLONG,BT_LONGLONG,BT_LONGLONG) ->
// Hexagon_udi_didi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_udi_didi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(SI_ftype_DISI,BT_INT,BT_LONGLONG,BT_INT) ->
// Hexagon_si_disi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_disi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i64_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(QI_ftype_DIDI,BT_BOOL,BT_LONGLONG,BT_LONGLONG) ->
// Hexagon_qi_didi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_didi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_i64_ty, llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(QI_ftype_SIDI,BT_BOOL,BT_INT,BT_LONGLONG) ->
// Hexagon_qi_didi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_sidi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_i32_ty, llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(QI_ftype_DISI,BT_BOOL,BT_LONGLONG,BT_INT) ->
// Hexagon_qi_disi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_disi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_i64_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(QI_ftype_QIQI,BT_BOOL,BT_BOOL,BT_BOOL) ->
// Hexagon_qi_qiqi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_qiqi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(QI_ftype_QIQIQI,BT_BOOL,BT_BOOL,BT_BOOL) ->
// Hexagon_qi_qiqiqi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_qiqiqi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_i1_ty, llvm_i1_ty, llvm_i1_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(SI_ftype_QIQI,BT_INT,BT_BOOL,BT_BOOL) ->
// Hexagon_si_qiqi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_qiqi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(SI_ftype_QISI,BT_INT,BT_BOOL,BT_INT) ->
// Hexagon_si_qisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_qisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i1_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(void_ftype_SISISI,BT_VOID,BT_INT,BT_INT,BT_INT) ->
// Hexagon_void_sisisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_void_sisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_void_ty], [llvm_i32_ty, llvm_i32_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(SI_ftype_SISISI,BT_INT,BT_INT,BT_INT,BT_INT) ->
// Hexagon_si_sisisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(DI_ftype_SISISI,BT_LONGLONG,BT_INT,BT_INT,BT_INT) ->
// Hexagon_di_sisisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_sisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i32_ty, llvm_i32_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(SI_ftype_DISISI,BT_INT,BT_LONGLONG,BT_INT,BT_INT) ->
// Hexagon_si_disisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_disisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i64_ty, llvm_i32_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(DI_ftype_DISISI,BT_LONGLONG,BT_LONGLONG,BT_INT,BT_INT) ->
// Hexagon_di_disisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_disisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i32_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(SI_ftype_SIDISI,BT_INT,BT_INT,BT_LONGLONG,BT_INT) ->
// Hexagon_si_sidisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sidisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i64_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(DI_ftype_DIDISI,BT_LONGLONG,BT_LONGLONG,
//                     BT_LONGLONG,BT_INT) ->
// Hexagon_di_didisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_didisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(SI_ftype_SIDIDI,BT_INT,BT_INT,BT_LONGLONG,BT_LONGLONG) ->
// Hexagon_si_sididi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sididi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i64_ty,
                           llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(DI_ftype_DIDIDI,BT_LONGLONG,BT_LONGLONG,BT_LONGLONG,
//                     BT_LONGLONG) ->
// Hexagon_di_dididi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_dididi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty,
                           llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(SI_ftype_SISIDI,BT_INT,BT_INT,BT_INT,BT_LONGLONG) ->
// Hexagon_si_sisidi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sisidi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty,
                           llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(SI_ftype_QISISI,BT_INT,BT_BOOL,BT_INT,BT_INT) ->
// Hexagon_si_qisisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_qisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(DI_ftype_QISISI,BT_LONGLONG,BT_BOOL,BT_INT,BT_INT) ->
// Hexagon_di_qisisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_qisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i1_ty, llvm_i32_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(DI_ftype_QIDIDI,BT_LONGLONG,BT_BOOL,BT_LONGLONG,
//                     BT_LONGLONG) ->
// Hexagon_di_qididi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_qididi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i32_ty, llvm_i64_ty,
                           llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(DI_ftype_DIDIQI,BT_LONGLONG,BT_LONGLONG,BT_LONGLONG,
//                     BT_BOOL) ->
// Hexagon_di_didiqi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_didiqi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_4(SI_ftype_SISISISI,BT_INT,BT_INT,BT_INT,BT_INT,BT_INT) ->
// Hexagon_si_sisisisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sisisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty,
                           llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_4(DI_ftype_DIDISISI,BT_LONGLONG,BT_LONGLONG,
//                     BT_LONGLONG,BT_INT,BT_INT) ->
// Hexagon_di_didisisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_didisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty,
                           llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;

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
                          [IntrArgMemOnly]>;

class Hexagon_mem_memsisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty,
                           llvm_i32_ty, llvm_i32_ty],
                          [IntrWriteMem]>;

class Hexagon_mem_memdisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_ptr_ty], [llvm_ptr_ty, llvm_i64_ty,
                           llvm_i32_ty, llvm_i32_ty],
                          [IntrWriteMem]>;

class Hexagon_v256_v256v256_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v8i32_ty], [llvm_v8i32_ty, llvm_v8i32_ty],
                          [IntrArgMemOnly]>;

//
// Hexagon_sf_df_Intrinsic<string GCCIntSuffix>
//
class Hexagon_sf_si_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_float_ty], [llvm_i32_ty],
                          [IntrNoMem, Throws]>;
//
// Hexagon_sf_df_Intrinsic<string GCCIntSuffix>
//
class Hexagon_sf_df_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_float_ty], [llvm_double_ty],
                          [IntrNoMem]>;
//
// Hexagon_sf_di_Intrinsic<string GCCIntSuffix>
//
class Hexagon_sf_di_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_float_ty], [llvm_i64_ty],
                          [IntrNoMem]>;
//
// Hexagon_df_sf_Intrinsic<string GCCIntSuffix>
//
class Hexagon_df_sf_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_double_ty], [llvm_float_ty],
                          [IntrNoMem]>;
//
// Hexagon_di_sf_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_sf_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_float_ty],
                          [IntrNoMem]>;
//
// Hexagon_sf_sf_Intrinsic<string GCCIntSuffix>
//
class Hexagon_sf_sf_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_float_ty], [llvm_float_ty],
                          [IntrNoMem]>;
//
// Hexagon_si_sf_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sf_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_float_ty],
                          [IntrNoMem]>;
//
// Hexagon_si_df_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_df_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_double_ty],
                          [IntrNoMem]>;
//
// Hexagon_sf_sfsf_Intrinsic<string GCCIntSuffix>
//
class Hexagon_sf_sfsf_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_float_ty], [llvm_float_ty, llvm_float_ty],
                          [IntrNoMem, Throws]>;
//
// Hexagon_si_sfsf_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sfsf_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_float_ty, llvm_float_ty],
                          [IntrNoMem, Throws]>;
//
// Hexagon_si_sfsi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sfsi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_float_ty, llvm_i32_ty],
                          [IntrNoMem, Throws]>;
//
// Hexagon_qi_sfqi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_sfqi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_float_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// Hexagon_sf_sfsfsf_Intrinsic<string GCCIntSuffix>
//
class Hexagon_sf_sfsfsf_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_float_ty], [llvm_float_ty, llvm_float_ty,
                                            llvm_float_ty],
                          [IntrNoMem, Throws]>;
//
// Hexagon_sf_sfsfsfqi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_sf_sfsfsfqi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_float_ty], [llvm_float_ty, llvm_float_ty,
                                            llvm_float_ty,
                           llvm_i32_ty],
                          [IntrNoMem, Throws]>;
//
// Hexagon_di_dididi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_dididisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty,
                           llvm_i64_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// Hexagon_df_si_Intrinsic<string GCCIntSuffix>
//
class Hexagon_df_si_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_double_ty], [llvm_i32_ty],
                          [IntrNoMem, Throws]>;
//
// Hexagon_df_di_Intrinsic<string GCCIntSuffix>
//
class Hexagon_df_di_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_double_ty], [llvm_i64_ty],
                          [IntrNoMem]>;
//
// Hexagon_di_df_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_df_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_double_ty],
                          [IntrNoMem]>;
//
// Hexagon_df_df_Intrinsic<string GCCIntSuffix>
//
class Hexagon_df_df_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_double_ty], [llvm_double_ty],
                          [IntrNoMem]>;
//
// Hexagon_df_dfdf_Intrinsic<string GCCIntSuffix>
//
class Hexagon_df_dfdf_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_double_ty], [llvm_double_ty, llvm_double_ty],
                          [IntrNoMem, Throws]>;
//
// Hexagon_si_dfdf_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_dfdf_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_double_ty, llvm_double_ty],
                          [IntrNoMem, Throws]>;
//
// Hexagon_si_dfsi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_dfsi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_double_ty, llvm_i32_ty],
                          [IntrNoMem, Throws]>;
//
//
// Hexagon_df_dfdfdf_Intrinsic<string GCCIntSuffix>
//
class Hexagon_df_dfdfdf_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_double_ty], [llvm_double_ty, llvm_double_ty,
                                             llvm_double_ty],
                          [IntrNoMem, Throws]>;
//
// Hexagon_df_dfdfdf_Intrinsic<string GCCIntSuffix>
//
class Hexagon_df_dfdfdfqi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_double_ty], [llvm_double_ty, llvm_double_ty,
                                             llvm_double_ty,
                          llvm_i32_ty],
                          [IntrNoMem, Throws]>;


// This one below will not be auto-generated,
// so make sure, you don't overwrite this one.
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


def int_hexagon_mm256i_vaddw :
Hexagon_v256_v256v256_Intrinsic<"_mm256i_vaddw">;


// This one above will not be auto-generated,
// so make sure, you don't overwrite this one.
//
// BUILTIN_INFO(HEXAGON.C2_cmpeq,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpeq :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_cmpeq">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpgt,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpgt :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_cmpgt">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpgtu,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpgtu :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_cmpgtu">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpeqp,QI_ftype_DIDI,2)
//
def int_hexagon_C2_cmpeqp :
Hexagon_si_didi_Intrinsic<"HEXAGON_C2_cmpeqp">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpgtp,QI_ftype_DIDI,2)
//
def int_hexagon_C2_cmpgtp :
Hexagon_si_didi_Intrinsic<"HEXAGON_C2_cmpgtp">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpgtup,QI_ftype_DIDI,2)
//
def int_hexagon_C2_cmpgtup :
Hexagon_si_didi_Intrinsic<"HEXAGON_C2_cmpgtup">;
//
// BUILTIN_INFO(HEXAGON.A4_rcmpeqi,SI_ftype_SISI,2)
//
def int_hexagon_A4_rcmpeqi :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_rcmpeqi">;
//
// BUILTIN_INFO(HEXAGON.A4_rcmpneqi,SI_ftype_SISI,2)
//
def int_hexagon_A4_rcmpneqi :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_rcmpneqi">;
//
// BUILTIN_INFO(HEXAGON.A4_rcmpeq,SI_ftype_SISI,2)
//
def int_hexagon_A4_rcmpeq :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_rcmpeq">;
//
// BUILTIN_INFO(HEXAGON.A4_rcmpneq,SI_ftype_SISI,2)
//
def int_hexagon_A4_rcmpneq :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_rcmpneq">;
//
// BUILTIN_INFO(HEXAGON.C2_bitsset,QI_ftype_SISI,2)
//
def int_hexagon_C2_bitsset :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_bitsset">;
//
// BUILTIN_INFO(HEXAGON.C2_bitsclr,QI_ftype_SISI,2)
//
def int_hexagon_C2_bitsclr :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_bitsclr">;
//
// BUILTIN_INFO(HEXAGON.C4_nbitsset,QI_ftype_SISI,2)
//
def int_hexagon_C4_nbitsset :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C4_nbitsset">;
//
// BUILTIN_INFO(HEXAGON.C4_nbitsclr,QI_ftype_SISI,2)
//
def int_hexagon_C4_nbitsclr :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C4_nbitsclr">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpeqi,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpeqi :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_cmpeqi">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpgti,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpgti :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_cmpgti">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpgtui,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpgtui :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_cmpgtui">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpgei,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpgei :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_cmpgei">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpgeui,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpgeui :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_cmpgeui">;
//
// BUILTIN_INFO(HEXAGON.C2_cmplt,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmplt :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_cmplt">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpltu,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpltu :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_cmpltu">;
//
// BUILTIN_INFO(HEXAGON.C2_bitsclri,QI_ftype_SISI,2)
//
def int_hexagon_C2_bitsclri :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_bitsclri">;
//
// BUILTIN_INFO(HEXAGON.C4_nbitsclri,QI_ftype_SISI,2)
//
def int_hexagon_C4_nbitsclri :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C4_nbitsclri">;
//
// BUILTIN_INFO(HEXAGON.C4_cmpneqi,QI_ftype_SISI,2)
//
def int_hexagon_C4_cmpneqi :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C4_cmpneqi">;
//
// BUILTIN_INFO(HEXAGON.C4_cmpltei,QI_ftype_SISI,2)
//
def int_hexagon_C4_cmpltei :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C4_cmpltei">;
//
// BUILTIN_INFO(HEXAGON.C4_cmplteui,QI_ftype_SISI,2)
//
def int_hexagon_C4_cmplteui :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C4_cmplteui">;
//
// BUILTIN_INFO(HEXAGON.C4_cmpneq,QI_ftype_SISI,2)
//
def int_hexagon_C4_cmpneq :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C4_cmpneq">;
//
// BUILTIN_INFO(HEXAGON.C4_cmplte,QI_ftype_SISI,2)
//
def int_hexagon_C4_cmplte :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C4_cmplte">;
//
// BUILTIN_INFO(HEXAGON.C4_cmplteu,QI_ftype_SISI,2)
//
def int_hexagon_C4_cmplteu :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C4_cmplteu">;
//
// BUILTIN_INFO(HEXAGON.C2_and,QI_ftype_QIQI,2)
//
def int_hexagon_C2_and :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_and">;
//
// BUILTIN_INFO(HEXAGON.C2_or,QI_ftype_QIQI,2)
//
def int_hexagon_C2_or :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_or">;
//
// BUILTIN_INFO(HEXAGON.C2_xor,QI_ftype_QIQI,2)
//
def int_hexagon_C2_xor :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_xor">;
//
// BUILTIN_INFO(HEXAGON.C2_andn,QI_ftype_QIQI,2)
//
def int_hexagon_C2_andn :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_andn">;
//
// BUILTIN_INFO(HEXAGON.C2_not,QI_ftype_QI,1)
//
def int_hexagon_C2_not :
Hexagon_si_si_Intrinsic<"HEXAGON_C2_not">;
//
// BUILTIN_INFO(HEXAGON.C2_orn,QI_ftype_QIQI,2)
//
def int_hexagon_C2_orn :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_orn">;
//
// BUILTIN_INFO(HEXAGON.C4_and_and,QI_ftype_QIQIQI,3)
//
def int_hexagon_C4_and_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_C4_and_and">;
//
// BUILTIN_INFO(HEXAGON.C4_and_or,QI_ftype_QIQIQI,3)
//
def int_hexagon_C4_and_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_C4_and_or">;
//
// BUILTIN_INFO(HEXAGON.C4_or_and,QI_ftype_QIQIQI,3)
//
def int_hexagon_C4_or_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_C4_or_and">;
//
// BUILTIN_INFO(HEXAGON.C4_or_or,QI_ftype_QIQIQI,3)
//
def int_hexagon_C4_or_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_C4_or_or">;
//
// BUILTIN_INFO(HEXAGON.C4_and_andn,QI_ftype_QIQIQI,3)
//
def int_hexagon_C4_and_andn :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_C4_and_andn">;
//
// BUILTIN_INFO(HEXAGON.C4_and_orn,QI_ftype_QIQIQI,3)
//
def int_hexagon_C4_and_orn :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_C4_and_orn">;
//
// BUILTIN_INFO(HEXAGON.C4_or_andn,QI_ftype_QIQIQI,3)
//
def int_hexagon_C4_or_andn :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_C4_or_andn">;
//
// BUILTIN_INFO(HEXAGON.C4_or_orn,QI_ftype_QIQIQI,3)
//
def int_hexagon_C4_or_orn :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_C4_or_orn">;
//
// BUILTIN_INFO(HEXAGON.C2_pxfer_map,QI_ftype_QI,1)
//
def int_hexagon_C2_pxfer_map :
Hexagon_si_qi_Intrinsic<"HEXAGON_C2_pxfer_map">;
//
// BUILTIN_INFO(HEXAGON.C2_any8,QI_ftype_QI,1)
//
def int_hexagon_C2_any8 :
Hexagon_si_qi_Intrinsic<"HEXAGON_C2_any8">;
//
// BUILTIN_INFO(HEXAGON.C2_all8,QI_ftype_QI,1)
//
def int_hexagon_C2_all8 :
Hexagon_si_qi_Intrinsic<"HEXAGON_C2_all8">;
//
// BUILTIN_INFO(HEXAGON.C2_vitpack,SI_ftype_QIQI,2)
//
def int_hexagon_C2_vitpack :
Hexagon_si_qiqi_Intrinsic<"HEXAGON_C2_vitpack">;
//
// BUILTIN_INFO(HEXAGON.C2_mux,SI_ftype_QISISI,3)
//
def int_hexagon_C2_mux :
Hexagon_si_qisisi_Intrinsic<"HEXAGON_C2_mux">;
//
// BUILTIN_INFO(HEXAGON.C2_muxii,SI_ftype_QISISI,3)
//
def int_hexagon_C2_muxii :
Hexagon_si_qisisi_Intrinsic<"HEXAGON_C2_muxii">;
//
// BUILTIN_INFO(HEXAGON.C2_muxir,SI_ftype_QISISI,3)
//
def int_hexagon_C2_muxir :
Hexagon_si_qisisi_Intrinsic<"HEXAGON_C2_muxir">;
//
// BUILTIN_INFO(HEXAGON.C2_muxri,SI_ftype_QISISI,3)
//
def int_hexagon_C2_muxri :
Hexagon_si_qisisi_Intrinsic<"HEXAGON_C2_muxri">;
//
// BUILTIN_INFO(HEXAGON.C2_vmux,DI_ftype_QIDIDI,3)
//
def int_hexagon_C2_vmux :
Hexagon_di_qididi_Intrinsic<"HEXAGON_C2_vmux">;
//
// BUILTIN_INFO(HEXAGON.C2_mask,DI_ftype_QI,1)
//
def int_hexagon_C2_mask :
Hexagon_di_qi_Intrinsic<"HEXAGON_C2_mask">;
//
// BUILTIN_INFO(HEXAGON.A2_vcmpbeq,QI_ftype_DIDI,2)
//
def int_hexagon_A2_vcmpbeq :
Hexagon_si_didi_Intrinsic<"HEXAGON_A2_vcmpbeq">;
//
// BUILTIN_INFO(HEXAGON.A4_vcmpbeqi,QI_ftype_DISI,2)
//
def int_hexagon_A4_vcmpbeqi :
Hexagon_si_disi_Intrinsic<"HEXAGON_A4_vcmpbeqi">;
//
// BUILTIN_INFO(HEXAGON.A4_vcmpbeq_any,QI_ftype_DIDI,2)
//
def int_hexagon_A4_vcmpbeq_any :
Hexagon_si_didi_Intrinsic<"HEXAGON_A4_vcmpbeq_any">;
//
// BUILTIN_INFO(HEXAGON.A2_vcmpbgtu,QI_ftype_DIDI,2)
//
def int_hexagon_A2_vcmpbgtu :
Hexagon_si_didi_Intrinsic<"HEXAGON_A2_vcmpbgtu">;
//
// BUILTIN_INFO(HEXAGON.A4_vcmpbgtui,QI_ftype_DISI,2)
//
def int_hexagon_A4_vcmpbgtui :
Hexagon_si_disi_Intrinsic<"HEXAGON_A4_vcmpbgtui">;
//
// BUILTIN_INFO(HEXAGON.A4_vcmpbgt,QI_ftype_DIDI,2)
//
def int_hexagon_A4_vcmpbgt :
Hexagon_si_didi_Intrinsic<"HEXAGON_A4_vcmpbgt">;
//
// BUILTIN_INFO(HEXAGON.A4_vcmpbgti,QI_ftype_DISI,2)
//
def int_hexagon_A4_vcmpbgti :
Hexagon_si_disi_Intrinsic<"HEXAGON_A4_vcmpbgti">;
//
// BUILTIN_INFO(HEXAGON.A4_cmpbeq,QI_ftype_SISI,2)
//
def int_hexagon_A4_cmpbeq :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cmpbeq">;
//
// BUILTIN_INFO(HEXAGON.A4_cmpbeqi,QI_ftype_SISI,2)
//
def int_hexagon_A4_cmpbeqi :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cmpbeqi">;
//
// BUILTIN_INFO(HEXAGON.A4_cmpbgtu,QI_ftype_SISI,2)
//
def int_hexagon_A4_cmpbgtu :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cmpbgtu">;
//
// BUILTIN_INFO(HEXAGON.A4_cmpbgtui,QI_ftype_SISI,2)
//
def int_hexagon_A4_cmpbgtui :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cmpbgtui">;
//
// BUILTIN_INFO(HEXAGON.A4_cmpbgt,QI_ftype_SISI,2)
//
def int_hexagon_A4_cmpbgt :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cmpbgt">;
//
// BUILTIN_INFO(HEXAGON.A4_cmpbgti,QI_ftype_SISI,2)
//
def int_hexagon_A4_cmpbgti :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cmpbgti">;
//
// BUILTIN_INFO(HEXAGON.A2_vcmpheq,QI_ftype_DIDI,2)
//
def int_hexagon_A2_vcmpheq :
Hexagon_si_didi_Intrinsic<"HEXAGON_A2_vcmpheq">;
//
// BUILTIN_INFO(HEXAGON.A2_vcmphgt,QI_ftype_DIDI,2)
//
def int_hexagon_A2_vcmphgt :
Hexagon_si_didi_Intrinsic<"HEXAGON_A2_vcmphgt">;
//
// BUILTIN_INFO(HEXAGON.A2_vcmphgtu,QI_ftype_DIDI,2)
//
def int_hexagon_A2_vcmphgtu :
Hexagon_si_didi_Intrinsic<"HEXAGON_A2_vcmphgtu">;
//
// BUILTIN_INFO(HEXAGON.A4_vcmpheqi,QI_ftype_DISI,2)
//
def int_hexagon_A4_vcmpheqi :
Hexagon_si_disi_Intrinsic<"HEXAGON_A4_vcmpheqi">;
//
// BUILTIN_INFO(HEXAGON.A4_vcmphgti,QI_ftype_DISI,2)
//
def int_hexagon_A4_vcmphgti :
Hexagon_si_disi_Intrinsic<"HEXAGON_A4_vcmphgti">;
//
// BUILTIN_INFO(HEXAGON.A4_vcmphgtui,QI_ftype_DISI,2)
//
def int_hexagon_A4_vcmphgtui :
Hexagon_si_disi_Intrinsic<"HEXAGON_A4_vcmphgtui">;
//
// BUILTIN_INFO(HEXAGON.A4_cmpheq,QI_ftype_SISI,2)
//
def int_hexagon_A4_cmpheq :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cmpheq">;
//
// BUILTIN_INFO(HEXAGON.A4_cmphgt,QI_ftype_SISI,2)
//
def int_hexagon_A4_cmphgt :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cmphgt">;
//
// BUILTIN_INFO(HEXAGON.A4_cmphgtu,QI_ftype_SISI,2)
//
def int_hexagon_A4_cmphgtu :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cmphgtu">;
//
// BUILTIN_INFO(HEXAGON.A4_cmpheqi,QI_ftype_SISI,2)
//
def int_hexagon_A4_cmpheqi :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cmpheqi">;
//
// BUILTIN_INFO(HEXAGON.A4_cmphgti,QI_ftype_SISI,2)
//
def int_hexagon_A4_cmphgti :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cmphgti">;
//
// BUILTIN_INFO(HEXAGON.A4_cmphgtui,QI_ftype_SISI,2)
//
def int_hexagon_A4_cmphgtui :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cmphgtui">;
//
// BUILTIN_INFO(HEXAGON.A2_vcmpweq,QI_ftype_DIDI,2)
//
def int_hexagon_A2_vcmpweq :
Hexagon_si_didi_Intrinsic<"HEXAGON_A2_vcmpweq">;
//
// BUILTIN_INFO(HEXAGON.A2_vcmpwgt,QI_ftype_DIDI,2)
//
def int_hexagon_A2_vcmpwgt :
Hexagon_si_didi_Intrinsic<"HEXAGON_A2_vcmpwgt">;
//
// BUILTIN_INFO(HEXAGON.A2_vcmpwgtu,QI_ftype_DIDI,2)
//
def int_hexagon_A2_vcmpwgtu :
Hexagon_si_didi_Intrinsic<"HEXAGON_A2_vcmpwgtu">;
//
// BUILTIN_INFO(HEXAGON.A4_vcmpweqi,QI_ftype_DISI,2)
//
def int_hexagon_A4_vcmpweqi :
Hexagon_si_disi_Intrinsic<"HEXAGON_A4_vcmpweqi">;
//
// BUILTIN_INFO(HEXAGON.A4_vcmpwgti,QI_ftype_DISI,2)
//
def int_hexagon_A4_vcmpwgti :
Hexagon_si_disi_Intrinsic<"HEXAGON_A4_vcmpwgti">;
//
// BUILTIN_INFO(HEXAGON.A4_vcmpwgtui,QI_ftype_DISI,2)
//
def int_hexagon_A4_vcmpwgtui :
Hexagon_si_disi_Intrinsic<"HEXAGON_A4_vcmpwgtui">;
//
// BUILTIN_INFO(HEXAGON.A4_boundscheck,QI_ftype_SIDI,2)
//
def int_hexagon_A4_boundscheck :
Hexagon_si_sidi_Intrinsic<"HEXAGON_A4_boundscheck">;
//
// BUILTIN_INFO(HEXAGON.A4_tlbmatch,QI_ftype_DISI,2)
//
def int_hexagon_A4_tlbmatch :
Hexagon_si_disi_Intrinsic<"HEXAGON_A4_tlbmatch">;
//
// BUILTIN_INFO(HEXAGON.C2_tfrpr,SI_ftype_QI,1)
//
def int_hexagon_C2_tfrpr :
Hexagon_si_qi_Intrinsic<"HEXAGON_C2_tfrpr">;
//
// BUILTIN_INFO(HEXAGON.C2_tfrrp,QI_ftype_SI,1)
//
def int_hexagon_C2_tfrrp :
Hexagon_si_si_Intrinsic<"HEXAGON_C2_tfrrp">;
//
// BUILTIN_INFO(HEXAGON.C4_fastcorner9,QI_ftype_QIQI,2)
//
def int_hexagon_C4_fastcorner9 :
Hexagon_si_qiqi_Intrinsic<"HEXAGON_C4_fastcorner9">;
//
// BUILTIN_INFO(HEXAGON.C4_fastcorner9_not,QI_ftype_QIQI,2)
//
def int_hexagon_C4_fastcorner9_not :
Hexagon_si_qiqi_Intrinsic<"HEXAGON_C4_fastcorner9_not">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_hh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_hh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_hh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_hh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_hl_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_hl_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_hl_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_hl_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_lh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_lh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_lh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_lh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_ll_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_ll_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_ll_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_ll_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_hh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_hh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_hh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_hh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_hl_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_hl_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_hl_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_hl_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_lh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_lh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_lh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_lh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_ll_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_ll_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_ll_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_ll_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_sat_hh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_sat_hh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_sat_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_sat_hh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_sat_hh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_sat_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_sat_hl_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_sat_hl_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_sat_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_sat_hl_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_sat_hl_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_sat_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_sat_lh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_sat_lh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_sat_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_sat_lh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_sat_lh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_sat_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_sat_ll_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_sat_ll_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_sat_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_sat_ll_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_sat_ll_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_sat_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_sat_hh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_sat_hh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_sat_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_sat_hh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_sat_hh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_sat_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_sat_hl_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_sat_hl_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_sat_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_sat_hl_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_sat_hl_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_sat_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_sat_lh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_sat_lh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_sat_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_sat_lh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_sat_lh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_sat_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_sat_ll_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_sat_ll_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_sat_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_sat_ll_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_sat_ll_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_sat_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_hh_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_hh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_hh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_hh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_hl_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_hl_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_hl_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_hl_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_lh_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_lh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_lh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_lh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_ll_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_ll_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_ll_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_ll_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_hh_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_hh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_hh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_hh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_hl_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_hl_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_hl_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_hl_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_lh_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_lh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_lh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_lh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_ll_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_ll_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_ll_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_ll_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_rnd_hh_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_rnd_hh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_rnd_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_rnd_hh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_rnd_hh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_rnd_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_rnd_hl_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_rnd_hl_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_rnd_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_rnd_hl_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_rnd_hl_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_rnd_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_rnd_lh_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_rnd_lh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_rnd_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_rnd_lh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_rnd_lh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_rnd_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_rnd_ll_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_rnd_ll_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_rnd_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_rnd_ll_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_rnd_ll_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_rnd_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_rnd_hh_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_rnd_hh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_rnd_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_rnd_hh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_rnd_hh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_rnd_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_rnd_hl_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_rnd_hl_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_rnd_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_rnd_hl_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_rnd_hl_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_rnd_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_rnd_lh_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_rnd_lh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_rnd_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_rnd_lh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_rnd_lh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_rnd_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_rnd_ll_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_rnd_ll_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_rnd_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_rnd_ll_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_rnd_ll_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_rnd_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_acc_hh_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_acc_hh_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_acc_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_acc_hh_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_acc_hh_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_acc_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_acc_hl_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_acc_hl_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_acc_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_acc_hl_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_acc_hl_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_acc_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_acc_lh_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_acc_lh_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_acc_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_acc_lh_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_acc_lh_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_acc_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_acc_ll_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_acc_ll_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_acc_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_acc_ll_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_acc_ll_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_acc_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_nac_hh_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_nac_hh_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_nac_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_nac_hh_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_nac_hh_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_nac_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_nac_hl_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_nac_hl_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_nac_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_nac_hl_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_nac_hl_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_nac_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_nac_lh_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_nac_lh_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_nac_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_nac_lh_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_nac_lh_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_nac_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_nac_ll_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_nac_ll_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_nac_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_nac_ll_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_nac_ll_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_nac_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_hh_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_hh_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_hh_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_hh_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_hl_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_hl_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_hl_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_hl_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_lh_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_lh_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_lh_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_lh_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_ll_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_ll_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_ll_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_ll_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_rnd_hh_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_rnd_hh_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_rnd_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_rnd_hh_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_rnd_hh_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_rnd_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_rnd_hl_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_rnd_hl_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_rnd_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_rnd_hl_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_rnd_hl_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_rnd_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_rnd_lh_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_rnd_lh_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_rnd_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_rnd_lh_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_rnd_lh_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_rnd_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_rnd_ll_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_rnd_ll_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_rnd_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_rnd_ll_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_rnd_ll_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_rnd_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_acc_hh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_acc_hh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_acc_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_acc_hh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_acc_hh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_acc_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_acc_hl_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_acc_hl_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_acc_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_acc_hl_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_acc_hl_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_acc_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_acc_lh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_acc_lh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_acc_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_acc_lh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_acc_lh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_acc_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_acc_ll_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_acc_ll_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_acc_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_acc_ll_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_acc_ll_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_acc_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_nac_hh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_nac_hh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_nac_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_nac_hh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_nac_hh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_nac_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_nac_hl_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_nac_hl_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_nac_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_nac_hl_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_nac_hl_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_nac_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_nac_lh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_nac_lh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_nac_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_nac_lh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_nac_lh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_nac_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_nac_ll_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_nac_ll_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_nac_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_nac_ll_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_nac_ll_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_nac_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_hh_s0,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_hh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpyu_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_hh_s1,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_hh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpyu_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_hl_s0,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_hl_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpyu_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_hl_s1,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_hl_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpyu_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_lh_s0,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_lh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpyu_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_lh_s1,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_lh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpyu_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_ll_s0,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_ll_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpyu_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_ll_s1,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_ll_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpyu_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_acc_hh_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_acc_hh_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_acc_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_acc_hh_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_acc_hh_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_acc_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_acc_hl_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_acc_hl_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_acc_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_acc_hl_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_acc_hl_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_acc_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_acc_lh_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_acc_lh_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_acc_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_acc_lh_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_acc_lh_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_acc_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_acc_ll_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_acc_ll_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_acc_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_acc_ll_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_acc_ll_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_acc_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_nac_hh_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_nac_hh_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_nac_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_nac_hh_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_nac_hh_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_nac_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_nac_hl_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_nac_hl_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_nac_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_nac_hl_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_nac_hl_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_nac_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_nac_lh_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_nac_lh_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_nac_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_nac_lh_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_nac_lh_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_nac_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_nac_ll_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_nac_ll_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_nac_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_nac_ll_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_nac_ll_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_nac_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_hh_s0,UDI_ftype_SISI,2)
//
def int_hexagon_M2_mpyud_hh_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyud_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_hh_s1,UDI_ftype_SISI,2)
//
def int_hexagon_M2_mpyud_hh_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyud_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_hl_s0,UDI_ftype_SISI,2)
//
def int_hexagon_M2_mpyud_hl_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyud_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_hl_s1,UDI_ftype_SISI,2)
//
def int_hexagon_M2_mpyud_hl_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyud_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_lh_s0,UDI_ftype_SISI,2)
//
def int_hexagon_M2_mpyud_lh_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyud_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_lh_s1,UDI_ftype_SISI,2)
//
def int_hexagon_M2_mpyud_lh_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyud_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_ll_s0,UDI_ftype_SISI,2)
//
def int_hexagon_M2_mpyud_ll_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyud_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_ll_s1,UDI_ftype_SISI,2)
//
def int_hexagon_M2_mpyud_ll_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyud_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpysmi,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpysmi :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpysmi">;
//
// BUILTIN_INFO(HEXAGON.M2_macsip,SI_ftype_SISISI,3)
//
def int_hexagon_M2_macsip :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_macsip">;
//
// BUILTIN_INFO(HEXAGON.M2_macsin,SI_ftype_SISISI,3)
//
def int_hexagon_M2_macsin :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_macsin">;
//
// BUILTIN_INFO(HEXAGON.M2_dpmpyss_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_dpmpyss_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_dpmpyss_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_dpmpyss_acc_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_dpmpyss_acc_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_dpmpyss_acc_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_dpmpyss_nac_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_dpmpyss_nac_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_dpmpyss_nac_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_dpmpyuu_s0,UDI_ftype_SISI,2)
//
def int_hexagon_M2_dpmpyuu_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_dpmpyuu_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_dpmpyuu_acc_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_dpmpyuu_acc_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_dpmpyuu_acc_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_dpmpyuu_nac_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_dpmpyuu_nac_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_dpmpyuu_nac_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_up,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_up :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_up">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_up_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_up_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_up_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_up_s1_sat,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_up_s1_sat :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_up_s1_sat">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_up,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_up :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpyu_up">;
//
// BUILTIN_INFO(HEXAGON.M2_mpysu_up,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpysu_up :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpysu_up">;
//
// BUILTIN_INFO(HEXAGON.M2_dpmpyss_rnd_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_dpmpyss_rnd_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_dpmpyss_rnd_s0">;
//
// BUILTIN_INFO(HEXAGON.M4_mac_up_s1_sat,SI_ftype_SISISI,3)
//
def int_hexagon_M4_mac_up_s1_sat :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_mac_up_s1_sat">;
//
// BUILTIN_INFO(HEXAGON.M4_nac_up_s1_sat,SI_ftype_SISISI,3)
//
def int_hexagon_M4_nac_up_s1_sat :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_nac_up_s1_sat">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyi,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpyi :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpyi">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyui,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpyui :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpyui">;
//
// BUILTIN_INFO(HEXAGON.M2_maci,SI_ftype_SISISI,3)
//
def int_hexagon_M2_maci :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_maci">;
//
// BUILTIN_INFO(HEXAGON.M2_acci,SI_ftype_SISISI,3)
//
def int_hexagon_M2_acci :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_acci">;
//
// BUILTIN_INFO(HEXAGON.M2_accii,SI_ftype_SISISI,3)
//
def int_hexagon_M2_accii :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_accii">;
//
// BUILTIN_INFO(HEXAGON.M2_nacci,SI_ftype_SISISI,3)
//
def int_hexagon_M2_nacci :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_nacci">;
//
// BUILTIN_INFO(HEXAGON.M2_naccii,SI_ftype_SISISI,3)
//
def int_hexagon_M2_naccii :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_naccii">;
//
// BUILTIN_INFO(HEXAGON.M2_subacc,SI_ftype_SISISI,3)
//
def int_hexagon_M2_subacc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_subacc">;
//
// BUILTIN_INFO(HEXAGON.M4_mpyrr_addr,SI_ftype_SISISI,3)
//
def int_hexagon_M4_mpyrr_addr :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_mpyrr_addr">;
//
// BUILTIN_INFO(HEXAGON.M4_mpyri_addr_u2,SI_ftype_SISISI,3)
//
def int_hexagon_M4_mpyri_addr_u2 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_mpyri_addr_u2">;
//
// BUILTIN_INFO(HEXAGON.M4_mpyri_addr,SI_ftype_SISISI,3)
//
def int_hexagon_M4_mpyri_addr :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_mpyri_addr">;
//
// BUILTIN_INFO(HEXAGON.M4_mpyri_addi,SI_ftype_SISISI,3)
//
def int_hexagon_M4_mpyri_addi :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_mpyri_addi">;
//
// BUILTIN_INFO(HEXAGON.M4_mpyrr_addi,SI_ftype_SISISI,3)
//
def int_hexagon_M4_mpyrr_addi :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_mpyrr_addi">;
//
// BUILTIN_INFO(HEXAGON.M2_vmpy2s_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_vmpy2s_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_vmpy2s_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vmpy2s_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_vmpy2s_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_vmpy2s_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vmac2s_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_vmac2s_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_vmac2s_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vmac2s_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_vmac2s_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_vmac2s_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vmpy2su_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_vmpy2su_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_vmpy2su_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vmpy2su_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_vmpy2su_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_vmpy2su_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vmac2su_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_vmac2su_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_vmac2su_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vmac2su_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_vmac2su_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_vmac2su_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vmpy2s_s0pack,SI_ftype_SISI,2)
//
def int_hexagon_M2_vmpy2s_s0pack :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_vmpy2s_s0pack">;
//
// BUILTIN_INFO(HEXAGON.M2_vmpy2s_s1pack,SI_ftype_SISI,2)
//
def int_hexagon_M2_vmpy2s_s1pack :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_vmpy2s_s1pack">;
//
// BUILTIN_INFO(HEXAGON.M2_vmac2,DI_ftype_DISISI,3)
//
def int_hexagon_M2_vmac2 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_vmac2">;
//
// BUILTIN_INFO(HEXAGON.M2_vmpy2es_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vmpy2es_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vmpy2es_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vmpy2es_s1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vmpy2es_s1 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vmpy2es_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vmac2es_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vmac2es_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_vmac2es_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vmac2es_s1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vmac2es_s1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_vmac2es_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vmac2es,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vmac2es :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_vmac2es">;
//
// BUILTIN_INFO(HEXAGON.M2_vrmac_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vrmac_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_vrmac_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vrmpy_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vrmpy_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vrmpy_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vdmpyrs_s0,SI_ftype_DIDI,2)
//
def int_hexagon_M2_vdmpyrs_s0 :
Hexagon_si_didi_Intrinsic<"HEXAGON_M2_vdmpyrs_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vdmpyrs_s1,SI_ftype_DIDI,2)
//
def int_hexagon_M2_vdmpyrs_s1 :
Hexagon_si_didi_Intrinsic<"HEXAGON_M2_vdmpyrs_s1">;
//
// BUILTIN_INFO(HEXAGON.M5_vrmpybuu,DI_ftype_DIDI,2)
//
def int_hexagon_M5_vrmpybuu :
Hexagon_di_didi_Intrinsic<"HEXAGON_M5_vrmpybuu">;
//
// BUILTIN_INFO(HEXAGON.M5_vrmacbuu,DI_ftype_DIDIDI,3)
//
def int_hexagon_M5_vrmacbuu :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M5_vrmacbuu">;
//
// BUILTIN_INFO(HEXAGON.M5_vrmpybsu,DI_ftype_DIDI,2)
//
def int_hexagon_M5_vrmpybsu :
Hexagon_di_didi_Intrinsic<"HEXAGON_M5_vrmpybsu">;
//
// BUILTIN_INFO(HEXAGON.M5_vrmacbsu,DI_ftype_DIDIDI,3)
//
def int_hexagon_M5_vrmacbsu :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M5_vrmacbsu">;
//
// BUILTIN_INFO(HEXAGON.M5_vmpybuu,DI_ftype_SISI,2)
//
def int_hexagon_M5_vmpybuu :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M5_vmpybuu">;
//
// BUILTIN_INFO(HEXAGON.M5_vmpybsu,DI_ftype_SISI,2)
//
def int_hexagon_M5_vmpybsu :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M5_vmpybsu">;
//
// BUILTIN_INFO(HEXAGON.M5_vmacbuu,DI_ftype_DISISI,3)
//
def int_hexagon_M5_vmacbuu :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M5_vmacbuu">;
//
// BUILTIN_INFO(HEXAGON.M5_vmacbsu,DI_ftype_DISISI,3)
//
def int_hexagon_M5_vmacbsu :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M5_vmacbsu">;
//
// BUILTIN_INFO(HEXAGON.M5_vdmpybsu,DI_ftype_DIDI,2)
//
def int_hexagon_M5_vdmpybsu :
Hexagon_di_didi_Intrinsic<"HEXAGON_M5_vdmpybsu">;
//
// BUILTIN_INFO(HEXAGON.M5_vdmacbsu,DI_ftype_DIDIDI,3)
//
def int_hexagon_M5_vdmacbsu :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M5_vdmacbsu">;
//
// BUILTIN_INFO(HEXAGON.M2_vdmacs_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vdmacs_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_vdmacs_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vdmacs_s1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vdmacs_s1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_vdmacs_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vdmpys_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vdmpys_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vdmpys_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vdmpys_s1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vdmpys_s1 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vdmpys_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpyrs_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_cmpyrs_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_cmpyrs_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpyrs_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_cmpyrs_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_cmpyrs_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpyrsc_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_cmpyrsc_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_cmpyrsc_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpyrsc_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_cmpyrsc_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_cmpyrsc_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_cmacs_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cmacs_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_cmacs_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cmacs_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cmacs_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_cmacs_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_cmacsc_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cmacsc_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_cmacsc_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cmacsc_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cmacsc_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_cmacsc_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpys_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_cmpys_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_cmpys_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpys_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_cmpys_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_cmpys_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpysc_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_cmpysc_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_cmpysc_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpysc_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_cmpysc_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_cmpysc_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_cnacs_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cnacs_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_cnacs_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cnacs_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cnacs_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_cnacs_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_cnacsc_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cnacsc_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_cnacsc_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cnacsc_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cnacsc_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_cnacsc_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmpys_s1,DI_ftype_DISI,2)
//
def int_hexagon_M2_vrcmpys_s1 :
Hexagon_di_disi_Intrinsic<"HEXAGON_M2_vrcmpys_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmpys_acc_s1,DI_ftype_DIDISI,3)
//
def int_hexagon_M2_vrcmpys_acc_s1 :
Hexagon_di_didisi_Intrinsic<"HEXAGON_M2_vrcmpys_acc_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmpys_s1rp,SI_ftype_DISI,2)
//
def int_hexagon_M2_vrcmpys_s1rp :
Hexagon_si_disi_Intrinsic<"HEXAGON_M2_vrcmpys_s1rp">;
//
// BUILTIN_INFO(HEXAGON.M2_mmacls_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmacls_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmacls_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmacls_s1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmacls_s1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmacls_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmachs_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmachs_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmachs_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmachs_s1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmachs_s1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmachs_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyl_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyl_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyl_s1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyl_s1 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyh_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyh_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyh_s1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyh_s1 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmacls_rs0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmacls_rs0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmacls_rs0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmacls_rs1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmacls_rs1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmacls_rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmachs_rs0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmachs_rs0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmachs_rs0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmachs_rs1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmachs_rs1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmachs_rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyl_rs0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyl_rs0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyl_rs0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyl_rs1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyl_rs1 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyl_rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyh_rs0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyh_rs0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyh_rs0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyh_rs1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyh_rs1 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyh_rs1">;
//
// BUILTIN_INFO(HEXAGON.M4_vrmpyeh_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M4_vrmpyeh_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M4_vrmpyeh_s0">;
//
// BUILTIN_INFO(HEXAGON.M4_vrmpyeh_s1,DI_ftype_DIDI,2)
//
def int_hexagon_M4_vrmpyeh_s1 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M4_vrmpyeh_s1">;
//
// BUILTIN_INFO(HEXAGON.M4_vrmpyeh_acc_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M4_vrmpyeh_acc_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M4_vrmpyeh_acc_s0">;
//
// BUILTIN_INFO(HEXAGON.M4_vrmpyeh_acc_s1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M4_vrmpyeh_acc_s1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M4_vrmpyeh_acc_s1">;
//
// BUILTIN_INFO(HEXAGON.M4_vrmpyoh_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M4_vrmpyoh_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M4_vrmpyoh_s0">;
//
// BUILTIN_INFO(HEXAGON.M4_vrmpyoh_s1,DI_ftype_DIDI,2)
//
def int_hexagon_M4_vrmpyoh_s1 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M4_vrmpyoh_s1">;
//
// BUILTIN_INFO(HEXAGON.M4_vrmpyoh_acc_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M4_vrmpyoh_acc_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M4_vrmpyoh_acc_s0">;
//
// BUILTIN_INFO(HEXAGON.M4_vrmpyoh_acc_s1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M4_vrmpyoh_acc_s1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M4_vrmpyoh_acc_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_hmmpyl_rs1,SI_ftype_SISI,2)
//
def int_hexagon_M2_hmmpyl_rs1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_hmmpyl_rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_hmmpyh_rs1,SI_ftype_SISI,2)
//
def int_hexagon_M2_hmmpyh_rs1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_hmmpyh_rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_hmmpyl_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_hmmpyl_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_hmmpyl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_hmmpyh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_hmmpyh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_hmmpyh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmaculs_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmaculs_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmaculs_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmaculs_s1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmaculs_s1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmaculs_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmacuhs_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmacuhs_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmacuhs_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmacuhs_s1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmacuhs_s1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmacuhs_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyul_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyul_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyul_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyul_s1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyul_s1 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyul_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyuh_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyuh_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyuh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyuh_s1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyuh_s1 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyuh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmaculs_rs0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmaculs_rs0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmaculs_rs0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmaculs_rs1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmaculs_rs1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmaculs_rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmacuhs_rs0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmacuhs_rs0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmacuhs_rs0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmacuhs_rs1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmacuhs_rs1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmacuhs_rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyul_rs0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyul_rs0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyul_rs0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyul_rs1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyul_rs1 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyul_rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyuh_rs0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyuh_rs0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyuh_rs0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyuh_rs1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyuh_rs1 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyuh_rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmaci_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vrcmaci_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_vrcmaci_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmacr_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vrcmacr_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_vrcmacr_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmaci_s0c,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vrcmaci_s0c :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_vrcmaci_s0c">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmacr_s0c,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vrcmacr_s0c :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_vrcmacr_s0c">;
//
// BUILTIN_INFO(HEXAGON.M2_cmaci_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cmaci_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_cmaci_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cmacr_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cmacr_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_cmacr_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmpyi_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vrcmpyi_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vrcmpyi_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmpyr_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vrcmpyr_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vrcmpyr_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmpyi_s0c,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vrcmpyi_s0c :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vrcmpyi_s0c">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmpyr_s0c,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vrcmpyr_s0c :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vrcmpyr_s0c">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpyi_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_cmpyi_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_cmpyi_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpyr_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_cmpyr_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_cmpyr_s0">;
//
// BUILTIN_INFO(HEXAGON.M4_cmpyi_wh,SI_ftype_DISI,2)
//
def int_hexagon_M4_cmpyi_wh :
Hexagon_si_disi_Intrinsic<"HEXAGON_M4_cmpyi_wh">;
//
// BUILTIN_INFO(HEXAGON.M4_cmpyr_wh,SI_ftype_DISI,2)
//
def int_hexagon_M4_cmpyr_wh :
Hexagon_si_disi_Intrinsic<"HEXAGON_M4_cmpyr_wh">;
//
// BUILTIN_INFO(HEXAGON.M4_cmpyi_whc,SI_ftype_DISI,2)
//
def int_hexagon_M4_cmpyi_whc :
Hexagon_si_disi_Intrinsic<"HEXAGON_M4_cmpyi_whc">;
//
// BUILTIN_INFO(HEXAGON.M4_cmpyr_whc,SI_ftype_DISI,2)
//
def int_hexagon_M4_cmpyr_whc :
Hexagon_si_disi_Intrinsic<"HEXAGON_M4_cmpyr_whc">;
//
// BUILTIN_INFO(HEXAGON.M2_vcmpy_s0_sat_i,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vcmpy_s0_sat_i :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vcmpy_s0_sat_i">;
//
// BUILTIN_INFO(HEXAGON.M2_vcmpy_s0_sat_r,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vcmpy_s0_sat_r :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vcmpy_s0_sat_r">;
//
// BUILTIN_INFO(HEXAGON.M2_vcmpy_s1_sat_i,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vcmpy_s1_sat_i :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vcmpy_s1_sat_i">;
//
// BUILTIN_INFO(HEXAGON.M2_vcmpy_s1_sat_r,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vcmpy_s1_sat_r :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vcmpy_s1_sat_r">;
//
// BUILTIN_INFO(HEXAGON.M2_vcmac_s0_sat_i,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vcmac_s0_sat_i :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_vcmac_s0_sat_i">;
//
// BUILTIN_INFO(HEXAGON.M2_vcmac_s0_sat_r,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vcmac_s0_sat_r :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_vcmac_s0_sat_r">;
//
// BUILTIN_INFO(HEXAGON.S2_vcrotate,DI_ftype_DISI,2)
//
def int_hexagon_S2_vcrotate :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_vcrotate">;
//
// BUILTIN_INFO(HEXAGON.S4_vrcrotate_acc,DI_ftype_DIDISISI,4)
//
def int_hexagon_S4_vrcrotate_acc :
Hexagon_di_didisisi_Intrinsic<"HEXAGON_S4_vrcrotate_acc">;
//
// BUILTIN_INFO(HEXAGON.S4_vrcrotate,DI_ftype_DISISI,3)
//
def int_hexagon_S4_vrcrotate :
Hexagon_di_disisi_Intrinsic<"HEXAGON_S4_vrcrotate">;
//
// BUILTIN_INFO(HEXAGON.S2_vcnegh,DI_ftype_DISI,2)
//
def int_hexagon_S2_vcnegh :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_vcnegh">;
//
// BUILTIN_INFO(HEXAGON.S2_vrcnegh,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_vrcnegh :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_vrcnegh">;
//
// BUILTIN_INFO(HEXAGON.M4_pmpyw,DI_ftype_SISI,2)
//
def int_hexagon_M4_pmpyw :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M4_pmpyw">;
//
// BUILTIN_INFO(HEXAGON.M4_vpmpyh,DI_ftype_SISI,2)
//
def int_hexagon_M4_vpmpyh :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M4_vpmpyh">;
//
// BUILTIN_INFO(HEXAGON.M4_pmpyw_acc,DI_ftype_DISISI,3)
//
def int_hexagon_M4_pmpyw_acc :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M4_pmpyw_acc">;
//
// BUILTIN_INFO(HEXAGON.M4_vpmpyh_acc,DI_ftype_DISISI,3)
//
def int_hexagon_M4_vpmpyh_acc :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M4_vpmpyh_acc">;
//
// BUILTIN_INFO(HEXAGON.A2_add,SI_ftype_SISI,2)
//
def int_hexagon_A2_add :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_add">;
//
// BUILTIN_INFO(HEXAGON.A2_sub,SI_ftype_SISI,2)
//
def int_hexagon_A2_sub :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_sub">;
//
// BUILTIN_INFO(HEXAGON.A2_addsat,SI_ftype_SISI,2)
//
def int_hexagon_A2_addsat :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addsat">;
//
// BUILTIN_INFO(HEXAGON.A2_subsat,SI_ftype_SISI,2)
//
def int_hexagon_A2_subsat :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subsat">;
//
// BUILTIN_INFO(HEXAGON.A2_addi,SI_ftype_SISI,2)
//
def int_hexagon_A2_addi :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addi">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_l16_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_l16_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addh_l16_ll">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_l16_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_l16_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addh_l16_hl">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_l16_sat_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_l16_sat_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addh_l16_sat_ll">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_l16_sat_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_l16_sat_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addh_l16_sat_hl">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_l16_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_l16_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subh_l16_ll">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_l16_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_l16_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subh_l16_hl">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_l16_sat_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_l16_sat_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subh_l16_sat_ll">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_l16_sat_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_l16_sat_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subh_l16_sat_hl">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_h16_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_h16_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addh_h16_ll">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_h16_lh,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_h16_lh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addh_h16_lh">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_h16_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_h16_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addh_h16_hl">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_h16_hh,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_h16_hh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addh_h16_hh">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_h16_sat_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_h16_sat_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addh_h16_sat_ll">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_h16_sat_lh,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_h16_sat_lh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addh_h16_sat_lh">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_h16_sat_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_h16_sat_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addh_h16_sat_hl">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_h16_sat_hh,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_h16_sat_hh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addh_h16_sat_hh">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_h16_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_h16_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subh_h16_ll">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_h16_lh,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_h16_lh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subh_h16_lh">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_h16_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_h16_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subh_h16_hl">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_h16_hh,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_h16_hh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subh_h16_hh">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_h16_sat_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_h16_sat_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subh_h16_sat_ll">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_h16_sat_lh,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_h16_sat_lh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subh_h16_sat_lh">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_h16_sat_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_h16_sat_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subh_h16_sat_hl">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_h16_sat_hh,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_h16_sat_hh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subh_h16_sat_hh">;
//
// BUILTIN_INFO(HEXAGON.A2_aslh,SI_ftype_SI,1)
//
def int_hexagon_A2_aslh :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_aslh">;
//
// BUILTIN_INFO(HEXAGON.A2_asrh,SI_ftype_SI,1)
//
def int_hexagon_A2_asrh :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_asrh">;
//
// BUILTIN_INFO(HEXAGON.A2_addp,DI_ftype_DIDI,2)
//
def int_hexagon_A2_addp :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_addp">;
//
// BUILTIN_INFO(HEXAGON.A2_addpsat,DI_ftype_DIDI,2)
//
def int_hexagon_A2_addpsat :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_addpsat">;
//
// BUILTIN_INFO(HEXAGON.A2_addsp,DI_ftype_SIDI,2)
//
def int_hexagon_A2_addsp :
Hexagon_di_sidi_Intrinsic<"HEXAGON_A2_addsp">;
//
// BUILTIN_INFO(HEXAGON.A2_subp,DI_ftype_DIDI,2)
//
def int_hexagon_A2_subp :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_subp">;
//
// BUILTIN_INFO(HEXAGON.A2_neg,SI_ftype_SI,1)
//
def int_hexagon_A2_neg :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_neg">;
//
// BUILTIN_INFO(HEXAGON.A2_negsat,SI_ftype_SI,1)
//
def int_hexagon_A2_negsat :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_negsat">;
//
// BUILTIN_INFO(HEXAGON.A2_abs,SI_ftype_SI,1)
//
def int_hexagon_A2_abs :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_abs">;
//
// BUILTIN_INFO(HEXAGON.A2_abssat,SI_ftype_SI,1)
//
def int_hexagon_A2_abssat :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_abssat">;
//
// BUILTIN_INFO(HEXAGON.A2_vconj,DI_ftype_DI,1)
//
def int_hexagon_A2_vconj :
Hexagon_di_di_Intrinsic<"HEXAGON_A2_vconj">;
//
// BUILTIN_INFO(HEXAGON.A2_negp,DI_ftype_DI,1)
//
def int_hexagon_A2_negp :
Hexagon_di_di_Intrinsic<"HEXAGON_A2_negp">;
//
// BUILTIN_INFO(HEXAGON.A2_absp,DI_ftype_DI,1)
//
def int_hexagon_A2_absp :
Hexagon_di_di_Intrinsic<"HEXAGON_A2_absp">;
//
// BUILTIN_INFO(HEXAGON.A2_max,SI_ftype_SISI,2)
//
def int_hexagon_A2_max :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_max">;
//
// BUILTIN_INFO(HEXAGON.A2_maxu,USI_ftype_SISI,2)
//
def int_hexagon_A2_maxu :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_maxu">;
//
// BUILTIN_INFO(HEXAGON.A2_min,SI_ftype_SISI,2)
//
def int_hexagon_A2_min :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_min">;
//
// BUILTIN_INFO(HEXAGON.A2_minu,USI_ftype_SISI,2)
//
def int_hexagon_A2_minu :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_minu">;
//
// BUILTIN_INFO(HEXAGON.A2_maxp,DI_ftype_DIDI,2)
//
def int_hexagon_A2_maxp :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_maxp">;
//
// BUILTIN_INFO(HEXAGON.A2_maxup,UDI_ftype_DIDI,2)
//
def int_hexagon_A2_maxup :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_maxup">;
//
// BUILTIN_INFO(HEXAGON.A2_minp,DI_ftype_DIDI,2)
//
def int_hexagon_A2_minp :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_minp">;
//
// BUILTIN_INFO(HEXAGON.A2_minup,UDI_ftype_DIDI,2)
//
def int_hexagon_A2_minup :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_minup">;
//
// BUILTIN_INFO(HEXAGON.A2_tfr,SI_ftype_SI,1)
//
def int_hexagon_A2_tfr :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_tfr">;
//
// BUILTIN_INFO(HEXAGON.A2_tfrsi,SI_ftype_SI,1)
//
def int_hexagon_A2_tfrsi :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_tfrsi">;
//
// BUILTIN_INFO(HEXAGON.A2_tfrp,DI_ftype_DI,1)
//
def int_hexagon_A2_tfrp :
Hexagon_di_di_Intrinsic<"HEXAGON_A2_tfrp">;
//
// BUILTIN_INFO(HEXAGON.A2_tfrpi,DI_ftype_SI,1)
//
def int_hexagon_A2_tfrpi :
Hexagon_di_si_Intrinsic<"HEXAGON_A2_tfrpi">;
//
// BUILTIN_INFO(HEXAGON.A2_zxtb,SI_ftype_SI,1)
//
def int_hexagon_A2_zxtb :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_zxtb">;
//
// BUILTIN_INFO(HEXAGON.A2_sxtb,SI_ftype_SI,1)
//
def int_hexagon_A2_sxtb :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_sxtb">;
//
// BUILTIN_INFO(HEXAGON.A2_zxth,SI_ftype_SI,1)
//
def int_hexagon_A2_zxth :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_zxth">;
//
// BUILTIN_INFO(HEXAGON.A2_sxth,SI_ftype_SI,1)
//
def int_hexagon_A2_sxth :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_sxth">;
//
// BUILTIN_INFO(HEXAGON.A2_combinew,DI_ftype_SISI,2)
//
def int_hexagon_A2_combinew :
Hexagon_di_sisi_Intrinsic<"HEXAGON_A2_combinew">;
//
// BUILTIN_INFO(HEXAGON.A4_combineri,DI_ftype_SISI,2)
//
def int_hexagon_A4_combineri :
Hexagon_di_sisi_Intrinsic<"HEXAGON_A4_combineri">;
//
// BUILTIN_INFO(HEXAGON.A4_combineir,DI_ftype_SISI,2)
//
def int_hexagon_A4_combineir :
Hexagon_di_sisi_Intrinsic<"HEXAGON_A4_combineir">;
//
// BUILTIN_INFO(HEXAGON.A2_combineii,DI_ftype_SISI,2)
//
def int_hexagon_A2_combineii :
Hexagon_di_sisi_Intrinsic<"HEXAGON_A2_combineii">;
//
// BUILTIN_INFO(HEXAGON.A2_combine_hh,SI_ftype_SISI,2)
//
def int_hexagon_A2_combine_hh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_combine_hh">;
//
// BUILTIN_INFO(HEXAGON.A2_combine_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_combine_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_combine_hl">;
//
// BUILTIN_INFO(HEXAGON.A2_combine_lh,SI_ftype_SISI,2)
//
def int_hexagon_A2_combine_lh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_combine_lh">;
//
// BUILTIN_INFO(HEXAGON.A2_combine_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_combine_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_combine_ll">;
//
// BUILTIN_INFO(HEXAGON.A2_tfril,SI_ftype_SISI,2)
//
def int_hexagon_A2_tfril :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_tfril">;
//
// BUILTIN_INFO(HEXAGON.A2_tfrih,SI_ftype_SISI,2)
//
def int_hexagon_A2_tfrih :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_tfrih">;
//
// BUILTIN_INFO(HEXAGON.A2_and,SI_ftype_SISI,2)
//
def int_hexagon_A2_and :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_and">;
//
// BUILTIN_INFO(HEXAGON.A2_or,SI_ftype_SISI,2)
//
def int_hexagon_A2_or :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_or">;
//
// BUILTIN_INFO(HEXAGON.A2_xor,SI_ftype_SISI,2)
//
def int_hexagon_A2_xor :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_xor">;
//
// BUILTIN_INFO(HEXAGON.A2_not,SI_ftype_SI,1)
//
def int_hexagon_A2_not :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_not">;
//
// BUILTIN_INFO(HEXAGON.M2_xor_xacc,SI_ftype_SISISI,3)
//
def int_hexagon_M2_xor_xacc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_xor_xacc">;
//
// BUILTIN_INFO(HEXAGON.M4_xor_xacc,DI_ftype_DIDIDI,3)
//
def int_hexagon_M4_xor_xacc :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M4_xor_xacc">;
//
// BUILTIN_INFO(HEXAGON.A4_andn,SI_ftype_SISI,2)
//
def int_hexagon_A4_andn :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_andn">;
//
// BUILTIN_INFO(HEXAGON.A4_orn,SI_ftype_SISI,2)
//
def int_hexagon_A4_orn :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_orn">;
//
// BUILTIN_INFO(HEXAGON.A4_andnp,DI_ftype_DIDI,2)
//
def int_hexagon_A4_andnp :
Hexagon_di_didi_Intrinsic<"HEXAGON_A4_andnp">;
//
// BUILTIN_INFO(HEXAGON.A4_ornp,DI_ftype_DIDI,2)
//
def int_hexagon_A4_ornp :
Hexagon_di_didi_Intrinsic<"HEXAGON_A4_ornp">;
//
// BUILTIN_INFO(HEXAGON.S4_addaddi,SI_ftype_SISISI,3)
//
def int_hexagon_S4_addaddi :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_addaddi">;
//
// BUILTIN_INFO(HEXAGON.S4_subaddi,SI_ftype_SISISI,3)
//
def int_hexagon_S4_subaddi :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_subaddi">;
//
// BUILTIN_INFO(HEXAGON.M4_and_and,SI_ftype_SISISI,3)
//
def int_hexagon_M4_and_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_and_and">;
//
// BUILTIN_INFO(HEXAGON.M4_and_andn,SI_ftype_SISISI,3)
//
def int_hexagon_M4_and_andn :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_and_andn">;
//
// BUILTIN_INFO(HEXAGON.M4_and_or,SI_ftype_SISISI,3)
//
def int_hexagon_M4_and_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_and_or">;
//
// BUILTIN_INFO(HEXAGON.M4_and_xor,SI_ftype_SISISI,3)
//
def int_hexagon_M4_and_xor :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_and_xor">;
//
// BUILTIN_INFO(HEXAGON.M4_or_and,SI_ftype_SISISI,3)
//
def int_hexagon_M4_or_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_or_and">;
//
// BUILTIN_INFO(HEXAGON.M4_or_andn,SI_ftype_SISISI,3)
//
def int_hexagon_M4_or_andn :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_or_andn">;
//
// BUILTIN_INFO(HEXAGON.M4_or_or,SI_ftype_SISISI,3)
//
def int_hexagon_M4_or_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_or_or">;
//
// BUILTIN_INFO(HEXAGON.M4_or_xor,SI_ftype_SISISI,3)
//
def int_hexagon_M4_or_xor :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_or_xor">;
//
// BUILTIN_INFO(HEXAGON.S4_or_andix,SI_ftype_SISISI,3)
//
def int_hexagon_S4_or_andix :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_or_andix">;
//
// BUILTIN_INFO(HEXAGON.S4_or_andi,SI_ftype_SISISI,3)
//
def int_hexagon_S4_or_andi :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_or_andi">;
//
// BUILTIN_INFO(HEXAGON.S4_or_ori,SI_ftype_SISISI,3)
//
def int_hexagon_S4_or_ori :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_or_ori">;
//
// BUILTIN_INFO(HEXAGON.M4_xor_and,SI_ftype_SISISI,3)
//
def int_hexagon_M4_xor_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_xor_and">;
//
// BUILTIN_INFO(HEXAGON.M4_xor_or,SI_ftype_SISISI,3)
//
def int_hexagon_M4_xor_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_xor_or">;
//
// BUILTIN_INFO(HEXAGON.M4_xor_andn,SI_ftype_SISISI,3)
//
def int_hexagon_M4_xor_andn :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_xor_andn">;
//
// BUILTIN_INFO(HEXAGON.A2_subri,SI_ftype_SISI,2)
//
def int_hexagon_A2_subri :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subri">;
//
// BUILTIN_INFO(HEXAGON.A2_andir,SI_ftype_SISI,2)
//
def int_hexagon_A2_andir :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_andir">;
//
// BUILTIN_INFO(HEXAGON.A2_orir,SI_ftype_SISI,2)
//
def int_hexagon_A2_orir :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_orir">;
//
// BUILTIN_INFO(HEXAGON.A2_andp,DI_ftype_DIDI,2)
//
def int_hexagon_A2_andp :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_andp">;
//
// BUILTIN_INFO(HEXAGON.A2_orp,DI_ftype_DIDI,2)
//
def int_hexagon_A2_orp :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_orp">;
//
// BUILTIN_INFO(HEXAGON.A2_xorp,DI_ftype_DIDI,2)
//
def int_hexagon_A2_xorp :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_xorp">;
//
// BUILTIN_INFO(HEXAGON.A2_notp,DI_ftype_DI,1)
//
def int_hexagon_A2_notp :
Hexagon_di_di_Intrinsic<"HEXAGON_A2_notp">;
//
// BUILTIN_INFO(HEXAGON.A2_sxtw,DI_ftype_SI,1)
//
def int_hexagon_A2_sxtw :
Hexagon_di_si_Intrinsic<"HEXAGON_A2_sxtw">;
//
// BUILTIN_INFO(HEXAGON.A2_sat,SI_ftype_DI,1)
//
def int_hexagon_A2_sat :
Hexagon_si_di_Intrinsic<"HEXAGON_A2_sat">;
//
// BUILTIN_INFO(HEXAGON.A2_roundsat,SI_ftype_DI,1)
//
def int_hexagon_A2_roundsat :
Hexagon_si_di_Intrinsic<"HEXAGON_A2_roundsat">;
//
// BUILTIN_INFO(HEXAGON.A2_sath,SI_ftype_SI,1)
//
def int_hexagon_A2_sath :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_sath">;
//
// BUILTIN_INFO(HEXAGON.A2_satuh,SI_ftype_SI,1)
//
def int_hexagon_A2_satuh :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_satuh">;
//
// BUILTIN_INFO(HEXAGON.A2_satub,SI_ftype_SI,1)
//
def int_hexagon_A2_satub :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_satub">;
//
// BUILTIN_INFO(HEXAGON.A2_satb,SI_ftype_SI,1)
//
def int_hexagon_A2_satb :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_satb">;
//
// BUILTIN_INFO(HEXAGON.A2_vaddub,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vaddub :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vaddub">;
//
// BUILTIN_INFO(HEXAGON.A2_vaddb_map,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vaddb_map :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vaddb_map">;
//
// BUILTIN_INFO(HEXAGON.A2_vaddubs,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vaddubs :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vaddubs">;
//
// BUILTIN_INFO(HEXAGON.A2_vaddh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vaddh :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vaddh">;
//
// BUILTIN_INFO(HEXAGON.A2_vaddhs,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vaddhs :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vaddhs">;
//
// BUILTIN_INFO(HEXAGON.A2_vadduhs,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vadduhs :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vadduhs">;
//
// BUILTIN_INFO(HEXAGON.A5_vaddhubs,SI_ftype_DIDI,2)
//
def int_hexagon_A5_vaddhubs :
Hexagon_si_didi_Intrinsic<"HEXAGON_A5_vaddhubs">;
//
// BUILTIN_INFO(HEXAGON.A2_vaddw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vaddw :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vaddw">;
//
// BUILTIN_INFO(HEXAGON.A2_vaddws,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vaddws :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vaddws">;
//
// BUILTIN_INFO(HEXAGON.S4_vxaddsubw,DI_ftype_DIDI,2)
//
def int_hexagon_S4_vxaddsubw :
Hexagon_di_didi_Intrinsic<"HEXAGON_S4_vxaddsubw">;
//
// BUILTIN_INFO(HEXAGON.S4_vxsubaddw,DI_ftype_DIDI,2)
//
def int_hexagon_S4_vxsubaddw :
Hexagon_di_didi_Intrinsic<"HEXAGON_S4_vxsubaddw">;
//
// BUILTIN_INFO(HEXAGON.S4_vxaddsubh,DI_ftype_DIDI,2)
//
def int_hexagon_S4_vxaddsubh :
Hexagon_di_didi_Intrinsic<"HEXAGON_S4_vxaddsubh">;
//
// BUILTIN_INFO(HEXAGON.S4_vxsubaddh,DI_ftype_DIDI,2)
//
def int_hexagon_S4_vxsubaddh :
Hexagon_di_didi_Intrinsic<"HEXAGON_S4_vxsubaddh">;
//
// BUILTIN_INFO(HEXAGON.S4_vxaddsubhr,DI_ftype_DIDI,2)
//
def int_hexagon_S4_vxaddsubhr :
Hexagon_di_didi_Intrinsic<"HEXAGON_S4_vxaddsubhr">;
//
// BUILTIN_INFO(HEXAGON.S4_vxsubaddhr,DI_ftype_DIDI,2)
//
def int_hexagon_S4_vxsubaddhr :
Hexagon_di_didi_Intrinsic<"HEXAGON_S4_vxsubaddhr">;
//
// BUILTIN_INFO(HEXAGON.A2_svavgh,SI_ftype_SISI,2)
//
def int_hexagon_A2_svavgh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_svavgh">;
//
// BUILTIN_INFO(HEXAGON.A2_svavghs,SI_ftype_SISI,2)
//
def int_hexagon_A2_svavghs :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_svavghs">;
//
// BUILTIN_INFO(HEXAGON.A2_svnavgh,SI_ftype_SISI,2)
//
def int_hexagon_A2_svnavgh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_svnavgh">;
//
// BUILTIN_INFO(HEXAGON.A2_svaddh,SI_ftype_SISI,2)
//
def int_hexagon_A2_svaddh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_svaddh">;
//
// BUILTIN_INFO(HEXAGON.A2_svaddhs,SI_ftype_SISI,2)
//
def int_hexagon_A2_svaddhs :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_svaddhs">;
//
// BUILTIN_INFO(HEXAGON.A2_svadduhs,SI_ftype_SISI,2)
//
def int_hexagon_A2_svadduhs :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_svadduhs">;
//
// BUILTIN_INFO(HEXAGON.A2_svsubh,SI_ftype_SISI,2)
//
def int_hexagon_A2_svsubh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_svsubh">;
//
// BUILTIN_INFO(HEXAGON.A2_svsubhs,SI_ftype_SISI,2)
//
def int_hexagon_A2_svsubhs :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_svsubhs">;
//
// BUILTIN_INFO(HEXAGON.A2_svsubuhs,SI_ftype_SISI,2)
//
def int_hexagon_A2_svsubuhs :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_svsubuhs">;
//
// BUILTIN_INFO(HEXAGON.A2_vraddub,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vraddub :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vraddub">;
//
// BUILTIN_INFO(HEXAGON.A2_vraddub_acc,DI_ftype_DIDIDI,3)
//
def int_hexagon_A2_vraddub_acc :
Hexagon_di_dididi_Intrinsic<"HEXAGON_A2_vraddub_acc">;
//
// BUILTIN_INFO(HEXAGON.M2_vraddh,SI_ftype_DIDI,2)
//
def int_hexagon_M2_vraddh :
Hexagon_si_didi_Intrinsic<"HEXAGON_M2_vraddh">;
//
// BUILTIN_INFO(HEXAGON.M2_vradduh,SI_ftype_DIDI,2)
//
def int_hexagon_M2_vradduh :
Hexagon_si_didi_Intrinsic<"HEXAGON_M2_vradduh">;
//
// BUILTIN_INFO(HEXAGON.A2_vsubub,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vsubub :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vsubub">;
//
// BUILTIN_INFO(HEXAGON.A2_vsubb_map,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vsubb_map :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vsubb_map">;
//
// BUILTIN_INFO(HEXAGON.A2_vsububs,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vsububs :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vsububs">;
//
// BUILTIN_INFO(HEXAGON.A2_vsubh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vsubh :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vsubh">;
//
// BUILTIN_INFO(HEXAGON.A2_vsubhs,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vsubhs :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vsubhs">;
//
// BUILTIN_INFO(HEXAGON.A2_vsubuhs,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vsubuhs :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vsubuhs">;
//
// BUILTIN_INFO(HEXAGON.A2_vsubw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vsubw :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vsubw">;
//
// BUILTIN_INFO(HEXAGON.A2_vsubws,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vsubws :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vsubws">;
//
// BUILTIN_INFO(HEXAGON.A2_vabsh,DI_ftype_DI,1)
//
def int_hexagon_A2_vabsh :
Hexagon_di_di_Intrinsic<"HEXAGON_A2_vabsh">;
//
// BUILTIN_INFO(HEXAGON.A2_vabshsat,DI_ftype_DI,1)
//
def int_hexagon_A2_vabshsat :
Hexagon_di_di_Intrinsic<"HEXAGON_A2_vabshsat">;
//
// BUILTIN_INFO(HEXAGON.A2_vabsw,DI_ftype_DI,1)
//
def int_hexagon_A2_vabsw :
Hexagon_di_di_Intrinsic<"HEXAGON_A2_vabsw">;
//
// BUILTIN_INFO(HEXAGON.A2_vabswsat,DI_ftype_DI,1)
//
def int_hexagon_A2_vabswsat :
Hexagon_di_di_Intrinsic<"HEXAGON_A2_vabswsat">;
//
// BUILTIN_INFO(HEXAGON.M2_vabsdiffw,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vabsdiffw :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vabsdiffw">;
//
// BUILTIN_INFO(HEXAGON.M2_vabsdiffh,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vabsdiffh :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vabsdiffh">;
//
// BUILTIN_INFO(HEXAGON.A2_vrsadub,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vrsadub :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vrsadub">;
//
// BUILTIN_INFO(HEXAGON.A2_vrsadub_acc,DI_ftype_DIDIDI,3)
//
def int_hexagon_A2_vrsadub_acc :
Hexagon_di_dididi_Intrinsic<"HEXAGON_A2_vrsadub_acc">;
//
// BUILTIN_INFO(HEXAGON.A2_vavgub,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavgub :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vavgub">;
//
// BUILTIN_INFO(HEXAGON.A2_vavguh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavguh :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vavguh">;
//
// BUILTIN_INFO(HEXAGON.A2_vavgh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavgh :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vavgh">;
//
// BUILTIN_INFO(HEXAGON.A2_vnavgh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vnavgh :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vnavgh">;
//
// BUILTIN_INFO(HEXAGON.A2_vavgw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavgw :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vavgw">;
//
// BUILTIN_INFO(HEXAGON.A2_vnavgw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vnavgw :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vnavgw">;
//
// BUILTIN_INFO(HEXAGON.A2_vavgwr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavgwr :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vavgwr">;
//
// BUILTIN_INFO(HEXAGON.A2_vnavgwr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vnavgwr :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vnavgwr">;
//
// BUILTIN_INFO(HEXAGON.A2_vavgwcr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavgwcr :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vavgwcr">;
//
// BUILTIN_INFO(HEXAGON.A2_vnavgwcr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vnavgwcr :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vnavgwcr">;
//
// BUILTIN_INFO(HEXAGON.A2_vavghcr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavghcr :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vavghcr">;
//
// BUILTIN_INFO(HEXAGON.A2_vnavghcr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vnavghcr :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vnavghcr">;
//
// BUILTIN_INFO(HEXAGON.A2_vavguw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavguw :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vavguw">;
//
// BUILTIN_INFO(HEXAGON.A2_vavguwr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavguwr :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vavguwr">;
//
// BUILTIN_INFO(HEXAGON.A2_vavgubr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavgubr :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vavgubr">;
//
// BUILTIN_INFO(HEXAGON.A2_vavguhr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavguhr :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vavguhr">;
//
// BUILTIN_INFO(HEXAGON.A2_vavghr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavghr :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vavghr">;
//
// BUILTIN_INFO(HEXAGON.A2_vnavghr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vnavghr :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vnavghr">;
//
// BUILTIN_INFO(HEXAGON.A4_round_ri,SI_ftype_SISI,2)
//
def int_hexagon_A4_round_ri :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_round_ri">;
//
// BUILTIN_INFO(HEXAGON.A4_round_rr,SI_ftype_SISI,2)
//
def int_hexagon_A4_round_rr :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_round_rr">;
//
// BUILTIN_INFO(HEXAGON.A4_round_ri_sat,SI_ftype_SISI,2)
//
def int_hexagon_A4_round_ri_sat :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_round_ri_sat">;
//
// BUILTIN_INFO(HEXAGON.A4_round_rr_sat,SI_ftype_SISI,2)
//
def int_hexagon_A4_round_rr_sat :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_round_rr_sat">;
//
// BUILTIN_INFO(HEXAGON.A4_cround_ri,SI_ftype_SISI,2)
//
def int_hexagon_A4_cround_ri :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cround_ri">;
//
// BUILTIN_INFO(HEXAGON.A4_cround_rr,SI_ftype_SISI,2)
//
def int_hexagon_A4_cround_rr :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cround_rr">;
//
// BUILTIN_INFO(HEXAGON.A4_vrminh,DI_ftype_DIDISI,3)
//
def int_hexagon_A4_vrminh :
Hexagon_di_didisi_Intrinsic<"HEXAGON_A4_vrminh">;
//
// BUILTIN_INFO(HEXAGON.A4_vrmaxh,DI_ftype_DIDISI,3)
//
def int_hexagon_A4_vrmaxh :
Hexagon_di_didisi_Intrinsic<"HEXAGON_A4_vrmaxh">;
//
// BUILTIN_INFO(HEXAGON.A4_vrminuh,DI_ftype_DIDISI,3)
//
def int_hexagon_A4_vrminuh :
Hexagon_di_didisi_Intrinsic<"HEXAGON_A4_vrminuh">;
//
// BUILTIN_INFO(HEXAGON.A4_vrmaxuh,DI_ftype_DIDISI,3)
//
def int_hexagon_A4_vrmaxuh :
Hexagon_di_didisi_Intrinsic<"HEXAGON_A4_vrmaxuh">;
//
// BUILTIN_INFO(HEXAGON.A4_vrminw,DI_ftype_DIDISI,3)
//
def int_hexagon_A4_vrminw :
Hexagon_di_didisi_Intrinsic<"HEXAGON_A4_vrminw">;
//
// BUILTIN_INFO(HEXAGON.A4_vrmaxw,DI_ftype_DIDISI,3)
//
def int_hexagon_A4_vrmaxw :
Hexagon_di_didisi_Intrinsic<"HEXAGON_A4_vrmaxw">;
//
// BUILTIN_INFO(HEXAGON.A4_vrminuw,DI_ftype_DIDISI,3)
//
def int_hexagon_A4_vrminuw :
Hexagon_di_didisi_Intrinsic<"HEXAGON_A4_vrminuw">;
//
// BUILTIN_INFO(HEXAGON.A4_vrmaxuw,DI_ftype_DIDISI,3)
//
def int_hexagon_A4_vrmaxuw :
Hexagon_di_didisi_Intrinsic<"HEXAGON_A4_vrmaxuw">;
//
// BUILTIN_INFO(HEXAGON.A2_vminb,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vminb :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vminb">;
//
// BUILTIN_INFO(HEXAGON.A2_vmaxb,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vmaxb :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vmaxb">;
//
// BUILTIN_INFO(HEXAGON.A2_vminub,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vminub :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vminub">;
//
// BUILTIN_INFO(HEXAGON.A2_vmaxub,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vmaxub :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vmaxub">;
//
// BUILTIN_INFO(HEXAGON.A2_vminh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vminh :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vminh">;
//
// BUILTIN_INFO(HEXAGON.A2_vmaxh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vmaxh :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vmaxh">;
//
// BUILTIN_INFO(HEXAGON.A2_vminuh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vminuh :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vminuh">;
//
// BUILTIN_INFO(HEXAGON.A2_vmaxuh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vmaxuh :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vmaxuh">;
//
// BUILTIN_INFO(HEXAGON.A2_vminw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vminw :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vminw">;
//
// BUILTIN_INFO(HEXAGON.A2_vmaxw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vmaxw :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vmaxw">;
//
// BUILTIN_INFO(HEXAGON.A2_vminuw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vminuw :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vminuw">;
//
// BUILTIN_INFO(HEXAGON.A2_vmaxuw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vmaxuw :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vmaxuw">;
//
// BUILTIN_INFO(HEXAGON.A4_modwrapu,SI_ftype_SISI,2)
//
def int_hexagon_A4_modwrapu :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_modwrapu">;
//
// BUILTIN_INFO(HEXAGON.F2_sfadd,SF_ftype_SFSF,2)
//
def int_hexagon_F2_sfadd :
Hexagon_sf_sfsf_Intrinsic<"HEXAGON_F2_sfadd">;
//
// BUILTIN_INFO(HEXAGON.F2_sfsub,SF_ftype_SFSF,2)
//
def int_hexagon_F2_sfsub :
Hexagon_sf_sfsf_Intrinsic<"HEXAGON_F2_sfsub">;
//
// BUILTIN_INFO(HEXAGON.F2_sfmpy,SF_ftype_SFSF,2)
//
def int_hexagon_F2_sfmpy :
Hexagon_sf_sfsf_Intrinsic<"HEXAGON_F2_sfmpy">;
//
// BUILTIN_INFO(HEXAGON.F2_sffma,SF_ftype_SFSFSF,3)
//
def int_hexagon_F2_sffma :
Hexagon_sf_sfsfsf_Intrinsic<"HEXAGON_F2_sffma">;
//
// BUILTIN_INFO(HEXAGON.F2_sffma_sc,SF_ftype_SFSFSFQI,4)
//
def int_hexagon_F2_sffma_sc :
Hexagon_sf_sfsfsfqi_Intrinsic<"HEXAGON_F2_sffma_sc">;
//
// BUILTIN_INFO(HEXAGON.F2_sffms,SF_ftype_SFSFSF,3)
//
def int_hexagon_F2_sffms :
Hexagon_sf_sfsfsf_Intrinsic<"HEXAGON_F2_sffms">;
//
// BUILTIN_INFO(HEXAGON.F2_sffma_lib,SF_ftype_SFSFSF,3)
//
def int_hexagon_F2_sffma_lib :
Hexagon_sf_sfsfsf_Intrinsic<"HEXAGON_F2_sffma_lib">;
//
// BUILTIN_INFO(HEXAGON.F2_sffms_lib,SF_ftype_SFSFSF,3)
//
def int_hexagon_F2_sffms_lib :
Hexagon_sf_sfsfsf_Intrinsic<"HEXAGON_F2_sffms_lib">;
//
// BUILTIN_INFO(HEXAGON.F2_sfcmpeq,QI_ftype_SFSF,2)
//
def int_hexagon_F2_sfcmpeq :
Hexagon_si_sfsf_Intrinsic<"HEXAGON_F2_sfcmpeq">;
//
// BUILTIN_INFO(HEXAGON.F2_sfcmpgt,QI_ftype_SFSF,2)
//
def int_hexagon_F2_sfcmpgt :
Hexagon_si_sfsf_Intrinsic<"HEXAGON_F2_sfcmpgt">;
//
// BUILTIN_INFO(HEXAGON.F2_sfcmpge,QI_ftype_SFSF,2)
//
def int_hexagon_F2_sfcmpge :
Hexagon_si_sfsf_Intrinsic<"HEXAGON_F2_sfcmpge">;
//
// BUILTIN_INFO(HEXAGON.F2_sfcmpuo,QI_ftype_SFSF,2)
//
def int_hexagon_F2_sfcmpuo :
Hexagon_si_sfsf_Intrinsic<"HEXAGON_F2_sfcmpuo">;
//
// BUILTIN_INFO(HEXAGON.F2_sfmax,SF_ftype_SFSF,2)
//
def int_hexagon_F2_sfmax :
Hexagon_sf_sfsf_Intrinsic<"HEXAGON_F2_sfmax">;
//
// BUILTIN_INFO(HEXAGON.F2_sfmin,SF_ftype_SFSF,2)
//
def int_hexagon_F2_sfmin :
Hexagon_sf_sfsf_Intrinsic<"HEXAGON_F2_sfmin">;
//
// BUILTIN_INFO(HEXAGON.F2_sfclass,QI_ftype_SFSI,2)
//
def int_hexagon_F2_sfclass :
Hexagon_si_sfsi_Intrinsic<"HEXAGON_F2_sfclass">;
//
// BUILTIN_INFO(HEXAGON.F2_sfimm_p,SF_ftype_SI,1)
//
def int_hexagon_F2_sfimm_p :
Hexagon_sf_si_Intrinsic<"HEXAGON_F2_sfimm_p">;
//
// BUILTIN_INFO(HEXAGON.F2_sfimm_n,SF_ftype_SI,1)
//
def int_hexagon_F2_sfimm_n :
Hexagon_sf_si_Intrinsic<"HEXAGON_F2_sfimm_n">;
//
// BUILTIN_INFO(HEXAGON.F2_sffixupn,SF_ftype_SFSF,2)
//
def int_hexagon_F2_sffixupn :
Hexagon_sf_sfsf_Intrinsic<"HEXAGON_F2_sffixupn">;
//
// BUILTIN_INFO(HEXAGON.F2_sffixupd,SF_ftype_SFSF,2)
//
def int_hexagon_F2_sffixupd :
Hexagon_sf_sfsf_Intrinsic<"HEXAGON_F2_sffixupd">;
//
// BUILTIN_INFO(HEXAGON.F2_sffixupr,SF_ftype_SF,1)
//
def int_hexagon_F2_sffixupr :
Hexagon_sf_sf_Intrinsic<"HEXAGON_F2_sffixupr">;
//
// BUILTIN_INFO(HEXAGON.F2_dfcmpeq,QI_ftype_DFDF,2)
//
def int_hexagon_F2_dfcmpeq :
Hexagon_si_dfdf_Intrinsic<"HEXAGON_F2_dfcmpeq">;
//
// BUILTIN_INFO(HEXAGON.F2_dfcmpgt,QI_ftype_DFDF,2)
//
def int_hexagon_F2_dfcmpgt :
Hexagon_si_dfdf_Intrinsic<"HEXAGON_F2_dfcmpgt">;
//
// BUILTIN_INFO(HEXAGON.F2_dfcmpge,QI_ftype_DFDF,2)
//
def int_hexagon_F2_dfcmpge :
Hexagon_si_dfdf_Intrinsic<"HEXAGON_F2_dfcmpge">;
//
// BUILTIN_INFO(HEXAGON.F2_dfcmpuo,QI_ftype_DFDF,2)
//
def int_hexagon_F2_dfcmpuo :
Hexagon_si_dfdf_Intrinsic<"HEXAGON_F2_dfcmpuo">;
//
// BUILTIN_INFO(HEXAGON.F2_dfclass,QI_ftype_DFSI,2)
//
def int_hexagon_F2_dfclass :
Hexagon_si_dfsi_Intrinsic<"HEXAGON_F2_dfclass">;
//
// BUILTIN_INFO(HEXAGON.F2_dfimm_p,DF_ftype_SI,1)
//
def int_hexagon_F2_dfimm_p :
Hexagon_df_si_Intrinsic<"HEXAGON_F2_dfimm_p">;
//
// BUILTIN_INFO(HEXAGON.F2_dfimm_n,DF_ftype_SI,1)
//
def int_hexagon_F2_dfimm_n :
Hexagon_df_si_Intrinsic<"HEXAGON_F2_dfimm_n">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_sf2df,DF_ftype_SF,1)
//
def int_hexagon_F2_conv_sf2df :
Hexagon_df_sf_Intrinsic<"HEXAGON_F2_conv_sf2df">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_df2sf,SF_ftype_DF,1)
//
def int_hexagon_F2_conv_df2sf :
Hexagon_sf_df_Intrinsic<"HEXAGON_F2_conv_df2sf">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_uw2sf,SF_ftype_SI,1)
//
def int_hexagon_F2_conv_uw2sf :
Hexagon_sf_si_Intrinsic<"HEXAGON_F2_conv_uw2sf">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_uw2df,DF_ftype_SI,1)
//
def int_hexagon_F2_conv_uw2df :
Hexagon_df_si_Intrinsic<"HEXAGON_F2_conv_uw2df">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_w2sf,SF_ftype_SI,1)
//
def int_hexagon_F2_conv_w2sf :
Hexagon_sf_si_Intrinsic<"HEXAGON_F2_conv_w2sf">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_w2df,DF_ftype_SI,1)
//
def int_hexagon_F2_conv_w2df :
Hexagon_df_si_Intrinsic<"HEXAGON_F2_conv_w2df">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_ud2sf,SF_ftype_DI,1)
//
def int_hexagon_F2_conv_ud2sf :
Hexagon_sf_di_Intrinsic<"HEXAGON_F2_conv_ud2sf">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_ud2df,DF_ftype_DI,1)
//
def int_hexagon_F2_conv_ud2df :
Hexagon_df_di_Intrinsic<"HEXAGON_F2_conv_ud2df">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_d2sf,SF_ftype_DI,1)
//
def int_hexagon_F2_conv_d2sf :
Hexagon_sf_di_Intrinsic<"HEXAGON_F2_conv_d2sf">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_d2df,DF_ftype_DI,1)
//
def int_hexagon_F2_conv_d2df :
Hexagon_df_di_Intrinsic<"HEXAGON_F2_conv_d2df">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_sf2uw,SI_ftype_SF,1)
//
def int_hexagon_F2_conv_sf2uw :
Hexagon_si_sf_Intrinsic<"HEXAGON_F2_conv_sf2uw">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_sf2w,SI_ftype_SF,1)
//
def int_hexagon_F2_conv_sf2w :
Hexagon_si_sf_Intrinsic<"HEXAGON_F2_conv_sf2w">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_sf2ud,DI_ftype_SF,1)
//
def int_hexagon_F2_conv_sf2ud :
Hexagon_di_sf_Intrinsic<"HEXAGON_F2_conv_sf2ud">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_sf2d,DI_ftype_SF,1)
//
def int_hexagon_F2_conv_sf2d :
Hexagon_di_sf_Intrinsic<"HEXAGON_F2_conv_sf2d">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_df2uw,SI_ftype_DF,1)
//
def int_hexagon_F2_conv_df2uw :
Hexagon_si_df_Intrinsic<"HEXAGON_F2_conv_df2uw">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_df2w,SI_ftype_DF,1)
//
def int_hexagon_F2_conv_df2w :
Hexagon_si_df_Intrinsic<"HEXAGON_F2_conv_df2w">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_df2ud,DI_ftype_DF,1)
//
def int_hexagon_F2_conv_df2ud :
Hexagon_di_df_Intrinsic<"HEXAGON_F2_conv_df2ud">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_df2d,DI_ftype_DF,1)
//
def int_hexagon_F2_conv_df2d :
Hexagon_di_df_Intrinsic<"HEXAGON_F2_conv_df2d">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_sf2uw_chop,SI_ftype_SF,1)
//
def int_hexagon_F2_conv_sf2uw_chop :
Hexagon_si_sf_Intrinsic<"HEXAGON_F2_conv_sf2uw_chop">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_sf2w_chop,SI_ftype_SF,1)
//
def int_hexagon_F2_conv_sf2w_chop :
Hexagon_si_sf_Intrinsic<"HEXAGON_F2_conv_sf2w_chop">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_sf2ud_chop,DI_ftype_SF,1)
//
def int_hexagon_F2_conv_sf2ud_chop :
Hexagon_di_sf_Intrinsic<"HEXAGON_F2_conv_sf2ud_chop">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_sf2d_chop,DI_ftype_SF,1)
//
def int_hexagon_F2_conv_sf2d_chop :
Hexagon_di_sf_Intrinsic<"HEXAGON_F2_conv_sf2d_chop">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_df2uw_chop,SI_ftype_DF,1)
//
def int_hexagon_F2_conv_df2uw_chop :
Hexagon_si_df_Intrinsic<"HEXAGON_F2_conv_df2uw_chop">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_df2w_chop,SI_ftype_DF,1)
//
def int_hexagon_F2_conv_df2w_chop :
Hexagon_si_df_Intrinsic<"HEXAGON_F2_conv_df2w_chop">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_df2ud_chop,DI_ftype_DF,1)
//
def int_hexagon_F2_conv_df2ud_chop :
Hexagon_di_df_Intrinsic<"HEXAGON_F2_conv_df2ud_chop">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_df2d_chop,DI_ftype_DF,1)
//
def int_hexagon_F2_conv_df2d_chop :
Hexagon_di_df_Intrinsic<"HEXAGON_F2_conv_df2d_chop">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_asr_r_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_asr_r_r">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_asl_r_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_asl_r_r">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_lsr_r_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_lsr_r_r">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_lsl_r_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_lsl_r_r">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_p,DI_ftype_DISI,2)
//
def int_hexagon_S2_asr_r_p :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asr_r_p">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_p,DI_ftype_DISI,2)
//
def int_hexagon_S2_asl_r_p :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asl_r_p">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_p,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsr_r_p :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_lsr_r_p">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_p,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsl_r_p :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_lsl_r_p">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_r_acc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asr_r_r_acc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asr_r_r_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_r_acc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_r_r_acc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asl_r_r_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_r_acc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_r_r_acc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsr_r_r_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_r_acc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsl_r_r_acc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsl_r_r_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_p_acc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_r_p_acc :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asr_r_p_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_p_acc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_r_p_acc :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asl_r_p_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_p_acc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_r_p_acc :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsr_r_p_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_p_acc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsl_r_p_acc :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsl_r_p_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_r_nac,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asr_r_r_nac :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asr_r_r_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_r_nac,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_r_r_nac :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asl_r_r_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_r_nac,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_r_r_nac :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsr_r_r_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_r_nac,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsl_r_r_nac :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsl_r_r_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_p_nac,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_r_p_nac :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asr_r_p_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_p_nac,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_r_p_nac :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asl_r_p_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_p_nac,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_r_p_nac :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsr_r_p_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_p_nac,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsl_r_p_nac :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsl_r_p_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_r_and,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asr_r_r_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asr_r_r_and">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_r_and,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_r_r_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asl_r_r_and">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_r_and,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_r_r_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsr_r_r_and">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_r_and,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsl_r_r_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsl_r_r_and">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_r_or,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asr_r_r_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asr_r_r_or">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_r_or,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_r_r_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asl_r_r_or">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_r_or,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_r_r_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsr_r_r_or">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_r_or,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsl_r_r_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsl_r_r_or">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_p_and,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_r_p_and :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asr_r_p_and">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_p_and,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_r_p_and :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asl_r_p_and">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_p_and,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_r_p_and :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsr_r_p_and">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_p_and,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsl_r_p_and :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsl_r_p_and">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_p_or,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_r_p_or :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asr_r_p_or">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_p_or,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_r_p_or :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asl_r_p_or">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_p_or,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_r_p_or :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsr_r_p_or">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_p_or,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsl_r_p_or :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsl_r_p_or">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_p_xor,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_r_p_xor :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asr_r_p_xor">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_p_xor,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_r_p_xor :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asl_r_p_xor">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_p_xor,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_r_p_xor :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsr_r_p_xor">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_p_xor,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsl_r_p_xor :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsl_r_p_xor">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_r_sat,SI_ftype_SISI,2)
//
def int_hexagon_S2_asr_r_r_sat :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_asr_r_r_sat">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_r_sat,SI_ftype_SISI,2)
//
def int_hexagon_S2_asl_r_r_sat :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_asl_r_r_sat">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_asr_i_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_asr_i_r">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_lsr_i_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_lsr_i_r">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_asl_i_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_asl_i_r">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_p,DI_ftype_DISI,2)
//
def int_hexagon_S2_asr_i_p :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asr_i_p">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_p,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsr_i_p :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_lsr_i_p">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_p,DI_ftype_DISI,2)
//
def int_hexagon_S2_asl_i_p :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asl_i_p">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_r_acc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asr_i_r_acc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asr_i_r_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_r_acc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_i_r_acc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsr_i_r_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_r_acc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_i_r_acc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asl_i_r_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_p_acc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_i_p_acc :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asr_i_p_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_p_acc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_i_p_acc :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsr_i_p_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_p_acc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_i_p_acc :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asl_i_p_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_r_nac,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asr_i_r_nac :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asr_i_r_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_r_nac,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_i_r_nac :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsr_i_r_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_r_nac,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_i_r_nac :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asl_i_r_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_p_nac,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_i_p_nac :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asr_i_p_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_p_nac,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_i_p_nac :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsr_i_p_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_p_nac,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_i_p_nac :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asl_i_p_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_r_xacc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_i_r_xacc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsr_i_r_xacc">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_r_xacc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_i_r_xacc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asl_i_r_xacc">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_p_xacc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_i_p_xacc :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsr_i_p_xacc">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_p_xacc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_i_p_xacc :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asl_i_p_xacc">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_r_and,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asr_i_r_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asr_i_r_and">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_r_and,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_i_r_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsr_i_r_and">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_r_and,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_i_r_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asl_i_r_and">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_r_or,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asr_i_r_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asr_i_r_or">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_r_or,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_i_r_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsr_i_r_or">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_r_or,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_i_r_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asl_i_r_or">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_p_and,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_i_p_and :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asr_i_p_and">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_p_and,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_i_p_and :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsr_i_p_and">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_p_and,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_i_p_and :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asl_i_p_and">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_p_or,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_i_p_or :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asr_i_p_or">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_p_or,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_i_p_or :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsr_i_p_or">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_p_or,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_i_p_or :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asl_i_p_or">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_r_sat,SI_ftype_SISI,2)
//
def int_hexagon_S2_asl_i_r_sat :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_asl_i_r_sat">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_r_rnd,SI_ftype_SISI,2)
//
def int_hexagon_S2_asr_i_r_rnd :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_asr_i_r_rnd">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_r_rnd_goodsyntax,SI_ftype_SISI,2)
//
def int_hexagon_S2_asr_i_r_rnd_goodsyntax :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_asr_i_r_rnd_goodsyntax">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_p_rnd,DI_ftype_DISI,2)
//
def int_hexagon_S2_asr_i_p_rnd :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asr_i_p_rnd">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_p_rnd_goodsyntax,DI_ftype_DISI,2)
//
def int_hexagon_S2_asr_i_p_rnd_goodsyntax :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asr_i_p_rnd_goodsyntax">;
//
// BUILTIN_INFO(HEXAGON.S4_lsli,SI_ftype_SISI,2)
//
def int_hexagon_S4_lsli :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S4_lsli">;
//
// BUILTIN_INFO(HEXAGON.S2_addasl_rrri,SI_ftype_SISISI,3)
//
def int_hexagon_S2_addasl_rrri :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_addasl_rrri">;
//
// BUILTIN_INFO(HEXAGON.S4_andi_asl_ri,SI_ftype_SISISI,3)
//
def int_hexagon_S4_andi_asl_ri :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_andi_asl_ri">;
//
// BUILTIN_INFO(HEXAGON.S4_ori_asl_ri,SI_ftype_SISISI,3)
//
def int_hexagon_S4_ori_asl_ri :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_ori_asl_ri">;
//
// BUILTIN_INFO(HEXAGON.S4_addi_asl_ri,SI_ftype_SISISI,3)
//
def int_hexagon_S4_addi_asl_ri :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_addi_asl_ri">;
//
// BUILTIN_INFO(HEXAGON.S4_subi_asl_ri,SI_ftype_SISISI,3)
//
def int_hexagon_S4_subi_asl_ri :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_subi_asl_ri">;
//
// BUILTIN_INFO(HEXAGON.S4_andi_lsr_ri,SI_ftype_SISISI,3)
//
def int_hexagon_S4_andi_lsr_ri :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_andi_lsr_ri">;
//
// BUILTIN_INFO(HEXAGON.S4_ori_lsr_ri,SI_ftype_SISISI,3)
//
def int_hexagon_S4_ori_lsr_ri :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_ori_lsr_ri">;
//
// BUILTIN_INFO(HEXAGON.S4_addi_lsr_ri,SI_ftype_SISISI,3)
//
def int_hexagon_S4_addi_lsr_ri :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_addi_lsr_ri">;
//
// BUILTIN_INFO(HEXAGON.S4_subi_lsr_ri,SI_ftype_SISISI,3)
//
def int_hexagon_S4_subi_lsr_ri :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_subi_lsr_ri">;
//
// BUILTIN_INFO(HEXAGON.S2_valignib,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_valignib :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_valignib">;
//
// BUILTIN_INFO(HEXAGON.S2_valignrb,DI_ftype_DIDIQI,3)
//
def int_hexagon_S2_valignrb :
Hexagon_di_didiqi_Intrinsic<"HEXAGON_S2_valignrb">;
//
// BUILTIN_INFO(HEXAGON.S2_vspliceib,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_vspliceib :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_vspliceib">;
//
// BUILTIN_INFO(HEXAGON.S2_vsplicerb,DI_ftype_DIDIQI,3)
//
def int_hexagon_S2_vsplicerb :
Hexagon_di_didiqi_Intrinsic<"HEXAGON_S2_vsplicerb">;
//
// BUILTIN_INFO(HEXAGON.S2_vsplatrh,DI_ftype_SI,1)
//
def int_hexagon_S2_vsplatrh :
Hexagon_di_si_Intrinsic<"HEXAGON_S2_vsplatrh">;
//
// BUILTIN_INFO(HEXAGON.S2_vsplatrb,SI_ftype_SI,1)
//
def int_hexagon_S2_vsplatrb :
Hexagon_si_si_Intrinsic<"HEXAGON_S2_vsplatrb">;
//
// BUILTIN_INFO(HEXAGON.S2_insert,SI_ftype_SISISISI,4)
//
def int_hexagon_S2_insert :
Hexagon_si_sisisisi_Intrinsic<"HEXAGON_S2_insert">;
//
// BUILTIN_INFO(HEXAGON.S2_tableidxb_goodsyntax,SI_ftype_SISISISI,4)
//
def int_hexagon_S2_tableidxb_goodsyntax :
Hexagon_si_sisisisi_Intrinsic<"HEXAGON_S2_tableidxb_goodsyntax">;
//
// BUILTIN_INFO(HEXAGON.S2_tableidxh_goodsyntax,SI_ftype_SISISISI,4)
//
def int_hexagon_S2_tableidxh_goodsyntax :
Hexagon_si_sisisisi_Intrinsic<"HEXAGON_S2_tableidxh_goodsyntax">;
//
// BUILTIN_INFO(HEXAGON.S2_tableidxw_goodsyntax,SI_ftype_SISISISI,4)
//
def int_hexagon_S2_tableidxw_goodsyntax :
Hexagon_si_sisisisi_Intrinsic<"HEXAGON_S2_tableidxw_goodsyntax">;
//
// BUILTIN_INFO(HEXAGON.S2_tableidxd_goodsyntax,SI_ftype_SISISISI,4)
//
def int_hexagon_S2_tableidxd_goodsyntax :
Hexagon_si_sisisisi_Intrinsic<"HEXAGON_S2_tableidxd_goodsyntax">;
//
// BUILTIN_INFO(HEXAGON.A4_bitspliti,DI_ftype_SISI,2)
//
def int_hexagon_A4_bitspliti :
Hexagon_di_sisi_Intrinsic<"HEXAGON_A4_bitspliti">;
//
// BUILTIN_INFO(HEXAGON.A4_bitsplit,DI_ftype_SISI,2)
//
def int_hexagon_A4_bitsplit :
Hexagon_di_sisi_Intrinsic<"HEXAGON_A4_bitsplit">;
//
// BUILTIN_INFO(HEXAGON.S4_extract,SI_ftype_SISISI,3)
//
def int_hexagon_S4_extract :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_extract">;
//
// BUILTIN_INFO(HEXAGON.S2_extractu,SI_ftype_SISISI,3)
//
def int_hexagon_S2_extractu :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_extractu">;
//
// BUILTIN_INFO(HEXAGON.S2_insertp,DI_ftype_DIDISISI,4)
//
def int_hexagon_S2_insertp :
Hexagon_di_didisisi_Intrinsic<"HEXAGON_S2_insertp">;
//
// BUILTIN_INFO(HEXAGON.S4_extractp,DI_ftype_DISISI,3)
//
def int_hexagon_S4_extractp :
Hexagon_di_disisi_Intrinsic<"HEXAGON_S4_extractp">;
//
// BUILTIN_INFO(HEXAGON.S2_extractup,DI_ftype_DISISI,3)
//
def int_hexagon_S2_extractup :
Hexagon_di_disisi_Intrinsic<"HEXAGON_S2_extractup">;
//
// BUILTIN_INFO(HEXAGON.S2_insert_rp,SI_ftype_SISIDI,3)
//
def int_hexagon_S2_insert_rp :
Hexagon_si_sisidi_Intrinsic<"HEXAGON_S2_insert_rp">;
//
// BUILTIN_INFO(HEXAGON.S4_extract_rp,SI_ftype_SIDI,2)
//
def int_hexagon_S4_extract_rp :
Hexagon_si_sidi_Intrinsic<"HEXAGON_S4_extract_rp">;
//
// BUILTIN_INFO(HEXAGON.S2_extractu_rp,SI_ftype_SIDI,2)
//
def int_hexagon_S2_extractu_rp :
Hexagon_si_sidi_Intrinsic<"HEXAGON_S2_extractu_rp">;
//
// BUILTIN_INFO(HEXAGON.S2_insertp_rp,DI_ftype_DIDIDI,3)
//
def int_hexagon_S2_insertp_rp :
Hexagon_di_dididi_Intrinsic<"HEXAGON_S2_insertp_rp">;
//
// BUILTIN_INFO(HEXAGON.S4_extractp_rp,DI_ftype_DIDI,2)
//
def int_hexagon_S4_extractp_rp :
Hexagon_di_didi_Intrinsic<"HEXAGON_S4_extractp_rp">;
//
// BUILTIN_INFO(HEXAGON.S2_extractup_rp,DI_ftype_DIDI,2)
//
def int_hexagon_S2_extractup_rp :
Hexagon_di_didi_Intrinsic<"HEXAGON_S2_extractup_rp">;
//
// BUILTIN_INFO(HEXAGON.S2_tstbit_i,QI_ftype_SISI,2)
//
def int_hexagon_S2_tstbit_i :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_tstbit_i">;
//
// BUILTIN_INFO(HEXAGON.S4_ntstbit_i,QI_ftype_SISI,2)
//
def int_hexagon_S4_ntstbit_i :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S4_ntstbit_i">;
//
// BUILTIN_INFO(HEXAGON.S2_setbit_i,SI_ftype_SISI,2)
//
def int_hexagon_S2_setbit_i :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_setbit_i">;
//
// BUILTIN_INFO(HEXAGON.S2_togglebit_i,SI_ftype_SISI,2)
//
def int_hexagon_S2_togglebit_i :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_togglebit_i">;
//
// BUILTIN_INFO(HEXAGON.S2_clrbit_i,SI_ftype_SISI,2)
//
def int_hexagon_S2_clrbit_i :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_clrbit_i">;
//
// BUILTIN_INFO(HEXAGON.S2_tstbit_r,QI_ftype_SISI,2)
//
def int_hexagon_S2_tstbit_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_tstbit_r">;
//
// BUILTIN_INFO(HEXAGON.S4_ntstbit_r,QI_ftype_SISI,2)
//
def int_hexagon_S4_ntstbit_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S4_ntstbit_r">;
//
// BUILTIN_INFO(HEXAGON.S2_setbit_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_setbit_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_setbit_r">;
//
// BUILTIN_INFO(HEXAGON.S2_togglebit_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_togglebit_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_togglebit_r">;
//
// BUILTIN_INFO(HEXAGON.S2_clrbit_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_clrbit_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_clrbit_r">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_vh,DI_ftype_DISI,2)
//
def int_hexagon_S2_asr_i_vh :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asr_i_vh">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_vh,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsr_i_vh :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_lsr_i_vh">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_vh,DI_ftype_DISI,2)
//
def int_hexagon_S2_asl_i_vh :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asl_i_vh">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_vh,DI_ftype_DISI,2)
//
def int_hexagon_S2_asr_r_vh :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asr_r_vh">;
//
// BUILTIN_INFO(HEXAGON.S5_asrhub_rnd_sat_goodsyntax,SI_ftype_DISI,2)
//
def int_hexagon_S5_asrhub_rnd_sat_goodsyntax :
Hexagon_si_disi_Intrinsic<"HEXAGON_S5_asrhub_rnd_sat_goodsyntax">;
//
// BUILTIN_INFO(HEXAGON.S5_asrhub_sat,SI_ftype_DISI,2)
//
def int_hexagon_S5_asrhub_sat :
Hexagon_si_disi_Intrinsic<"HEXAGON_S5_asrhub_sat">;
//
// BUILTIN_INFO(HEXAGON.S5_vasrhrnd_goodsyntax,DI_ftype_DISI,2)
//
def int_hexagon_S5_vasrhrnd_goodsyntax :
Hexagon_di_disi_Intrinsic<"HEXAGON_S5_vasrhrnd_goodsyntax">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_vh,DI_ftype_DISI,2)
//
def int_hexagon_S2_asl_r_vh :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asl_r_vh">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_vh,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsr_r_vh :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_lsr_r_vh">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_vh,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsl_r_vh :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_lsl_r_vh">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_vw,DI_ftype_DISI,2)
//
def int_hexagon_S2_asr_i_vw :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asr_i_vw">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_svw_trun,SI_ftype_DISI,2)
//
def int_hexagon_S2_asr_i_svw_trun :
Hexagon_si_disi_Intrinsic<"HEXAGON_S2_asr_i_svw_trun">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_svw_trun,SI_ftype_DISI,2)
//
def int_hexagon_S2_asr_r_svw_trun :
Hexagon_si_disi_Intrinsic<"HEXAGON_S2_asr_r_svw_trun">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_vw,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsr_i_vw :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_lsr_i_vw">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_vw,DI_ftype_DISI,2)
//
def int_hexagon_S2_asl_i_vw :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asl_i_vw">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_vw,DI_ftype_DISI,2)
//
def int_hexagon_S2_asr_r_vw :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asr_r_vw">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_vw,DI_ftype_DISI,2)
//
def int_hexagon_S2_asl_r_vw :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asl_r_vw">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_vw,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsr_r_vw :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_lsr_r_vw">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_vw,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsl_r_vw :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_lsl_r_vw">;
//
// BUILTIN_INFO(HEXAGON.S2_vrndpackwh,SI_ftype_DI,1)
//
def int_hexagon_S2_vrndpackwh :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_vrndpackwh">;
//
// BUILTIN_INFO(HEXAGON.S2_vrndpackwhs,SI_ftype_DI,1)
//
def int_hexagon_S2_vrndpackwhs :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_vrndpackwhs">;
//
// BUILTIN_INFO(HEXAGON.S2_vsxtbh,DI_ftype_SI,1)
//
def int_hexagon_S2_vsxtbh :
Hexagon_di_si_Intrinsic<"HEXAGON_S2_vsxtbh">;
//
// BUILTIN_INFO(HEXAGON.S2_vzxtbh,DI_ftype_SI,1)
//
def int_hexagon_S2_vzxtbh :
Hexagon_di_si_Intrinsic<"HEXAGON_S2_vzxtbh">;
//
// BUILTIN_INFO(HEXAGON.S2_vsathub,SI_ftype_DI,1)
//
def int_hexagon_S2_vsathub :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_vsathub">;
//
// BUILTIN_INFO(HEXAGON.S2_svsathub,SI_ftype_SI,1)
//
def int_hexagon_S2_svsathub :
Hexagon_si_si_Intrinsic<"HEXAGON_S2_svsathub">;
//
// BUILTIN_INFO(HEXAGON.S2_svsathb,SI_ftype_SI,1)
//
def int_hexagon_S2_svsathb :
Hexagon_si_si_Intrinsic<"HEXAGON_S2_svsathb">;
//
// BUILTIN_INFO(HEXAGON.S2_vsathb,SI_ftype_DI,1)
//
def int_hexagon_S2_vsathb :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_vsathb">;
//
// BUILTIN_INFO(HEXAGON.S2_vtrunohb,SI_ftype_DI,1)
//
def int_hexagon_S2_vtrunohb :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_vtrunohb">;
//
// BUILTIN_INFO(HEXAGON.S2_vtrunewh,DI_ftype_DIDI,2)
//
def int_hexagon_S2_vtrunewh :
Hexagon_di_didi_Intrinsic<"HEXAGON_S2_vtrunewh">;
//
// BUILTIN_INFO(HEXAGON.S2_vtrunowh,DI_ftype_DIDI,2)
//
def int_hexagon_S2_vtrunowh :
Hexagon_di_didi_Intrinsic<"HEXAGON_S2_vtrunowh">;
//
// BUILTIN_INFO(HEXAGON.S2_vtrunehb,SI_ftype_DI,1)
//
def int_hexagon_S2_vtrunehb :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_vtrunehb">;
//
// BUILTIN_INFO(HEXAGON.S2_vsxthw,DI_ftype_SI,1)
//
def int_hexagon_S2_vsxthw :
Hexagon_di_si_Intrinsic<"HEXAGON_S2_vsxthw">;
//
// BUILTIN_INFO(HEXAGON.S2_vzxthw,DI_ftype_SI,1)
//
def int_hexagon_S2_vzxthw :
Hexagon_di_si_Intrinsic<"HEXAGON_S2_vzxthw">;
//
// BUILTIN_INFO(HEXAGON.S2_vsatwh,SI_ftype_DI,1)
//
def int_hexagon_S2_vsatwh :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_vsatwh">;
//
// BUILTIN_INFO(HEXAGON.S2_vsatwuh,SI_ftype_DI,1)
//
def int_hexagon_S2_vsatwuh :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_vsatwuh">;
//
// BUILTIN_INFO(HEXAGON.S2_packhl,DI_ftype_SISI,2)
//
def int_hexagon_S2_packhl :
Hexagon_di_sisi_Intrinsic<"HEXAGON_S2_packhl">;
//
// BUILTIN_INFO(HEXAGON.A2_swiz,SI_ftype_SI,1)
//
def int_hexagon_A2_swiz :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_swiz">;
//
// BUILTIN_INFO(HEXAGON.S2_vsathub_nopack,DI_ftype_DI,1)
//
def int_hexagon_S2_vsathub_nopack :
Hexagon_di_di_Intrinsic<"HEXAGON_S2_vsathub_nopack">;
//
// BUILTIN_INFO(HEXAGON.S2_vsathb_nopack,DI_ftype_DI,1)
//
def int_hexagon_S2_vsathb_nopack :
Hexagon_di_di_Intrinsic<"HEXAGON_S2_vsathb_nopack">;
//
// BUILTIN_INFO(HEXAGON.S2_vsatwh_nopack,DI_ftype_DI,1)
//
def int_hexagon_S2_vsatwh_nopack :
Hexagon_di_di_Intrinsic<"HEXAGON_S2_vsatwh_nopack">;
//
// BUILTIN_INFO(HEXAGON.S2_vsatwuh_nopack,DI_ftype_DI,1)
//
def int_hexagon_S2_vsatwuh_nopack :
Hexagon_di_di_Intrinsic<"HEXAGON_S2_vsatwuh_nopack">;
//
// BUILTIN_INFO(HEXAGON.S2_shuffob,DI_ftype_DIDI,2)
//
def int_hexagon_S2_shuffob :
Hexagon_di_didi_Intrinsic<"HEXAGON_S2_shuffob">;
//
// BUILTIN_INFO(HEXAGON.S2_shuffeb,DI_ftype_DIDI,2)
//
def int_hexagon_S2_shuffeb :
Hexagon_di_didi_Intrinsic<"HEXAGON_S2_shuffeb">;
//
// BUILTIN_INFO(HEXAGON.S2_shuffoh,DI_ftype_DIDI,2)
//
def int_hexagon_S2_shuffoh :
Hexagon_di_didi_Intrinsic<"HEXAGON_S2_shuffoh">;
//
// BUILTIN_INFO(HEXAGON.S2_shuffeh,DI_ftype_DIDI,2)
//
def int_hexagon_S2_shuffeh :
Hexagon_di_didi_Intrinsic<"HEXAGON_S2_shuffeh">;
//
// BUILTIN_INFO(HEXAGON.S5_popcountp,SI_ftype_DI,1)
//
def int_hexagon_S5_popcountp :
Hexagon_si_di_Intrinsic<"HEXAGON_S5_popcountp">;
//
// BUILTIN_INFO(HEXAGON.S4_parity,SI_ftype_SISI,2)
//
def int_hexagon_S4_parity :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S4_parity">;
//
// BUILTIN_INFO(HEXAGON.S2_parityp,SI_ftype_DIDI,2)
//
def int_hexagon_S2_parityp :
Hexagon_si_didi_Intrinsic<"HEXAGON_S2_parityp">;
//
// BUILTIN_INFO(HEXAGON.S2_lfsp,DI_ftype_DIDI,2)
//
def int_hexagon_S2_lfsp :
Hexagon_di_didi_Intrinsic<"HEXAGON_S2_lfsp">;
//
// BUILTIN_INFO(HEXAGON.S2_clbnorm,SI_ftype_SI,1)
//
def int_hexagon_S2_clbnorm :
Hexagon_si_si_Intrinsic<"HEXAGON_S2_clbnorm">;
//
// BUILTIN_INFO(HEXAGON.S4_clbaddi,SI_ftype_SISI,2)
//
def int_hexagon_S4_clbaddi :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S4_clbaddi">;
//
// BUILTIN_INFO(HEXAGON.S4_clbpnorm,SI_ftype_DI,1)
//
def int_hexagon_S4_clbpnorm :
Hexagon_si_di_Intrinsic<"HEXAGON_S4_clbpnorm">;
//
// BUILTIN_INFO(HEXAGON.S4_clbpaddi,SI_ftype_DISI,2)
//
def int_hexagon_S4_clbpaddi :
Hexagon_si_disi_Intrinsic<"HEXAGON_S4_clbpaddi">;
//
// BUILTIN_INFO(HEXAGON.S2_clb,SI_ftype_SI,1)
//
def int_hexagon_S2_clb :
Hexagon_si_si_Intrinsic<"HEXAGON_S2_clb">;
//
// BUILTIN_INFO(HEXAGON.S2_cl0,SI_ftype_SI,1)
//
def int_hexagon_S2_cl0 :
Hexagon_si_si_Intrinsic<"HEXAGON_S2_cl0">;
//
// BUILTIN_INFO(HEXAGON.S2_cl1,SI_ftype_SI,1)
//
def int_hexagon_S2_cl1 :
Hexagon_si_si_Intrinsic<"HEXAGON_S2_cl1">;
//
// BUILTIN_INFO(HEXAGON.S2_clbp,SI_ftype_DI,1)
//
def int_hexagon_S2_clbp :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_clbp">;
//
// BUILTIN_INFO(HEXAGON.S2_cl0p,SI_ftype_DI,1)
//
def int_hexagon_S2_cl0p :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_cl0p">;
//
// BUILTIN_INFO(HEXAGON.S2_cl1p,SI_ftype_DI,1)
//
def int_hexagon_S2_cl1p :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_cl1p">;
//
// BUILTIN_INFO(HEXAGON.S2_brev,SI_ftype_SI,1)
//
def int_hexagon_S2_brev :
Hexagon_si_si_Intrinsic<"HEXAGON_S2_brev">;
//
// BUILTIN_INFO(HEXAGON.S2_brevp,DI_ftype_DI,1)
//
def int_hexagon_S2_brevp :
Hexagon_di_di_Intrinsic<"HEXAGON_S2_brevp">;
//
// BUILTIN_INFO(HEXAGON.S2_ct0,SI_ftype_SI,1)
//
def int_hexagon_S2_ct0 :
Hexagon_si_si_Intrinsic<"HEXAGON_S2_ct0">;
//
// BUILTIN_INFO(HEXAGON.S2_ct1,SI_ftype_SI,1)
//
def int_hexagon_S2_ct1 :
Hexagon_si_si_Intrinsic<"HEXAGON_S2_ct1">;
//
// BUILTIN_INFO(HEXAGON.S2_ct0p,SI_ftype_DI,1)
//
def int_hexagon_S2_ct0p :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_ct0p">;
//
// BUILTIN_INFO(HEXAGON.S2_ct1p,SI_ftype_DI,1)
//
def int_hexagon_S2_ct1p :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_ct1p">;
//
// BUILTIN_INFO(HEXAGON.S2_interleave,DI_ftype_DI,1)
//
def int_hexagon_S2_interleave :
Hexagon_di_di_Intrinsic<"HEXAGON_S2_interleave">;
//
// BUILTIN_INFO(HEXAGON.S2_deinterleave,DI_ftype_DI,1)
//
def int_hexagon_S2_deinterleave :
Hexagon_di_di_Intrinsic<"HEXAGON_S2_deinterleave">;

//
// BUILTIN_INFO(HEXAGON.dcfetch_A,v_ftype_DI*,1)
//
def int_hexagon_prefetch :
Hexagon_Intrinsic<"HEXAGON_prefetch", [], [llvm_ptr_ty], []>;
def int_hexagon_Y2_dccleana :
Hexagon_Intrinsic<"HEXAGON_Y2_dccleana", [], [llvm_ptr_ty], []>;
def int_hexagon_Y2_dccleaninva :
Hexagon_Intrinsic<"HEXAGON_Y2_dccleaninva", [], [llvm_ptr_ty], []>;
def int_hexagon_Y2_dcinva :
Hexagon_Intrinsic<"HEXAGON_Y2_dcinva", [], [llvm_ptr_ty], []>;
def int_hexagon_Y2_dczeroa :
Hexagon_Intrinsic<"HEXAGON_Y2_dczeroa", [], [llvm_ptr_ty],
      [IntrWriteMem, IntrArgMemOnly, IntrHasSideEffects]>;
def int_hexagon_Y4_l2fetch :
Hexagon_Intrinsic<"HEXAGON_Y4_l2fetch", [], [llvm_ptr_ty, llvm_i32_ty], []>;
def int_hexagon_Y5_l2fetch :
Hexagon_Intrinsic<"HEXAGON_Y5_l2fetch", [], [llvm_ptr_ty, llvm_i64_ty], []>;

def llvm_ptr32_ty : LLVMPointerType<llvm_i32_ty>;
def llvm_ptr64_ty : LLVMPointerType<llvm_i64_ty>;

// Mark locked loads as read/write to prevent any accidental reordering.
def int_hexagon_L2_loadw_locked :
Hexagon_Intrinsic<"HEXAGON_L2_loadw_locked", [llvm_i32_ty], [llvm_ptr32_ty],
      [IntrArgMemOnly, NoCapture<0>]>;
def int_hexagon_L4_loadd_locked :
Hexagon_Intrinsic<"HEXAGON_L4_loadd_locked", [llvm_i64_ty], [llvm_ptr64_ty],
      [IntrArgMemOnly, NoCapture<0>]>;

def int_hexagon_S2_storew_locked :
Hexagon_Intrinsic<"HEXAGON_S2_storew_locked", [llvm_i32_ty],
      [llvm_ptr32_ty, llvm_i32_ty], [IntrArgMemOnly, NoCapture<0>]>;
def int_hexagon_S4_stored_locked :
Hexagon_Intrinsic<"HEXAGON_S4_stored_locked", [llvm_i32_ty],
      [llvm_ptr64_ty, llvm_i64_ty], [IntrArgMemOnly, NoCapture<0>]>;

// V60

class Hexagon_v2048v2048_Intrinsic_T<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v64i32_ty],
                          [IntrNoMem]>;

// tag : V6_hi_W
// tag : V6_lo_W
class Hexagon_v512v1024_Intrinsic_T<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v32i32_ty],
                          [IntrNoMem]>;

// tag : V6_hi_W_128B
// tag : V6_lo_W_128B
class Hexagon_v1024v2048_Intrinsic_T<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v64i32_ty],
                          [IntrNoMem]>;

class Hexagon_v1024v1024_Intrinsic_T<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty],
                          [IntrNoMem]>;

// BUILTIN_INFO(HEXAGON.V6_hi_W,VI_ftype_VI,1)
// tag : V6_hi
def int_hexagon_V6_hi :
Hexagon_v512v1024_Intrinsic_T<"HEXAGON_V6_hi">;

// BUILTIN_INFO(HEXAGON.V6_lo_W,VI_ftype_VI,1)
// tag : V6_lo
def int_hexagon_V6_lo :
Hexagon_v512v1024_Intrinsic_T<"HEXAGON_V6_lo">;

// BUILTIN_INFO(HEXAGON.V6_hi_W,VI_ftype_VI,1)
// tag : V6_hi_128B
def int_hexagon_V6_hi_128B :
Hexagon_v1024v2048_Intrinsic_T<"HEXAGON_V6_hi_128B">;

// BUILTIN_INFO(HEXAGON.V6_lo_W,VI_ftype_VI,1)
// tag : V6_lo_128B
def int_hexagon_V6_lo_128B :
Hexagon_v1024v2048_Intrinsic_T<"HEXAGON_V6_lo_128B">;

// BUILTIN_INFO(HEXAGON.V6_vassignp,VI_ftype_VI,1)
// tag : V6_vassignp
def int_hexagon_V6_vassignp :
Hexagon_v1024v1024_Intrinsic_T<"HEXAGON_V6_vassignp">;

// BUILTIN_INFO(HEXAGON.V6_vassignp,VI_ftype_VI,1)
// tag : V6_vassignp_128B
def int_hexagon_V6_vassignp_128B :
Hexagon_v2048v2048_Intrinsic_T<"HEXAGON_V6_vassignp_128B">;


//
// Hexagon_iii_Intrinsic<string GCCIntSuffix>
// tag : S6_rol_i_r
class Hexagon_iii_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_LLiLLii_Intrinsic<string GCCIntSuffix>
// tag : S6_rol_i_p
class Hexagon_LLiLLii_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_iiii_Intrinsic<string GCCIntSuffix>
// tag : S6_rol_i_r_acc
class Hexagon_iiii_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty,llvm_i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_LLiLLiLLii_Intrinsic<string GCCIntSuffix>
// tag : S6_rol_i_p_acc
class Hexagon_LLiLLiLLii_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty,llvm_i64_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v512v512v512i_Intrinsic<string GCCIntSuffix>
// tag : V6_valignb
class Hexagon_v512v512v512i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v16i32_ty,llvm_v16i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v1024v1024v1024i_Intrinsic<string GCCIntSuffix>
// tag : V6_valignb_128B
class Hexagon_v1024v1024v1024i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_v32i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v512v512i_Intrinsic<string GCCIntSuffix>
// tag : V6_vror
class Hexagon_v512v512i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v16i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v1024v1024i_Intrinsic<string GCCIntSuffix>
// tag : V6_vror_128B
class Hexagon_v1024v1024i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v1024v512_Intrinsic<string GCCIntSuffix>
// tag : V6_vunpackub
class Hexagon_v1024v512_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v16i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v2048v1024_Intrinsic<string GCCIntSuffix>
// tag : V6_vunpackub_128B
class Hexagon_v2048v1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v32i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v1024v1024v512_Intrinsic<string GCCIntSuffix>
// tag : V6_vunpackob
class Hexagon_v1024v1024v512_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_v16i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v2048v2048v1024_Intrinsic<string GCCIntSuffix>
// tag : V6_vunpackob_128B
class Hexagon_v2048v2048v1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v64i32_ty,llvm_v32i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v512v512v512_Intrinsic<string GCCIntSuffix>
// tag : V6_vpackeb
class Hexagon_v512v512v512_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v16i32_ty,llvm_v16i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v1024v1024v1024_Intrinsic<string GCCIntSuffix>
// tag : V6_vpackeb_128B
class Hexagon_v1024v1024v1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_v32i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v2048v2048i_Intrinsic<string GCCIntSuffix>
// tag : V6_vdmpybus_dv_128B
class Hexagon_v2048v2048i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v64i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v2048v2048v2048i_Intrinsic<string GCCIntSuffix>
// tag : V6_vdmpybus_dv_acc_128B
class Hexagon_v2048v2048v2048i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v64i32_ty,llvm_v64i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v512v512v512v512_Intrinsic<string GCCIntSuffix>
// tag : V6_vdmpyhvsat_acc
class Hexagon_v512v512v512v512_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v16i32_ty,llvm_v16i32_ty,llvm_v16i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v1024v1024v1024v1024_Intrinsic<string GCCIntSuffix>
// tag : V6_vdmpyhvsat_acc_128B
class Hexagon_v1024v1024v1024v1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_v32i32_ty,llvm_v32i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v512v1024i_Intrinsic<string GCCIntSuffix>
// tag : V6_vdmpyhisat
class Hexagon_v512v1024i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v32i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v1024v2048i_Intrinsic<string GCCIntSuffix>
// tag : V6_vdmpyhisat_128B
class Hexagon_v1024v2048i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v64i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v512v512v1024i_Intrinsic<string GCCIntSuffix>
// tag : V6_vdmpyhisat_acc
class Hexagon_v512v512v1024i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v16i32_ty,llvm_v32i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v1024v1024v2048i_Intrinsic<string GCCIntSuffix>
// tag : V6_vdmpyhisat_acc_128B
class Hexagon_v1024v1024v2048i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_v64i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v1024v1024ii_Intrinsic<string GCCIntSuffix>
// tag : V6_vrmpyubi
class Hexagon_v1024v1024ii_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v2048v2048ii_Intrinsic<string GCCIntSuffix>
// tag : V6_vrmpyubi_128B
class Hexagon_v2048v2048ii_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v64i32_ty,llvm_i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v1024v1024v1024ii_Intrinsic<string GCCIntSuffix>
// tag : V6_vrmpyubi_acc
class Hexagon_v1024v1024v1024ii_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_v32i32_ty,llvm_i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v2048v2048v2048ii_Intrinsic<string GCCIntSuffix>
// tag : V6_vrmpyubi_acc_128B
class Hexagon_v2048v2048v2048ii_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v64i32_ty,llvm_v64i32_ty,llvm_i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v2048v2048v2048_Intrinsic<string GCCIntSuffix>
// tag : V6_vaddb_dv_128B
class Hexagon_v2048v2048v2048_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v64i32_ty,llvm_v64i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v1024v512v512_Intrinsic<string GCCIntSuffix>
// tag : V6_vaddubh
class Hexagon_v1024v512v512_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v16i32_ty,llvm_v16i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v2048v1024v1024_Intrinsic<string GCCIntSuffix>
// tag : V6_vaddubh_128B
class Hexagon_v2048v1024v1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v32i32_ty,llvm_v32i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v512_Intrinsic<string GCCIntSuffix>
// tag : V6_vd0
class Hexagon_v512_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [],
                          [IntrNoMem]>;

//
// Hexagon_v1024_Intrinsic<string GCCIntSuffix>
// tag : V6_vd0_128B
class Hexagon_v1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [],
                          [IntrNoMem]>;

//
// Hexagon_v512v64iv512v512_Intrinsic<string GCCIntSuffix>
// tag : V6_vaddbq
class Hexagon_v512v64iv512v512_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v512i1_ty,llvm_v16i32_ty,llvm_v16i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v1024v128iv1024v1024_Intrinsic<string GCCIntSuffix>
// tag : V6_vaddbq_128B
class Hexagon_v1024v128iv1024v1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v1024i1_ty,llvm_v32i32_ty,llvm_v32i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v512v512_Intrinsic<string GCCIntSuffix>
// tag : V6_vabsh
class Hexagon_v512v512_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v16i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v1024v1024_Intrinsic<string GCCIntSuffix>
// tag : V6_vabsh_128B
class Hexagon_v1024v1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v1024v1024v512v512_Intrinsic<string GCCIntSuffix>
// tag : V6_vmpybv_acc
class Hexagon_v1024v1024v512v512_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_v16i32_ty,llvm_v16i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v2048v2048v1024v1024_Intrinsic<string GCCIntSuffix>
// tag : V6_vmpybv_acc_128B
class Hexagon_v2048v2048v1024v1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v64i32_ty,llvm_v32i32_ty,llvm_v32i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v1024v512i_Intrinsic<string GCCIntSuffix>
// tag : V6_vmpyub
class Hexagon_v1024v512i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v16i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v2048v1024i_Intrinsic<string GCCIntSuffix>
// tag : V6_vmpyub_128B
class Hexagon_v2048v1024i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v32i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v1024v1024v512i_Intrinsic<string GCCIntSuffix>
// tag : V6_vmpyub_acc
class Hexagon_v1024v1024v512i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_v16i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v2048v2048v1024i_Intrinsic<string GCCIntSuffix>
// tag : V6_vmpyub_acc_128B
class Hexagon_v2048v2048v1024i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v64i32_ty,llvm_v32i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v512v64ii_Intrinsic<string GCCIntSuffix>
// tag : V6_vandqrt
class Hexagon_v512v64ii_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v512i1_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v1024v128ii_Intrinsic<string GCCIntSuffix>
// tag : V6_vandqrt_128B
class Hexagon_v1024v128ii_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v1024i1_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v512v512v64ii_Intrinsic<string GCCIntSuffix>
// tag : V6_vandqrt_acc
class Hexagon_v512v512v64ii_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v16i32_ty,llvm_v512i1_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v1024v1024v128ii_Intrinsic<string GCCIntSuffix>
// tag : V6_vandqrt_acc_128B
class Hexagon_v1024v1024v128ii_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_v1024i1_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v64iv512i_Intrinsic<string GCCIntSuffix>
// tag : V6_vandvrt
class Hexagon_v64iv512i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v512i1_ty], [llvm_v16i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v128iv1024i_Intrinsic<string GCCIntSuffix>
// tag : V6_vandvrt_128B
class Hexagon_v128iv1024i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v1024i1_ty], [llvm_v32i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v64iv64iv512i_Intrinsic<string GCCIntSuffix>
// tag : V6_vandvrt_acc
class Hexagon_v64iv64iv512i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v512i1_ty], [llvm_v512i1_ty,llvm_v16i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v128iv128iv1024i_Intrinsic<string GCCIntSuffix>
// tag : V6_vandvrt_acc_128B
class Hexagon_v128iv128iv1024i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v1024i1_ty], [llvm_v1024i1_ty,llvm_v32i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v64iv512v512_Intrinsic<string GCCIntSuffix>
// tag : V6_vgtw
class Hexagon_v64iv512v512_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v512i1_ty], [llvm_v16i32_ty,llvm_v16i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v128iv1024v1024_Intrinsic<string GCCIntSuffix>
// tag : V6_vgtw_128B
class Hexagon_v128iv1024v1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v1024i1_ty], [llvm_v32i32_ty,llvm_v32i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v64iv64iv512v512_Intrinsic<string GCCIntSuffix>
// tag : V6_vgtw_and
class Hexagon_v64iv64iv512v512_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v512i1_ty], [llvm_v512i1_ty,llvm_v16i32_ty,llvm_v16i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v128iv128iv1024v1024_Intrinsic<string GCCIntSuffix>
// tag : V6_vgtw_and_128B
class Hexagon_v128iv128iv1024v1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v1024i1_ty], [llvm_v1024i1_ty,llvm_v32i32_ty,llvm_v32i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v64iv64iv64i_Intrinsic<string GCCIntSuffix>
// tag : V6_pred_or
class Hexagon_v64iv64iv64i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v512i1_ty], [llvm_v512i1_ty,llvm_v512i1_ty],
                          [IntrNoMem]>;

//
// Hexagon_v128iv128iv128i_Intrinsic<string GCCIntSuffix>
// tag : V6_pred_or_128B
class Hexagon_v128iv128iv128i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v1024i1_ty], [llvm_v1024i1_ty,llvm_v1024i1_ty],
                          [IntrNoMem]>;

//
// Hexagon_v64iv64i_Intrinsic<string GCCIntSuffix>
// tag : V6_pred_not
class Hexagon_v64iv64i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v512i1_ty], [llvm_v512i1_ty],
                          [IntrNoMem]>;

//
// Hexagon_v128iv128i_Intrinsic<string GCCIntSuffix>
// tag : V6_pred_not_128B
class Hexagon_v128iv128i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v1024i1_ty], [llvm_v1024i1_ty],
                          [IntrNoMem]>;

//
// Hexagon_v64ii_Intrinsic<string GCCIntSuffix>
// tag : V6_pred_scalar2
class Hexagon_v64ii_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v512i1_ty], [llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v128ii_Intrinsic<string GCCIntSuffix>
// tag : V6_pred_scalar2_128B
class Hexagon_v128ii_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v1024i1_ty], [llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v1024v64iv512v512_Intrinsic<string GCCIntSuffix>
// tag : V6_vswap
class Hexagon_v1024v64iv512v512_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v512i1_ty,llvm_v16i32_ty,llvm_v16i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v2048v128iv1024v1024_Intrinsic<string GCCIntSuffix>
// tag : V6_vswap_128B
class Hexagon_v2048v128iv1024v1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v1024i1_ty,llvm_v32i32_ty,llvm_v32i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v1024v512v512i_Intrinsic<string GCCIntSuffix>
// tag : V6_vshuffvdd
class Hexagon_v1024v512v512i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v16i32_ty,llvm_v16i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v2048v1024v1024i_Intrinsic<string GCCIntSuffix>
// tag : V6_vshuffvdd_128B
class Hexagon_v2048v1024v1024i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v32i32_ty,llvm_v32i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;


//
// Hexagon_iv512i_Intrinsic<string GCCIntSuffix>
// tag : V6_extractw
class Hexagon_iv512i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_v16i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_iv1024i_Intrinsic<string GCCIntSuffix>
// tag : V6_extractw_128B
class Hexagon_iv1024i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_v32i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v512i_Intrinsic<string GCCIntSuffix>
// tag : V6_lvsplatw
class Hexagon_v512i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v1024i_Intrinsic<string GCCIntSuffix>
// tag : V6_lvsplatw_128B
class Hexagon_v1024i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v512v512v512v512i_Intrinsic<string GCCIntSuffix>
// tag : V6_vlutvvb_oracc
class Hexagon_v512v512v512v512i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v16i32_ty,llvm_v16i32_ty,llvm_v16i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v1024v1024v1024v1024i_Intrinsic<string GCCIntSuffix>
// tag : V6_vlutvvb_oracc_128B
class Hexagon_v1024v1024v1024v1024i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_v32i32_ty,llvm_v32i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v1024v1024v512v512i_Intrinsic<string GCCIntSuffix>
// tag : V6_vlutvwh_oracc
class Hexagon_v1024v1024v512v512i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_v16i32_ty,llvm_v16i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_v2048v2048v1024v1024i_Intrinsic<string GCCIntSuffix>
// tag : V6_vlutvwh_oracc_128B
class Hexagon_v2048v2048v1024v1024i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v64i32_ty,llvm_v32i32_ty,llvm_v32i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_vv64ivmemv512_Intrinsic<string GCCIntSuffix>
// tag: V6_vS32b_qpred_ai
class Hexagon_vv64ivmemv512_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [], [llvm_v512i1_ty,llvm_ptr_ty,llvm_v16i32_ty],
                          [IntrArgMemOnly]>;

//
// Hexagon_vv128ivmemv1024_Intrinsic<string GCCIntSuffix>
// tag: V6_vS32b_qpred_ai_128B
class Hexagon_vv128ivmemv1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [], [llvm_v1024i1_ty,llvm_ptr_ty,llvm_v32i32_ty],
                          [IntrArgMemOnly]>;

//
// BUILTIN_INFO(HEXAGON.S6_rol_i_r,SI_ftype_SISI,2)
// tag : S6_rol_i_r
def int_hexagon_S6_rol_i_r :
Hexagon_iii_Intrinsic<"HEXAGON_S6_rol_i_r">;

//
// BUILTIN_INFO(HEXAGON.S6_rol_i_p,DI_ftype_DISI,2)
// tag : S6_rol_i_p
def int_hexagon_S6_rol_i_p :
Hexagon_LLiLLii_Intrinsic<"HEXAGON_S6_rol_i_p">;

//
// BUILTIN_INFO(HEXAGON.S6_rol_i_r_acc,SI_ftype_SISISI,3)
// tag : S6_rol_i_r_acc
def int_hexagon_S6_rol_i_r_acc :
Hexagon_iiii_Intrinsic<"HEXAGON_S6_rol_i_r_acc">;

//
// BUILTIN_INFO(HEXAGON.S6_rol_i_p_acc,DI_ftype_DIDISI,3)
// tag : S6_rol_i_p_acc
def int_hexagon_S6_rol_i_p_acc :
Hexagon_LLiLLiLLii_Intrinsic<"HEXAGON_S6_rol_i_p_acc">;

//
// BUILTIN_INFO(HEXAGON.S6_rol_i_r_nac,SI_ftype_SISISI,3)
// tag : S6_rol_i_r_nac
def int_hexagon_S6_rol_i_r_nac :
Hexagon_iiii_Intrinsic<"HEXAGON_S6_rol_i_r_nac">;

//
// BUILTIN_INFO(HEXAGON.S6_rol_i_p_nac,DI_ftype_DIDISI,3)
// tag : S6_rol_i_p_nac
def int_hexagon_S6_rol_i_p_nac :
Hexagon_LLiLLiLLii_Intrinsic<"HEXAGON_S6_rol_i_p_nac">;

//
// BUILTIN_INFO(HEXAGON.S6_rol_i_r_xacc,SI_ftype_SISISI,3)
// tag : S6_rol_i_r_xacc
def int_hexagon_S6_rol_i_r_xacc :
Hexagon_iiii_Intrinsic<"HEXAGON_S6_rol_i_r_xacc">;

//
// BUILTIN_INFO(HEXAGON.S6_rol_i_p_xacc,DI_ftype_DIDISI,3)
// tag : S6_rol_i_p_xacc
def int_hexagon_S6_rol_i_p_xacc :
Hexagon_LLiLLiLLii_Intrinsic<"HEXAGON_S6_rol_i_p_xacc">;

//
// BUILTIN_INFO(HEXAGON.S6_rol_i_r_and,SI_ftype_SISISI,3)
// tag : S6_rol_i_r_and
def int_hexagon_S6_rol_i_r_and :
Hexagon_iiii_Intrinsic<"HEXAGON_S6_rol_i_r_and">;

//
// BUILTIN_INFO(HEXAGON.S6_rol_i_r_or,SI_ftype_SISISI,3)
// tag : S6_rol_i_r_or
def int_hexagon_S6_rol_i_r_or :
Hexagon_iiii_Intrinsic<"HEXAGON_S6_rol_i_r_or">;

//
// BUILTIN_INFO(HEXAGON.S6_rol_i_p_and,DI_ftype_DIDISI,3)
// tag : S6_rol_i_p_and
def int_hexagon_S6_rol_i_p_and :
Hexagon_LLiLLiLLii_Intrinsic<"HEXAGON_S6_rol_i_p_and">;

//
// BUILTIN_INFO(HEXAGON.S6_rol_i_p_or,DI_ftype_DIDISI,3)
// tag : S6_rol_i_p_or
def int_hexagon_S6_rol_i_p_or :
Hexagon_LLiLLiLLii_Intrinsic<"HEXAGON_S6_rol_i_p_or">;

//
// BUILTIN_INFO(HEXAGON.S2_cabacencbin,DI_ftype_DIDIQI,3)
// tag : S2_cabacencbin
def int_hexagon_S2_cabacencbin :
Hexagon_LLiLLiLLii_Intrinsic<"HEXAGON_S2_cabacencbin">;

//
// BUILTIN_INFO(HEXAGON.V6_valignb,VI_ftype_VIVISI,3)
// tag : V6_valignb
def int_hexagon_V6_valignb :
Hexagon_v512v512v512i_Intrinsic<"HEXAGON_V6_valignb">;

//
// BUILTIN_INFO(HEXAGON.V6_valignb_128B,VI_ftype_VIVISI,3)
// tag : V6_valignb_128B
def int_hexagon_V6_valignb_128B :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_valignb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vlalignb,VI_ftype_VIVISI,3)
// tag : V6_vlalignb
def int_hexagon_V6_vlalignb :
Hexagon_v512v512v512i_Intrinsic<"HEXAGON_V6_vlalignb">;

//
// BUILTIN_INFO(HEXAGON.V6_vlalignb_128B,VI_ftype_VIVISI,3)
// tag : V6_vlalignb_128B
def int_hexagon_V6_vlalignb_128B :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vlalignb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_valignbi,VI_ftype_VIVISI,3)
// tag : V6_valignbi
def int_hexagon_V6_valignbi :
Hexagon_v512v512v512i_Intrinsic<"HEXAGON_V6_valignbi">;

//
// BUILTIN_INFO(HEXAGON.V6_valignbi_128B,VI_ftype_VIVISI,3)
// tag : V6_valignbi_128B
def int_hexagon_V6_valignbi_128B :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_valignbi_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vlalignbi,VI_ftype_VIVISI,3)
// tag : V6_vlalignbi
def int_hexagon_V6_vlalignbi :
Hexagon_v512v512v512i_Intrinsic<"HEXAGON_V6_vlalignbi">;

//
// BUILTIN_INFO(HEXAGON.V6_vlalignbi_128B,VI_ftype_VIVISI,3)
// tag : V6_vlalignbi_128B
def int_hexagon_V6_vlalignbi_128B :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vlalignbi_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vror,VI_ftype_VISI,2)
// tag : V6_vror
def int_hexagon_V6_vror :
Hexagon_v512v512i_Intrinsic<"HEXAGON_V6_vror">;

//
// BUILTIN_INFO(HEXAGON.V6_vror_128B,VI_ftype_VISI,2)
// tag : V6_vror_128B
def int_hexagon_V6_vror_128B :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vror_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vunpackub,VD_ftype_VI,1)
// tag : V6_vunpackub
def int_hexagon_V6_vunpackub :
Hexagon_v1024v512_Intrinsic<"HEXAGON_V6_vunpackub">;

//
// BUILTIN_INFO(HEXAGON.V6_vunpackub_128B,VD_ftype_VI,1)
// tag : V6_vunpackub_128B
def int_hexagon_V6_vunpackub_128B :
Hexagon_v2048v1024_Intrinsic<"HEXAGON_V6_vunpackub_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vunpackb,VD_ftype_VI,1)
// tag : V6_vunpackb
def int_hexagon_V6_vunpackb :
Hexagon_v1024v512_Intrinsic<"HEXAGON_V6_vunpackb">;

//
// BUILTIN_INFO(HEXAGON.V6_vunpackb_128B,VD_ftype_VI,1)
// tag : V6_vunpackb_128B
def int_hexagon_V6_vunpackb_128B :
Hexagon_v2048v1024_Intrinsic<"HEXAGON_V6_vunpackb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vunpackuh,VD_ftype_VI,1)
// tag : V6_vunpackuh
def int_hexagon_V6_vunpackuh :
Hexagon_v1024v512_Intrinsic<"HEXAGON_V6_vunpackuh">;

//
// BUILTIN_INFO(HEXAGON.V6_vunpackuh_128B,VD_ftype_VI,1)
// tag : V6_vunpackuh_128B
def int_hexagon_V6_vunpackuh_128B :
Hexagon_v2048v1024_Intrinsic<"HEXAGON_V6_vunpackuh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vunpackh,VD_ftype_VI,1)
// tag : V6_vunpackh
def int_hexagon_V6_vunpackh :
Hexagon_v1024v512_Intrinsic<"HEXAGON_V6_vunpackh">;

//
// BUILTIN_INFO(HEXAGON.V6_vunpackh_128B,VD_ftype_VI,1)
// tag : V6_vunpackh_128B
def int_hexagon_V6_vunpackh_128B :
Hexagon_v2048v1024_Intrinsic<"HEXAGON_V6_vunpackh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vunpackob,VD_ftype_VDVI,2)
// tag : V6_vunpackob
def int_hexagon_V6_vunpackob :
Hexagon_v1024v1024v512_Intrinsic<"HEXAGON_V6_vunpackob">;

//
// BUILTIN_INFO(HEXAGON.V6_vunpackob_128B,VD_ftype_VDVI,2)
// tag : V6_vunpackob_128B
def int_hexagon_V6_vunpackob_128B :
Hexagon_v2048v2048v1024_Intrinsic<"HEXAGON_V6_vunpackob_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vunpackoh,VD_ftype_VDVI,2)
// tag : V6_vunpackoh
def int_hexagon_V6_vunpackoh :
Hexagon_v1024v1024v512_Intrinsic<"HEXAGON_V6_vunpackoh">;

//
// BUILTIN_INFO(HEXAGON.V6_vunpackoh_128B,VD_ftype_VDVI,2)
// tag : V6_vunpackoh_128B
def int_hexagon_V6_vunpackoh_128B :
Hexagon_v2048v2048v1024_Intrinsic<"HEXAGON_V6_vunpackoh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vpackeb,VI_ftype_VIVI,2)
// tag : V6_vpackeb
def int_hexagon_V6_vpackeb :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vpackeb">;

//
// BUILTIN_INFO(HEXAGON.V6_vpackeb_128B,VI_ftype_VIVI,2)
// tag : V6_vpackeb_128B
def int_hexagon_V6_vpackeb_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vpackeb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vpackeh,VI_ftype_VIVI,2)
// tag : V6_vpackeh
def int_hexagon_V6_vpackeh :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vpackeh">;

//
// BUILTIN_INFO(HEXAGON.V6_vpackeh_128B,VI_ftype_VIVI,2)
// tag : V6_vpackeh_128B
def int_hexagon_V6_vpackeh_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vpackeh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vpackob,VI_ftype_VIVI,2)
// tag : V6_vpackob
def int_hexagon_V6_vpackob :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vpackob">;

//
// BUILTIN_INFO(HEXAGON.V6_vpackob_128B,VI_ftype_VIVI,2)
// tag : V6_vpackob_128B
def int_hexagon_V6_vpackob_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vpackob_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vpackoh,VI_ftype_VIVI,2)
// tag : V6_vpackoh
def int_hexagon_V6_vpackoh :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vpackoh">;

//
// BUILTIN_INFO(HEXAGON.V6_vpackoh_128B,VI_ftype_VIVI,2)
// tag : V6_vpackoh_128B
def int_hexagon_V6_vpackoh_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vpackoh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vpackhub_sat,VI_ftype_VIVI,2)
// tag : V6_vpackhub_sat
def int_hexagon_V6_vpackhub_sat :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vpackhub_sat">;

//
// BUILTIN_INFO(HEXAGON.V6_vpackhub_sat_128B,VI_ftype_VIVI,2)
// tag : V6_vpackhub_sat_128B
def int_hexagon_V6_vpackhub_sat_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vpackhub_sat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vpackhb_sat,VI_ftype_VIVI,2)
// tag : V6_vpackhb_sat
def int_hexagon_V6_vpackhb_sat :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vpackhb_sat">;

//
// BUILTIN_INFO(HEXAGON.V6_vpackhb_sat_128B,VI_ftype_VIVI,2)
// tag : V6_vpackhb_sat_128B
def int_hexagon_V6_vpackhb_sat_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vpackhb_sat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vpackwuh_sat,VI_ftype_VIVI,2)
// tag : V6_vpackwuh_sat
def int_hexagon_V6_vpackwuh_sat :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vpackwuh_sat">;

//
// BUILTIN_INFO(HEXAGON.V6_vpackwuh_sat_128B,VI_ftype_VIVI,2)
// tag : V6_vpackwuh_sat_128B
def int_hexagon_V6_vpackwuh_sat_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vpackwuh_sat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vpackwh_sat,VI_ftype_VIVI,2)
// tag : V6_vpackwh_sat
def int_hexagon_V6_vpackwh_sat :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vpackwh_sat">;

//
// BUILTIN_INFO(HEXAGON.V6_vpackwh_sat_128B,VI_ftype_VIVI,2)
// tag : V6_vpackwh_sat_128B
def int_hexagon_V6_vpackwh_sat_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vpackwh_sat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vzb,VD_ftype_VI,1)
// tag : V6_vzb
def int_hexagon_V6_vzb :
Hexagon_v1024v512_Intrinsic<"HEXAGON_V6_vzb">;

//
// BUILTIN_INFO(HEXAGON.V6_vzb_128B,VD_ftype_VI,1)
// tag : V6_vzb_128B
def int_hexagon_V6_vzb_128B :
Hexagon_v2048v1024_Intrinsic<"HEXAGON_V6_vzb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsb,VD_ftype_VI,1)
// tag : V6_vsb
def int_hexagon_V6_vsb :
Hexagon_v1024v512_Intrinsic<"HEXAGON_V6_vsb">;

//
// BUILTIN_INFO(HEXAGON.V6_vsb_128B,VD_ftype_VI,1)
// tag : V6_vsb_128B
def int_hexagon_V6_vsb_128B :
Hexagon_v2048v1024_Intrinsic<"HEXAGON_V6_vsb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vzh,VD_ftype_VI,1)
// tag : V6_vzh
def int_hexagon_V6_vzh :
Hexagon_v1024v512_Intrinsic<"HEXAGON_V6_vzh">;

//
// BUILTIN_INFO(HEXAGON.V6_vzh_128B,VD_ftype_VI,1)
// tag : V6_vzh_128B
def int_hexagon_V6_vzh_128B :
Hexagon_v2048v1024_Intrinsic<"HEXAGON_V6_vzh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsh,VD_ftype_VI,1)
// tag : V6_vsh
def int_hexagon_V6_vsh :
Hexagon_v1024v512_Intrinsic<"HEXAGON_V6_vsh">;

//
// BUILTIN_INFO(HEXAGON.V6_vsh_128B,VD_ftype_VI,1)
// tag : V6_vsh_128B
def int_hexagon_V6_vsh_128B :
Hexagon_v2048v1024_Intrinsic<"HEXAGON_V6_vsh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpybus,VI_ftype_VISI,2)
// tag : V6_vdmpybus
def int_hexagon_V6_vdmpybus :
Hexagon_v512v512i_Intrinsic<"HEXAGON_V6_vdmpybus">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpybus_128B,VI_ftype_VISI,2)
// tag : V6_vdmpybus_128B
def int_hexagon_V6_vdmpybus_128B :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vdmpybus_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpybus_acc,VI_ftype_VIVISI,3)
// tag : V6_vdmpybus_acc
def int_hexagon_V6_vdmpybus_acc :
Hexagon_v512v512v512i_Intrinsic<"HEXAGON_V6_vdmpybus_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpybus_acc_128B,VI_ftype_VIVISI,3)
// tag : V6_vdmpybus_acc_128B
def int_hexagon_V6_vdmpybus_acc_128B :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vdmpybus_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpybus_dv,VD_ftype_VDSI,2)
// tag : V6_vdmpybus_dv
def int_hexagon_V6_vdmpybus_dv :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vdmpybus_dv">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpybus_dv_128B,VD_ftype_VDSI,2)
// tag : V6_vdmpybus_dv_128B
def int_hexagon_V6_vdmpybus_dv_128B :
Hexagon_v2048v2048i_Intrinsic<"HEXAGON_V6_vdmpybus_dv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpybus_dv_acc,VD_ftype_VDVDSI,3)
// tag : V6_vdmpybus_dv_acc
def int_hexagon_V6_vdmpybus_dv_acc :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vdmpybus_dv_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpybus_dv_acc_128B,VD_ftype_VDVDSI,3)
// tag : V6_vdmpybus_dv_acc_128B
def int_hexagon_V6_vdmpybus_dv_acc_128B :
Hexagon_v2048v2048v2048i_Intrinsic<"HEXAGON_V6_vdmpybus_dv_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhb,VI_ftype_VISI,2)
// tag : V6_vdmpyhb
def int_hexagon_V6_vdmpyhb :
Hexagon_v512v512i_Intrinsic<"HEXAGON_V6_vdmpyhb">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhb_128B,VI_ftype_VISI,2)
// tag : V6_vdmpyhb_128B
def int_hexagon_V6_vdmpyhb_128B :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vdmpyhb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhb_acc,VI_ftype_VIVISI,3)
// tag : V6_vdmpyhb_acc
def int_hexagon_V6_vdmpyhb_acc :
Hexagon_v512v512v512i_Intrinsic<"HEXAGON_V6_vdmpyhb_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhb_acc_128B,VI_ftype_VIVISI,3)
// tag : V6_vdmpyhb_acc_128B
def int_hexagon_V6_vdmpyhb_acc_128B :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vdmpyhb_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhb_dv,VD_ftype_VDSI,2)
// tag : V6_vdmpyhb_dv
def int_hexagon_V6_vdmpyhb_dv :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vdmpyhb_dv">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhb_dv_128B,VD_ftype_VDSI,2)
// tag : V6_vdmpyhb_dv_128B
def int_hexagon_V6_vdmpyhb_dv_128B :
Hexagon_v2048v2048i_Intrinsic<"HEXAGON_V6_vdmpyhb_dv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhb_dv_acc,VD_ftype_VDVDSI,3)
// tag : V6_vdmpyhb_dv_acc
def int_hexagon_V6_vdmpyhb_dv_acc :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vdmpyhb_dv_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhb_dv_acc_128B,VD_ftype_VDVDSI,3)
// tag : V6_vdmpyhb_dv_acc_128B
def int_hexagon_V6_vdmpyhb_dv_acc_128B :
Hexagon_v2048v2048v2048i_Intrinsic<"HEXAGON_V6_vdmpyhb_dv_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhvsat,VI_ftype_VIVI,2)
// tag : V6_vdmpyhvsat
def int_hexagon_V6_vdmpyhvsat :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vdmpyhvsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhvsat_128B,VI_ftype_VIVI,2)
// tag : V6_vdmpyhvsat_128B
def int_hexagon_V6_vdmpyhvsat_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vdmpyhvsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhvsat_acc,VI_ftype_VIVIVI,3)
// tag : V6_vdmpyhvsat_acc
def int_hexagon_V6_vdmpyhvsat_acc :
Hexagon_v512v512v512v512_Intrinsic<"HEXAGON_V6_vdmpyhvsat_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhvsat_acc_128B,VI_ftype_VIVIVI,3)
// tag : V6_vdmpyhvsat_acc_128B
def int_hexagon_V6_vdmpyhvsat_acc_128B :
Hexagon_v1024v1024v1024v1024_Intrinsic<"HEXAGON_V6_vdmpyhvsat_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhsat,VI_ftype_VISI,2)
// tag : V6_vdmpyhsat
def int_hexagon_V6_vdmpyhsat :
Hexagon_v512v512i_Intrinsic<"HEXAGON_V6_vdmpyhsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhsat_128B,VI_ftype_VISI,2)
// tag : V6_vdmpyhsat_128B
def int_hexagon_V6_vdmpyhsat_128B :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vdmpyhsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhsat_acc,VI_ftype_VIVISI,3)
// tag : V6_vdmpyhsat_acc
def int_hexagon_V6_vdmpyhsat_acc :
Hexagon_v512v512v512i_Intrinsic<"HEXAGON_V6_vdmpyhsat_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhsat_acc_128B,VI_ftype_VIVISI,3)
// tag : V6_vdmpyhsat_acc_128B
def int_hexagon_V6_vdmpyhsat_acc_128B :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vdmpyhsat_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhisat,VI_ftype_VDSI,2)
// tag : V6_vdmpyhisat
def int_hexagon_V6_vdmpyhisat :
Hexagon_v512v1024i_Intrinsic<"HEXAGON_V6_vdmpyhisat">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhisat_128B,VI_ftype_VDSI,2)
// tag : V6_vdmpyhisat_128B
def int_hexagon_V6_vdmpyhisat_128B :
Hexagon_v1024v2048i_Intrinsic<"HEXAGON_V6_vdmpyhisat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhisat_acc,VI_ftype_VIVDSI,3)
// tag : V6_vdmpyhisat_acc
def int_hexagon_V6_vdmpyhisat_acc :
Hexagon_v512v512v1024i_Intrinsic<"HEXAGON_V6_vdmpyhisat_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhisat_acc_128B,VI_ftype_VIVDSI,3)
// tag : V6_vdmpyhisat_acc_128B
def int_hexagon_V6_vdmpyhisat_acc_128B :
Hexagon_v1024v1024v2048i_Intrinsic<"HEXAGON_V6_vdmpyhisat_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhsusat,VI_ftype_VISI,2)
// tag : V6_vdmpyhsusat
def int_hexagon_V6_vdmpyhsusat :
Hexagon_v512v512i_Intrinsic<"HEXAGON_V6_vdmpyhsusat">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhsusat_128B,VI_ftype_VISI,2)
// tag : V6_vdmpyhsusat_128B
def int_hexagon_V6_vdmpyhsusat_128B :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vdmpyhsusat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhsusat_acc,VI_ftype_VIVISI,3)
// tag : V6_vdmpyhsusat_acc
def int_hexagon_V6_vdmpyhsusat_acc :
Hexagon_v512v512v512i_Intrinsic<"HEXAGON_V6_vdmpyhsusat_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhsusat_acc_128B,VI_ftype_VIVISI,3)
// tag : V6_vdmpyhsusat_acc_128B
def int_hexagon_V6_vdmpyhsusat_acc_128B :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vdmpyhsusat_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhsuisat,VI_ftype_VDSI,2)
// tag : V6_vdmpyhsuisat
def int_hexagon_V6_vdmpyhsuisat :
Hexagon_v512v1024i_Intrinsic<"HEXAGON_V6_vdmpyhsuisat">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhsuisat_128B,VI_ftype_VDSI,2)
// tag : V6_vdmpyhsuisat_128B
def int_hexagon_V6_vdmpyhsuisat_128B :
Hexagon_v1024v2048i_Intrinsic<"HEXAGON_V6_vdmpyhsuisat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhsuisat_acc,VI_ftype_VIVDSI,3)
// tag : V6_vdmpyhsuisat_acc
def int_hexagon_V6_vdmpyhsuisat_acc :
Hexagon_v512v512v1024i_Intrinsic<"HEXAGON_V6_vdmpyhsuisat_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vdmpyhsuisat_acc_128B,VI_ftype_VIVDSI,3)
// tag : V6_vdmpyhsuisat_acc_128B
def int_hexagon_V6_vdmpyhsuisat_acc_128B :
Hexagon_v1024v1024v2048i_Intrinsic<"HEXAGON_V6_vdmpyhsuisat_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vtmpyb,VD_ftype_VDSI,2)
// tag : V6_vtmpyb
def int_hexagon_V6_vtmpyb :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vtmpyb">;

//
// BUILTIN_INFO(HEXAGON.V6_vtmpyb_128B,VD_ftype_VDSI,2)
// tag : V6_vtmpyb_128B
def int_hexagon_V6_vtmpyb_128B :
Hexagon_v2048v2048i_Intrinsic<"HEXAGON_V6_vtmpyb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vtmpyb_acc,VD_ftype_VDVDSI,3)
// tag : V6_vtmpyb_acc
def int_hexagon_V6_vtmpyb_acc :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vtmpyb_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vtmpyb_acc_128B,VD_ftype_VDVDSI,3)
// tag : V6_vtmpyb_acc_128B
def int_hexagon_V6_vtmpyb_acc_128B :
Hexagon_v2048v2048v2048i_Intrinsic<"HEXAGON_V6_vtmpyb_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vtmpybus,VD_ftype_VDSI,2)
// tag : V6_vtmpybus
def int_hexagon_V6_vtmpybus :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vtmpybus">;

//
// BUILTIN_INFO(HEXAGON.V6_vtmpybus_128B,VD_ftype_VDSI,2)
// tag : V6_vtmpybus_128B
def int_hexagon_V6_vtmpybus_128B :
Hexagon_v2048v2048i_Intrinsic<"HEXAGON_V6_vtmpybus_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vtmpybus_acc,VD_ftype_VDVDSI,3)
// tag : V6_vtmpybus_acc
def int_hexagon_V6_vtmpybus_acc :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vtmpybus_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vtmpybus_acc_128B,VD_ftype_VDVDSI,3)
// tag : V6_vtmpybus_acc_128B
def int_hexagon_V6_vtmpybus_acc_128B :
Hexagon_v2048v2048v2048i_Intrinsic<"HEXAGON_V6_vtmpybus_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vtmpyhb,VD_ftype_VDSI,2)
// tag : V6_vtmpyhb
def int_hexagon_V6_vtmpyhb :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vtmpyhb">;

//
// BUILTIN_INFO(HEXAGON.V6_vtmpyhb_128B,VD_ftype_VDSI,2)
// tag : V6_vtmpyhb_128B
def int_hexagon_V6_vtmpyhb_128B :
Hexagon_v2048v2048i_Intrinsic<"HEXAGON_V6_vtmpyhb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vtmpyhb_acc,VD_ftype_VDVDSI,3)
// tag : V6_vtmpyhb_acc
def int_hexagon_V6_vtmpyhb_acc :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vtmpyhb_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vtmpyhb_acc_128B,VD_ftype_VDVDSI,3)
// tag : V6_vtmpyhb_acc_128B
def int_hexagon_V6_vtmpyhb_acc_128B :
Hexagon_v2048v2048v2048i_Intrinsic<"HEXAGON_V6_vtmpyhb_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpyub,VI_ftype_VISI,2)
// tag : V6_vrmpyub
def int_hexagon_V6_vrmpyub :
Hexagon_v512v512i_Intrinsic<"HEXAGON_V6_vrmpyub">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpyub_128B,VI_ftype_VISI,2)
// tag : V6_vrmpyub_128B
def int_hexagon_V6_vrmpyub_128B :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vrmpyub_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpyub_acc,VI_ftype_VIVISI,3)
// tag : V6_vrmpyub_acc
def int_hexagon_V6_vrmpyub_acc :
Hexagon_v512v512v512i_Intrinsic<"HEXAGON_V6_vrmpyub_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpyub_acc_128B,VI_ftype_VIVISI,3)
// tag : V6_vrmpyub_acc_128B
def int_hexagon_V6_vrmpyub_acc_128B :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vrmpyub_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpyubv,VI_ftype_VIVI,2)
// tag : V6_vrmpyubv
def int_hexagon_V6_vrmpyubv :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vrmpyubv">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpyubv_128B,VI_ftype_VIVI,2)
// tag : V6_vrmpyubv_128B
def int_hexagon_V6_vrmpyubv_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vrmpyubv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpyubv_acc,VI_ftype_VIVIVI,3)
// tag : V6_vrmpyubv_acc
def int_hexagon_V6_vrmpyubv_acc :
Hexagon_v512v512v512v512_Intrinsic<"HEXAGON_V6_vrmpyubv_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpyubv_acc_128B,VI_ftype_VIVIVI,3)
// tag : V6_vrmpyubv_acc_128B
def int_hexagon_V6_vrmpyubv_acc_128B :
Hexagon_v1024v1024v1024v1024_Intrinsic<"HEXAGON_V6_vrmpyubv_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpybv,VI_ftype_VIVI,2)
// tag : V6_vrmpybv
def int_hexagon_V6_vrmpybv :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vrmpybv">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpybv_128B,VI_ftype_VIVI,2)
// tag : V6_vrmpybv_128B
def int_hexagon_V6_vrmpybv_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vrmpybv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpybv_acc,VI_ftype_VIVIVI,3)
// tag : V6_vrmpybv_acc
def int_hexagon_V6_vrmpybv_acc :
Hexagon_v512v512v512v512_Intrinsic<"HEXAGON_V6_vrmpybv_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpybv_acc_128B,VI_ftype_VIVIVI,3)
// tag : V6_vrmpybv_acc_128B
def int_hexagon_V6_vrmpybv_acc_128B :
Hexagon_v1024v1024v1024v1024_Intrinsic<"HEXAGON_V6_vrmpybv_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpyubi,VD_ftype_VDSISI,3)
// tag : V6_vrmpyubi
def int_hexagon_V6_vrmpyubi :
Hexagon_v1024v1024ii_Intrinsic<"HEXAGON_V6_vrmpyubi">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpyubi_128B,VD_ftype_VDSISI,3)
// tag : V6_vrmpyubi_128B
def int_hexagon_V6_vrmpyubi_128B :
Hexagon_v2048v2048ii_Intrinsic<"HEXAGON_V6_vrmpyubi_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpyubi_acc,VD_ftype_VDVDSISI,4)
// tag : V6_vrmpyubi_acc
def int_hexagon_V6_vrmpyubi_acc :
Hexagon_v1024v1024v1024ii_Intrinsic<"HEXAGON_V6_vrmpyubi_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpyubi_acc_128B,VD_ftype_VDVDSISI,4)
// tag : V6_vrmpyubi_acc_128B
def int_hexagon_V6_vrmpyubi_acc_128B :
Hexagon_v2048v2048v2048ii_Intrinsic<"HEXAGON_V6_vrmpyubi_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpybus,VI_ftype_VISI,2)
// tag : V6_vrmpybus
def int_hexagon_V6_vrmpybus :
Hexagon_v512v512i_Intrinsic<"HEXAGON_V6_vrmpybus">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpybus_128B,VI_ftype_VISI,2)
// tag : V6_vrmpybus_128B
def int_hexagon_V6_vrmpybus_128B :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vrmpybus_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpybus_acc,VI_ftype_VIVISI,3)
// tag : V6_vrmpybus_acc
def int_hexagon_V6_vrmpybus_acc :
Hexagon_v512v512v512i_Intrinsic<"HEXAGON_V6_vrmpybus_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpybus_acc_128B,VI_ftype_VIVISI,3)
// tag : V6_vrmpybus_acc_128B
def int_hexagon_V6_vrmpybus_acc_128B :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vrmpybus_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpybusi,VD_ftype_VDSISI,3)
// tag : V6_vrmpybusi
def int_hexagon_V6_vrmpybusi :
Hexagon_v1024v1024ii_Intrinsic<"HEXAGON_V6_vrmpybusi">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpybusi_128B,VD_ftype_VDSISI,3)
// tag : V6_vrmpybusi_128B
def int_hexagon_V6_vrmpybusi_128B :
Hexagon_v2048v2048ii_Intrinsic<"HEXAGON_V6_vrmpybusi_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpybusi_acc,VD_ftype_VDVDSISI,4)
// tag : V6_vrmpybusi_acc
def int_hexagon_V6_vrmpybusi_acc :
Hexagon_v1024v1024v1024ii_Intrinsic<"HEXAGON_V6_vrmpybusi_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpybusi_acc_128B,VD_ftype_VDVDSISI,4)
// tag : V6_vrmpybusi_acc_128B
def int_hexagon_V6_vrmpybusi_acc_128B :
Hexagon_v2048v2048v2048ii_Intrinsic<"HEXAGON_V6_vrmpybusi_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpybusv,VI_ftype_VIVI,2)
// tag : V6_vrmpybusv
def int_hexagon_V6_vrmpybusv :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vrmpybusv">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpybusv_128B,VI_ftype_VIVI,2)
// tag : V6_vrmpybusv_128B
def int_hexagon_V6_vrmpybusv_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vrmpybusv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpybusv_acc,VI_ftype_VIVIVI,3)
// tag : V6_vrmpybusv_acc
def int_hexagon_V6_vrmpybusv_acc :
Hexagon_v512v512v512v512_Intrinsic<"HEXAGON_V6_vrmpybusv_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpybusv_acc_128B,VI_ftype_VIVIVI,3)
// tag : V6_vrmpybusv_acc_128B
def int_hexagon_V6_vrmpybusv_acc_128B :
Hexagon_v1024v1024v1024v1024_Intrinsic<"HEXAGON_V6_vrmpybusv_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdsaduh,VD_ftype_VDSI,2)
// tag : V6_vdsaduh
def int_hexagon_V6_vdsaduh :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vdsaduh">;

//
// BUILTIN_INFO(HEXAGON.V6_vdsaduh_128B,VD_ftype_VDSI,2)
// tag : V6_vdsaduh_128B
def int_hexagon_V6_vdsaduh_128B :
Hexagon_v2048v2048i_Intrinsic<"HEXAGON_V6_vdsaduh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdsaduh_acc,VD_ftype_VDVDSI,3)
// tag : V6_vdsaduh_acc
def int_hexagon_V6_vdsaduh_acc :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vdsaduh_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vdsaduh_acc_128B,VD_ftype_VDVDSI,3)
// tag : V6_vdsaduh_acc_128B
def int_hexagon_V6_vdsaduh_acc_128B :
Hexagon_v2048v2048v2048i_Intrinsic<"HEXAGON_V6_vdsaduh_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vrsadubi,VD_ftype_VDSISI,3)
// tag : V6_vrsadubi
def int_hexagon_V6_vrsadubi :
Hexagon_v1024v1024ii_Intrinsic<"HEXAGON_V6_vrsadubi">;

//
// BUILTIN_INFO(HEXAGON.V6_vrsadubi_128B,VD_ftype_VDSISI,3)
// tag : V6_vrsadubi_128B
def int_hexagon_V6_vrsadubi_128B :
Hexagon_v2048v2048ii_Intrinsic<"HEXAGON_V6_vrsadubi_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vrsadubi_acc,VD_ftype_VDVDSISI,4)
// tag : V6_vrsadubi_acc
def int_hexagon_V6_vrsadubi_acc :
Hexagon_v1024v1024v1024ii_Intrinsic<"HEXAGON_V6_vrsadubi_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vrsadubi_acc_128B,VD_ftype_VDVDSISI,4)
// tag : V6_vrsadubi_acc_128B
def int_hexagon_V6_vrsadubi_acc_128B :
Hexagon_v2048v2048v2048ii_Intrinsic<"HEXAGON_V6_vrsadubi_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrw,VI_ftype_VISI,2)
// tag : V6_vasrw
def int_hexagon_V6_vasrw :
Hexagon_v512v512i_Intrinsic<"HEXAGON_V6_vasrw">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrw_128B,VI_ftype_VISI,2)
// tag : V6_vasrw_128B
def int_hexagon_V6_vasrw_128B :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vasrw_128B">;


//
// BUILTIN_INFO(HEXAGON.V6_vaslw,VI_ftype_VISI,2)
// tag : V6_vaslw
def int_hexagon_V6_vaslw :
Hexagon_v512v512i_Intrinsic<"HEXAGON_V6_vaslw">;

//
// BUILTIN_INFO(HEXAGON.V6_vaslw_128B,VI_ftype_VISI,2)
// tag : V6_vaslw_128B
def int_hexagon_V6_vaslw_128B :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vaslw_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vlsrw,VI_ftype_VISI,2)
// tag : V6_vlsrw
def int_hexagon_V6_vlsrw :
Hexagon_v512v512i_Intrinsic<"HEXAGON_V6_vlsrw">;

//
// BUILTIN_INFO(HEXAGON.V6_vlsrw_128B,VI_ftype_VISI,2)
// tag : V6_vlsrw_128B
def int_hexagon_V6_vlsrw_128B :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vlsrw_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrwv,VI_ftype_VIVI,2)
// tag : V6_vasrwv
def int_hexagon_V6_vasrwv :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vasrwv">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrwv_128B,VI_ftype_VIVI,2)
// tag : V6_vasrwv_128B
def int_hexagon_V6_vasrwv_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vasrwv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaslwv,VI_ftype_VIVI,2)
// tag : V6_vaslwv
def int_hexagon_V6_vaslwv :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vaslwv">;

//
// BUILTIN_INFO(HEXAGON.V6_vaslwv_128B,VI_ftype_VIVI,2)
// tag : V6_vaslwv_128B
def int_hexagon_V6_vaslwv_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vaslwv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vlsrwv,VI_ftype_VIVI,2)
// tag : V6_vlsrwv
def int_hexagon_V6_vlsrwv :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vlsrwv">;

//
// BUILTIN_INFO(HEXAGON.V6_vlsrwv_128B,VI_ftype_VIVI,2)
// tag : V6_vlsrwv_128B
def int_hexagon_V6_vlsrwv_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vlsrwv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrh,VI_ftype_VISI,2)
// tag : V6_vasrh
def int_hexagon_V6_vasrh :
Hexagon_v512v512i_Intrinsic<"HEXAGON_V6_vasrh">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrh_128B,VI_ftype_VISI,2)
// tag : V6_vasrh_128B
def int_hexagon_V6_vasrh_128B :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vasrh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaslh,VI_ftype_VISI,2)
// tag : V6_vaslh
def int_hexagon_V6_vaslh :
Hexagon_v512v512i_Intrinsic<"HEXAGON_V6_vaslh">;

//
// BUILTIN_INFO(HEXAGON.V6_vaslh_128B,VI_ftype_VISI,2)
// tag : V6_vaslh_128B
def int_hexagon_V6_vaslh_128B :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vaslh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vlsrh,VI_ftype_VISI,2)
// tag : V6_vlsrh
def int_hexagon_V6_vlsrh :
Hexagon_v512v512i_Intrinsic<"HEXAGON_V6_vlsrh">;

//
// BUILTIN_INFO(HEXAGON.V6_vlsrh_128B,VI_ftype_VISI,2)
// tag : V6_vlsrh_128B
def int_hexagon_V6_vlsrh_128B :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vlsrh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrhv,VI_ftype_VIVI,2)
// tag : V6_vasrhv
def int_hexagon_V6_vasrhv :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vasrhv">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrhv_128B,VI_ftype_VIVI,2)
// tag : V6_vasrhv_128B
def int_hexagon_V6_vasrhv_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vasrhv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaslhv,VI_ftype_VIVI,2)
// tag : V6_vaslhv
def int_hexagon_V6_vaslhv :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vaslhv">;

//
// BUILTIN_INFO(HEXAGON.V6_vaslhv_128B,VI_ftype_VIVI,2)
// tag : V6_vaslhv_128B
def int_hexagon_V6_vaslhv_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vaslhv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vlsrhv,VI_ftype_VIVI,2)
// tag : V6_vlsrhv
def int_hexagon_V6_vlsrhv :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vlsrhv">;

//
// BUILTIN_INFO(HEXAGON.V6_vlsrhv_128B,VI_ftype_VIVI,2)
// tag : V6_vlsrhv_128B
def int_hexagon_V6_vlsrhv_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vlsrhv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrwh,VI_ftype_VIVISI,3)
// tag : V6_vasrwh
def int_hexagon_V6_vasrwh :
Hexagon_v512v512v512i_Intrinsic<"HEXAGON_V6_vasrwh">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrwh_128B,VI_ftype_VIVISI,3)
// tag : V6_vasrwh_128B
def int_hexagon_V6_vasrwh_128B :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vasrwh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrwhsat,VI_ftype_VIVISI,3)
// tag : V6_vasrwhsat
def int_hexagon_V6_vasrwhsat :
Hexagon_v512v512v512i_Intrinsic<"HEXAGON_V6_vasrwhsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrwhsat_128B,VI_ftype_VIVISI,3)
// tag : V6_vasrwhsat_128B
def int_hexagon_V6_vasrwhsat_128B :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vasrwhsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrwhrndsat,VI_ftype_VIVISI,3)
// tag : V6_vasrwhrndsat
def int_hexagon_V6_vasrwhrndsat :
Hexagon_v512v512v512i_Intrinsic<"HEXAGON_V6_vasrwhrndsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrwhrndsat_128B,VI_ftype_VIVISI,3)
// tag : V6_vasrwhrndsat_128B
def int_hexagon_V6_vasrwhrndsat_128B :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vasrwhrndsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrwuhsat,VI_ftype_VIVISI,3)
// tag : V6_vasrwuhsat
def int_hexagon_V6_vasrwuhsat :
Hexagon_v512v512v512i_Intrinsic<"HEXAGON_V6_vasrwuhsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrwuhsat_128B,VI_ftype_VIVISI,3)
// tag : V6_vasrwuhsat_128B
def int_hexagon_V6_vasrwuhsat_128B :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vasrwuhsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vroundwh,VI_ftype_VIVI,2)
// tag : V6_vroundwh
def int_hexagon_V6_vroundwh :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vroundwh">;

//
// BUILTIN_INFO(HEXAGON.V6_vroundwh_128B,VI_ftype_VIVI,2)
// tag : V6_vroundwh_128B
def int_hexagon_V6_vroundwh_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vroundwh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vroundwuh,VI_ftype_VIVI,2)
// tag : V6_vroundwuh
def int_hexagon_V6_vroundwuh :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vroundwuh">;

//
// BUILTIN_INFO(HEXAGON.V6_vroundwuh_128B,VI_ftype_VIVI,2)
// tag : V6_vroundwuh_128B
def int_hexagon_V6_vroundwuh_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vroundwuh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrhubsat,VI_ftype_VIVISI,3)
// tag : V6_vasrhubsat
def int_hexagon_V6_vasrhubsat :
Hexagon_v512v512v512i_Intrinsic<"HEXAGON_V6_vasrhubsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrhubsat_128B,VI_ftype_VIVISI,3)
// tag : V6_vasrhubsat_128B
def int_hexagon_V6_vasrhubsat_128B :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vasrhubsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrhubrndsat,VI_ftype_VIVISI,3)
// tag : V6_vasrhubrndsat
def int_hexagon_V6_vasrhubrndsat :
Hexagon_v512v512v512i_Intrinsic<"HEXAGON_V6_vasrhubrndsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrhubrndsat_128B,VI_ftype_VIVISI,3)
// tag : V6_vasrhubrndsat_128B
def int_hexagon_V6_vasrhubrndsat_128B :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vasrhubrndsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrhbrndsat,VI_ftype_VIVISI,3)
// tag : V6_vasrhbrndsat
def int_hexagon_V6_vasrhbrndsat :
Hexagon_v512v512v512i_Intrinsic<"HEXAGON_V6_vasrhbrndsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrhbrndsat_128B,VI_ftype_VIVISI,3)
// tag : V6_vasrhbrndsat_128B
def int_hexagon_V6_vasrhbrndsat_128B :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vasrhbrndsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vroundhb,VI_ftype_VIVI,2)
// tag : V6_vroundhb
def int_hexagon_V6_vroundhb :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vroundhb">;

//
// BUILTIN_INFO(HEXAGON.V6_vroundhb_128B,VI_ftype_VIVI,2)
// tag : V6_vroundhb_128B
def int_hexagon_V6_vroundhb_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vroundhb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vroundhub,VI_ftype_VIVI,2)
// tag : V6_vroundhub
def int_hexagon_V6_vroundhub :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vroundhub">;

//
// BUILTIN_INFO(HEXAGON.V6_vroundhub_128B,VI_ftype_VIVI,2)
// tag : V6_vroundhub_128B
def int_hexagon_V6_vroundhub_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vroundhub_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaslw_acc,VI_ftype_VIVISI,3)
// tag : V6_vaslw_acc
def int_hexagon_V6_vaslw_acc :
Hexagon_v512v512v512i_Intrinsic<"HEXAGON_V6_vaslw_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vaslw_acc_128B,VI_ftype_VIVISI,3)
// tag : V6_vaslw_acc_128B
def int_hexagon_V6_vaslw_acc_128B :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vaslw_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrw_acc,VI_ftype_VIVISI,3)
// tag : V6_vasrw_acc
def int_hexagon_V6_vasrw_acc :
Hexagon_v512v512v512i_Intrinsic<"HEXAGON_V6_vasrw_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrw_acc_128B,VI_ftype_VIVISI,3)
// tag : V6_vasrw_acc_128B
def int_hexagon_V6_vasrw_acc_128B :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vasrw_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddb,VI_ftype_VIVI,2)
// tag : V6_vaddb
def int_hexagon_V6_vaddb :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vaddb">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddb_128B,VI_ftype_VIVI,2)
// tag : V6_vaddb_128B
def int_hexagon_V6_vaddb_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vaddb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubb,VI_ftype_VIVI,2)
// tag : V6_vsubb
def int_hexagon_V6_vsubb :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vsubb">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubb_128B,VI_ftype_VIVI,2)
// tag : V6_vsubb_128B
def int_hexagon_V6_vsubb_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vsubb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddb_dv,VD_ftype_VDVD,2)
// tag : V6_vaddb_dv
def int_hexagon_V6_vaddb_dv :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vaddb_dv">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddb_dv_128B,VD_ftype_VDVD,2)
// tag : V6_vaddb_dv_128B
def int_hexagon_V6_vaddb_dv_128B :
Hexagon_v2048v2048v2048_Intrinsic<"HEXAGON_V6_vaddb_dv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubb_dv,VD_ftype_VDVD,2)
// tag : V6_vsubb_dv
def int_hexagon_V6_vsubb_dv :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vsubb_dv">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubb_dv_128B,VD_ftype_VDVD,2)
// tag : V6_vsubb_dv_128B
def int_hexagon_V6_vsubb_dv_128B :
Hexagon_v2048v2048v2048_Intrinsic<"HEXAGON_V6_vsubb_dv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddh,VI_ftype_VIVI,2)
// tag : V6_vaddh
def int_hexagon_V6_vaddh :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vaddh">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddh_128B,VI_ftype_VIVI,2)
// tag : V6_vaddh_128B
def int_hexagon_V6_vaddh_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vaddh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubh,VI_ftype_VIVI,2)
// tag : V6_vsubh
def int_hexagon_V6_vsubh :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vsubh">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubh_128B,VI_ftype_VIVI,2)
// tag : V6_vsubh_128B
def int_hexagon_V6_vsubh_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vsubh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddh_dv,VD_ftype_VDVD,2)
// tag : V6_vaddh_dv
def int_hexagon_V6_vaddh_dv :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vaddh_dv">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddh_dv_128B,VD_ftype_VDVD,2)
// tag : V6_vaddh_dv_128B
def int_hexagon_V6_vaddh_dv_128B :
Hexagon_v2048v2048v2048_Intrinsic<"HEXAGON_V6_vaddh_dv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubh_dv,VD_ftype_VDVD,2)
// tag : V6_vsubh_dv
def int_hexagon_V6_vsubh_dv :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vsubh_dv">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubh_dv_128B,VD_ftype_VDVD,2)
// tag : V6_vsubh_dv_128B
def int_hexagon_V6_vsubh_dv_128B :
Hexagon_v2048v2048v2048_Intrinsic<"HEXAGON_V6_vsubh_dv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddw,VI_ftype_VIVI,2)
// tag : V6_vaddw
def int_hexagon_V6_vaddw :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vaddw">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddw_128B,VI_ftype_VIVI,2)
// tag : V6_vaddw_128B
def int_hexagon_V6_vaddw_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vaddw_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubw,VI_ftype_VIVI,2)
// tag : V6_vsubw
def int_hexagon_V6_vsubw :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vsubw">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubw_128B,VI_ftype_VIVI,2)
// tag : V6_vsubw_128B
def int_hexagon_V6_vsubw_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vsubw_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddw_dv,VD_ftype_VDVD,2)
// tag : V6_vaddw_dv
def int_hexagon_V6_vaddw_dv :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vaddw_dv">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddw_dv_128B,VD_ftype_VDVD,2)
// tag : V6_vaddw_dv_128B
def int_hexagon_V6_vaddw_dv_128B :
Hexagon_v2048v2048v2048_Intrinsic<"HEXAGON_V6_vaddw_dv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubw_dv,VD_ftype_VDVD,2)
// tag : V6_vsubw_dv
def int_hexagon_V6_vsubw_dv :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vsubw_dv">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubw_dv_128B,VD_ftype_VDVD,2)
// tag : V6_vsubw_dv_128B
def int_hexagon_V6_vsubw_dv_128B :
Hexagon_v2048v2048v2048_Intrinsic<"HEXAGON_V6_vsubw_dv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddubsat,VI_ftype_VIVI,2)
// tag : V6_vaddubsat
def int_hexagon_V6_vaddubsat :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vaddubsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddubsat_128B,VI_ftype_VIVI,2)
// tag : V6_vaddubsat_128B
def int_hexagon_V6_vaddubsat_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vaddubsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddubsat_dv,VD_ftype_VDVD,2)
// tag : V6_vaddubsat_dv
def int_hexagon_V6_vaddubsat_dv :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vaddubsat_dv">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddubsat_dv_128B,VD_ftype_VDVD,2)
// tag : V6_vaddubsat_dv_128B
def int_hexagon_V6_vaddubsat_dv_128B :
Hexagon_v2048v2048v2048_Intrinsic<"HEXAGON_V6_vaddubsat_dv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsububsat,VI_ftype_VIVI,2)
// tag : V6_vsububsat
def int_hexagon_V6_vsububsat :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vsububsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vsububsat_128B,VI_ftype_VIVI,2)
// tag : V6_vsububsat_128B
def int_hexagon_V6_vsububsat_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vsububsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsububsat_dv,VD_ftype_VDVD,2)
// tag : V6_vsububsat_dv
def int_hexagon_V6_vsububsat_dv :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vsububsat_dv">;

//
// BUILTIN_INFO(HEXAGON.V6_vsububsat_dv_128B,VD_ftype_VDVD,2)
// tag : V6_vsububsat_dv_128B
def int_hexagon_V6_vsububsat_dv_128B :
Hexagon_v2048v2048v2048_Intrinsic<"HEXAGON_V6_vsububsat_dv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vadduhsat,VI_ftype_VIVI,2)
// tag : V6_vadduhsat
def int_hexagon_V6_vadduhsat :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vadduhsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vadduhsat_128B,VI_ftype_VIVI,2)
// tag : V6_vadduhsat_128B
def int_hexagon_V6_vadduhsat_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vadduhsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vadduhsat_dv,VD_ftype_VDVD,2)
// tag : V6_vadduhsat_dv
def int_hexagon_V6_vadduhsat_dv :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vadduhsat_dv">;

//
// BUILTIN_INFO(HEXAGON.V6_vadduhsat_dv_128B,VD_ftype_VDVD,2)
// tag : V6_vadduhsat_dv_128B
def int_hexagon_V6_vadduhsat_dv_128B :
Hexagon_v2048v2048v2048_Intrinsic<"HEXAGON_V6_vadduhsat_dv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubuhsat,VI_ftype_VIVI,2)
// tag : V6_vsubuhsat
def int_hexagon_V6_vsubuhsat :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vsubuhsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubuhsat_128B,VI_ftype_VIVI,2)
// tag : V6_vsubuhsat_128B
def int_hexagon_V6_vsubuhsat_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vsubuhsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubuhsat_dv,VD_ftype_VDVD,2)
// tag : V6_vsubuhsat_dv
def int_hexagon_V6_vsubuhsat_dv :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vsubuhsat_dv">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubuhsat_dv_128B,VD_ftype_VDVD,2)
// tag : V6_vsubuhsat_dv_128B
def int_hexagon_V6_vsubuhsat_dv_128B :
Hexagon_v2048v2048v2048_Intrinsic<"HEXAGON_V6_vsubuhsat_dv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddhsat,VI_ftype_VIVI,2)
// tag : V6_vaddhsat
def int_hexagon_V6_vaddhsat :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vaddhsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddhsat_128B,VI_ftype_VIVI,2)
// tag : V6_vaddhsat_128B
def int_hexagon_V6_vaddhsat_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vaddhsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddhsat_dv,VD_ftype_VDVD,2)
// tag : V6_vaddhsat_dv
def int_hexagon_V6_vaddhsat_dv :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vaddhsat_dv">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddhsat_dv_128B,VD_ftype_VDVD,2)
// tag : V6_vaddhsat_dv_128B
def int_hexagon_V6_vaddhsat_dv_128B :
Hexagon_v2048v2048v2048_Intrinsic<"HEXAGON_V6_vaddhsat_dv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubhsat,VI_ftype_VIVI,2)
// tag : V6_vsubhsat
def int_hexagon_V6_vsubhsat :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vsubhsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubhsat_128B,VI_ftype_VIVI,2)
// tag : V6_vsubhsat_128B
def int_hexagon_V6_vsubhsat_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vsubhsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubhsat_dv,VD_ftype_VDVD,2)
// tag : V6_vsubhsat_dv
def int_hexagon_V6_vsubhsat_dv :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vsubhsat_dv">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubhsat_dv_128B,VD_ftype_VDVD,2)
// tag : V6_vsubhsat_dv_128B
def int_hexagon_V6_vsubhsat_dv_128B :
Hexagon_v2048v2048v2048_Intrinsic<"HEXAGON_V6_vsubhsat_dv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddwsat,VI_ftype_VIVI,2)
// tag : V6_vaddwsat
def int_hexagon_V6_vaddwsat :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vaddwsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddwsat_128B,VI_ftype_VIVI,2)
// tag : V6_vaddwsat_128B
def int_hexagon_V6_vaddwsat_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vaddwsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddwsat_dv,VD_ftype_VDVD,2)
// tag : V6_vaddwsat_dv
def int_hexagon_V6_vaddwsat_dv :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vaddwsat_dv">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddwsat_dv_128B,VD_ftype_VDVD,2)
// tag : V6_vaddwsat_dv_128B
def int_hexagon_V6_vaddwsat_dv_128B :
Hexagon_v2048v2048v2048_Intrinsic<"HEXAGON_V6_vaddwsat_dv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubwsat,VI_ftype_VIVI,2)
// tag : V6_vsubwsat
def int_hexagon_V6_vsubwsat :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vsubwsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubwsat_128B,VI_ftype_VIVI,2)
// tag : V6_vsubwsat_128B
def int_hexagon_V6_vsubwsat_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vsubwsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubwsat_dv,VD_ftype_VDVD,2)
// tag : V6_vsubwsat_dv
def int_hexagon_V6_vsubwsat_dv :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vsubwsat_dv">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubwsat_dv_128B,VD_ftype_VDVD,2)
// tag : V6_vsubwsat_dv_128B
def int_hexagon_V6_vsubwsat_dv_128B :
Hexagon_v2048v2048v2048_Intrinsic<"HEXAGON_V6_vsubwsat_dv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vavgub,VI_ftype_VIVI,2)
// tag : V6_vavgub
def int_hexagon_V6_vavgub :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vavgub">;

//
// BUILTIN_INFO(HEXAGON.V6_vavgub_128B,VI_ftype_VIVI,2)
// tag : V6_vavgub_128B
def int_hexagon_V6_vavgub_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vavgub_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vavgubrnd,VI_ftype_VIVI,2)
// tag : V6_vavgubrnd
def int_hexagon_V6_vavgubrnd :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vavgubrnd">;

//
// BUILTIN_INFO(HEXAGON.V6_vavgubrnd_128B,VI_ftype_VIVI,2)
// tag : V6_vavgubrnd_128B
def int_hexagon_V6_vavgubrnd_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vavgubrnd_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vavguh,VI_ftype_VIVI,2)
// tag : V6_vavguh
def int_hexagon_V6_vavguh :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vavguh">;

//
// BUILTIN_INFO(HEXAGON.V6_vavguh_128B,VI_ftype_VIVI,2)
// tag : V6_vavguh_128B
def int_hexagon_V6_vavguh_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vavguh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vavguhrnd,VI_ftype_VIVI,2)
// tag : V6_vavguhrnd
def int_hexagon_V6_vavguhrnd :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vavguhrnd">;

//
// BUILTIN_INFO(HEXAGON.V6_vavguhrnd_128B,VI_ftype_VIVI,2)
// tag : V6_vavguhrnd_128B
def int_hexagon_V6_vavguhrnd_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vavguhrnd_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vavgh,VI_ftype_VIVI,2)
// tag : V6_vavgh
def int_hexagon_V6_vavgh :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vavgh">;

//
// BUILTIN_INFO(HEXAGON.V6_vavgh_128B,VI_ftype_VIVI,2)
// tag : V6_vavgh_128B
def int_hexagon_V6_vavgh_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vavgh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vavghrnd,VI_ftype_VIVI,2)
// tag : V6_vavghrnd
def int_hexagon_V6_vavghrnd :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vavghrnd">;

//
// BUILTIN_INFO(HEXAGON.V6_vavghrnd_128B,VI_ftype_VIVI,2)
// tag : V6_vavghrnd_128B
def int_hexagon_V6_vavghrnd_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vavghrnd_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vnavgh,VI_ftype_VIVI,2)
// tag : V6_vnavgh
def int_hexagon_V6_vnavgh :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vnavgh">;

//
// BUILTIN_INFO(HEXAGON.V6_vnavgh_128B,VI_ftype_VIVI,2)
// tag : V6_vnavgh_128B
def int_hexagon_V6_vnavgh_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vnavgh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vavgw,VI_ftype_VIVI,2)
// tag : V6_vavgw
def int_hexagon_V6_vavgw :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vavgw">;

//
// BUILTIN_INFO(HEXAGON.V6_vavgw_128B,VI_ftype_VIVI,2)
// tag : V6_vavgw_128B
def int_hexagon_V6_vavgw_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vavgw_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vavgwrnd,VI_ftype_VIVI,2)
// tag : V6_vavgwrnd
def int_hexagon_V6_vavgwrnd :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vavgwrnd">;

//
// BUILTIN_INFO(HEXAGON.V6_vavgwrnd_128B,VI_ftype_VIVI,2)
// tag : V6_vavgwrnd_128B
def int_hexagon_V6_vavgwrnd_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vavgwrnd_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vnavgw,VI_ftype_VIVI,2)
// tag : V6_vnavgw
def int_hexagon_V6_vnavgw :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vnavgw">;

//
// BUILTIN_INFO(HEXAGON.V6_vnavgw_128B,VI_ftype_VIVI,2)
// tag : V6_vnavgw_128B
def int_hexagon_V6_vnavgw_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vnavgw_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vabsdiffub,VI_ftype_VIVI,2)
// tag : V6_vabsdiffub
def int_hexagon_V6_vabsdiffub :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vabsdiffub">;

//
// BUILTIN_INFO(HEXAGON.V6_vabsdiffub_128B,VI_ftype_VIVI,2)
// tag : V6_vabsdiffub_128B
def int_hexagon_V6_vabsdiffub_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vabsdiffub_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vabsdiffuh,VI_ftype_VIVI,2)
// tag : V6_vabsdiffuh
def int_hexagon_V6_vabsdiffuh :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vabsdiffuh">;

//
// BUILTIN_INFO(HEXAGON.V6_vabsdiffuh_128B,VI_ftype_VIVI,2)
// tag : V6_vabsdiffuh_128B
def int_hexagon_V6_vabsdiffuh_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vabsdiffuh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vabsdiffh,VI_ftype_VIVI,2)
// tag : V6_vabsdiffh
def int_hexagon_V6_vabsdiffh :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vabsdiffh">;

//
// BUILTIN_INFO(HEXAGON.V6_vabsdiffh_128B,VI_ftype_VIVI,2)
// tag : V6_vabsdiffh_128B
def int_hexagon_V6_vabsdiffh_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vabsdiffh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vabsdiffw,VI_ftype_VIVI,2)
// tag : V6_vabsdiffw
def int_hexagon_V6_vabsdiffw :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vabsdiffw">;

//
// BUILTIN_INFO(HEXAGON.V6_vabsdiffw_128B,VI_ftype_VIVI,2)
// tag : V6_vabsdiffw_128B
def int_hexagon_V6_vabsdiffw_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vabsdiffw_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vnavgub,VI_ftype_VIVI,2)
// tag : V6_vnavgub
def int_hexagon_V6_vnavgub :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vnavgub">;

//
// BUILTIN_INFO(HEXAGON.V6_vnavgub_128B,VI_ftype_VIVI,2)
// tag : V6_vnavgub_128B
def int_hexagon_V6_vnavgub_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vnavgub_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddubh,VD_ftype_VIVI,2)
// tag : V6_vaddubh
def int_hexagon_V6_vaddubh :
Hexagon_v1024v512v512_Intrinsic<"HEXAGON_V6_vaddubh">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddubh_128B,VD_ftype_VIVI,2)
// tag : V6_vaddubh_128B
def int_hexagon_V6_vaddubh_128B :
Hexagon_v2048v1024v1024_Intrinsic<"HEXAGON_V6_vaddubh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsububh,VD_ftype_VIVI,2)
// tag : V6_vsububh
def int_hexagon_V6_vsububh :
Hexagon_v1024v512v512_Intrinsic<"HEXAGON_V6_vsububh">;

//
// BUILTIN_INFO(HEXAGON.V6_vsububh_128B,VD_ftype_VIVI,2)
// tag : V6_vsububh_128B
def int_hexagon_V6_vsububh_128B :
Hexagon_v2048v1024v1024_Intrinsic<"HEXAGON_V6_vsububh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddhw,VD_ftype_VIVI,2)
// tag : V6_vaddhw
def int_hexagon_V6_vaddhw :
Hexagon_v1024v512v512_Intrinsic<"HEXAGON_V6_vaddhw">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddhw_128B,VD_ftype_VIVI,2)
// tag : V6_vaddhw_128B
def int_hexagon_V6_vaddhw_128B :
Hexagon_v2048v1024v1024_Intrinsic<"HEXAGON_V6_vaddhw_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubhw,VD_ftype_VIVI,2)
// tag : V6_vsubhw
def int_hexagon_V6_vsubhw :
Hexagon_v1024v512v512_Intrinsic<"HEXAGON_V6_vsubhw">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubhw_128B,VD_ftype_VIVI,2)
// tag : V6_vsubhw_128B
def int_hexagon_V6_vsubhw_128B :
Hexagon_v2048v1024v1024_Intrinsic<"HEXAGON_V6_vsubhw_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vadduhw,VD_ftype_VIVI,2)
// tag : V6_vadduhw
def int_hexagon_V6_vadduhw :
Hexagon_v1024v512v512_Intrinsic<"HEXAGON_V6_vadduhw">;

//
// BUILTIN_INFO(HEXAGON.V6_vadduhw_128B,VD_ftype_VIVI,2)
// tag : V6_vadduhw_128B
def int_hexagon_V6_vadduhw_128B :
Hexagon_v2048v1024v1024_Intrinsic<"HEXAGON_V6_vadduhw_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubuhw,VD_ftype_VIVI,2)
// tag : V6_vsubuhw
def int_hexagon_V6_vsubuhw :
Hexagon_v1024v512v512_Intrinsic<"HEXAGON_V6_vsubuhw">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubuhw_128B,VD_ftype_VIVI,2)
// tag : V6_vsubuhw_128B
def int_hexagon_V6_vsubuhw_128B :
Hexagon_v2048v1024v1024_Intrinsic<"HEXAGON_V6_vsubuhw_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vd0,VI_ftype_,0)
// tag : V6_vd0
def int_hexagon_V6_vd0 :
Hexagon_v512_Intrinsic<"HEXAGON_V6_vd0">;

//
// BUILTIN_INFO(HEXAGON.V6_vd0_128B,VI_ftype_,0)
// tag : V6_vd0_128B
def int_hexagon_V6_vd0_128B :
Hexagon_v1024_Intrinsic<"HEXAGON_V6_vd0_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddbq,VI_ftype_QVVIVI,3)
// tag : V6_vaddbq
def int_hexagon_V6_vaddbq :
Hexagon_v512v64iv512v512_Intrinsic<"HEXAGON_V6_vaddbq">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddbq_128B,VI_ftype_QVVIVI,3)
// tag : V6_vaddbq_128B
def int_hexagon_V6_vaddbq_128B :
Hexagon_v1024v128iv1024v1024_Intrinsic<"HEXAGON_V6_vaddbq_128B">;


//
// BUILTIN_INFO(HEXAGON.V6_vsubbq,VI_ftype_QVVIVI,3)
// tag : V6_vsubbq
def int_hexagon_V6_vsubbq :
Hexagon_v512v64iv512v512_Intrinsic<"HEXAGON_V6_vsubbq">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubbq_128B,VI_ftype_QVVIVI,3)
// tag : V6_vsubbq_128B
def int_hexagon_V6_vsubbq_128B :
Hexagon_v1024v128iv1024v1024_Intrinsic<"HEXAGON_V6_vsubbq_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddbnq,VI_ftype_QVVIVI,3)
// tag : V6_vaddbnq
def int_hexagon_V6_vaddbnq :
Hexagon_v512v64iv512v512_Intrinsic<"HEXAGON_V6_vaddbnq">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddbnq_128B,VI_ftype_QVVIVI,3)
// tag : V6_vaddbnq_128B
def int_hexagon_V6_vaddbnq_128B :
Hexagon_v1024v128iv1024v1024_Intrinsic<"HEXAGON_V6_vaddbnq_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubbnq,VI_ftype_QVVIVI,3)
// tag : V6_vsubbnq
def int_hexagon_V6_vsubbnq :
Hexagon_v512v64iv512v512_Intrinsic<"HEXAGON_V6_vsubbnq">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubbnq_128B,VI_ftype_QVVIVI,3)
// tag : V6_vsubbnq_128B
def int_hexagon_V6_vsubbnq_128B :
Hexagon_v1024v128iv1024v1024_Intrinsic<"HEXAGON_V6_vsubbnq_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddhq,VI_ftype_QVVIVI,3)
// tag : V6_vaddhq
def int_hexagon_V6_vaddhq :
Hexagon_v512v64iv512v512_Intrinsic<"HEXAGON_V6_vaddhq">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddhq_128B,VI_ftype_QVVIVI,3)
// tag : V6_vaddhq_128B
def int_hexagon_V6_vaddhq_128B :
Hexagon_v1024v128iv1024v1024_Intrinsic<"HEXAGON_V6_vaddhq_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubhq,VI_ftype_QVVIVI,3)
// tag : V6_vsubhq
def int_hexagon_V6_vsubhq :
Hexagon_v512v64iv512v512_Intrinsic<"HEXAGON_V6_vsubhq">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubhq_128B,VI_ftype_QVVIVI,3)
// tag : V6_vsubhq_128B
def int_hexagon_V6_vsubhq_128B :
Hexagon_v1024v128iv1024v1024_Intrinsic<"HEXAGON_V6_vsubhq_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddhnq,VI_ftype_QVVIVI,3)
// tag : V6_vaddhnq
def int_hexagon_V6_vaddhnq :
Hexagon_v512v64iv512v512_Intrinsic<"HEXAGON_V6_vaddhnq">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddhnq_128B,VI_ftype_QVVIVI,3)
// tag : V6_vaddhnq_128B
def int_hexagon_V6_vaddhnq_128B :
Hexagon_v1024v128iv1024v1024_Intrinsic<"HEXAGON_V6_vaddhnq_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubhnq,VI_ftype_QVVIVI,3)
// tag : V6_vsubhnq
def int_hexagon_V6_vsubhnq :
Hexagon_v512v64iv512v512_Intrinsic<"HEXAGON_V6_vsubhnq">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubhnq_128B,VI_ftype_QVVIVI,3)
// tag : V6_vsubhnq_128B
def int_hexagon_V6_vsubhnq_128B :
Hexagon_v1024v128iv1024v1024_Intrinsic<"HEXAGON_V6_vsubhnq_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddwq,VI_ftype_QVVIVI,3)
// tag : V6_vaddwq
def int_hexagon_V6_vaddwq :
Hexagon_v512v64iv512v512_Intrinsic<"HEXAGON_V6_vaddwq">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddwq_128B,VI_ftype_QVVIVI,3)
// tag : V6_vaddwq_128B
def int_hexagon_V6_vaddwq_128B :
Hexagon_v1024v128iv1024v1024_Intrinsic<"HEXAGON_V6_vaddwq_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubwq,VI_ftype_QVVIVI,3)
// tag : V6_vsubwq
def int_hexagon_V6_vsubwq :
Hexagon_v512v64iv512v512_Intrinsic<"HEXAGON_V6_vsubwq">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubwq_128B,VI_ftype_QVVIVI,3)
// tag : V6_vsubwq_128B
def int_hexagon_V6_vsubwq_128B :
Hexagon_v1024v128iv1024v1024_Intrinsic<"HEXAGON_V6_vsubwq_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddwnq,VI_ftype_QVVIVI,3)
// tag : V6_vaddwnq
def int_hexagon_V6_vaddwnq :
Hexagon_v512v64iv512v512_Intrinsic<"HEXAGON_V6_vaddwnq">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddwnq_128B,VI_ftype_QVVIVI,3)
// tag : V6_vaddwnq_128B
def int_hexagon_V6_vaddwnq_128B :
Hexagon_v1024v128iv1024v1024_Intrinsic<"HEXAGON_V6_vaddwnq_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubwnq,VI_ftype_QVVIVI,3)
// tag : V6_vsubwnq
def int_hexagon_V6_vsubwnq :
Hexagon_v512v64iv512v512_Intrinsic<"HEXAGON_V6_vsubwnq">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubwnq_128B,VI_ftype_QVVIVI,3)
// tag : V6_vsubwnq_128B
def int_hexagon_V6_vsubwnq_128B :
Hexagon_v1024v128iv1024v1024_Intrinsic<"HEXAGON_V6_vsubwnq_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vabsh,VI_ftype_VI,1)
// tag : V6_vabsh
def int_hexagon_V6_vabsh :
Hexagon_v512v512_Intrinsic<"HEXAGON_V6_vabsh">;

//
// BUILTIN_INFO(HEXAGON.V6_vabsh_128B,VI_ftype_VI,1)
// tag : V6_vabsh_128B
def int_hexagon_V6_vabsh_128B :
Hexagon_v1024v1024_Intrinsic<"HEXAGON_V6_vabsh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vabsh_sat,VI_ftype_VI,1)
// tag : V6_vabsh_sat
def int_hexagon_V6_vabsh_sat :
Hexagon_v512v512_Intrinsic<"HEXAGON_V6_vabsh_sat">;

//
// BUILTIN_INFO(HEXAGON.V6_vabsh_sat_128B,VI_ftype_VI,1)
// tag : V6_vabsh_sat_128B
def int_hexagon_V6_vabsh_sat_128B :
Hexagon_v1024v1024_Intrinsic<"HEXAGON_V6_vabsh_sat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vabsw,VI_ftype_VI,1)
// tag : V6_vabsw
def int_hexagon_V6_vabsw :
Hexagon_v512v512_Intrinsic<"HEXAGON_V6_vabsw">;

//
// BUILTIN_INFO(HEXAGON.V6_vabsw_128B,VI_ftype_VI,1)
// tag : V6_vabsw_128B
def int_hexagon_V6_vabsw_128B :
Hexagon_v1024v1024_Intrinsic<"HEXAGON_V6_vabsw_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vabsw_sat,VI_ftype_VI,1)
// tag : V6_vabsw_sat
def int_hexagon_V6_vabsw_sat :
Hexagon_v512v512_Intrinsic<"HEXAGON_V6_vabsw_sat">;

//
// BUILTIN_INFO(HEXAGON.V6_vabsw_sat_128B,VI_ftype_VI,1)
// tag : V6_vabsw_sat_128B
def int_hexagon_V6_vabsw_sat_128B :
Hexagon_v1024v1024_Intrinsic<"HEXAGON_V6_vabsw_sat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpybv,VD_ftype_VIVI,2)
// tag : V6_vmpybv
def int_hexagon_V6_vmpybv :
Hexagon_v1024v512v512_Intrinsic<"HEXAGON_V6_vmpybv">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpybv_128B,VD_ftype_VIVI,2)
// tag : V6_vmpybv_128B
def int_hexagon_V6_vmpybv_128B :
Hexagon_v2048v1024v1024_Intrinsic<"HEXAGON_V6_vmpybv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpybv_acc,VD_ftype_VDVIVI,3)
// tag : V6_vmpybv_acc
def int_hexagon_V6_vmpybv_acc :
Hexagon_v1024v1024v512v512_Intrinsic<"HEXAGON_V6_vmpybv_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpybv_acc_128B,VD_ftype_VDVIVI,3)
// tag : V6_vmpybv_acc_128B
def int_hexagon_V6_vmpybv_acc_128B :
Hexagon_v2048v2048v1024v1024_Intrinsic<"HEXAGON_V6_vmpybv_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyubv,VD_ftype_VIVI,2)
// tag : V6_vmpyubv
def int_hexagon_V6_vmpyubv :
Hexagon_v1024v512v512_Intrinsic<"HEXAGON_V6_vmpyubv">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyubv_128B,VD_ftype_VIVI,2)
// tag : V6_vmpyubv_128B
def int_hexagon_V6_vmpyubv_128B :
Hexagon_v2048v1024v1024_Intrinsic<"HEXAGON_V6_vmpyubv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyubv_acc,VD_ftype_VDVIVI,3)
// tag : V6_vmpyubv_acc
def int_hexagon_V6_vmpyubv_acc :
Hexagon_v1024v1024v512v512_Intrinsic<"HEXAGON_V6_vmpyubv_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyubv_acc_128B,VD_ftype_VDVIVI,3)
// tag : V6_vmpyubv_acc_128B
def int_hexagon_V6_vmpyubv_acc_128B :
Hexagon_v2048v2048v1024v1024_Intrinsic<"HEXAGON_V6_vmpyubv_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpybusv,VD_ftype_VIVI,2)
// tag : V6_vmpybusv
def int_hexagon_V6_vmpybusv :
Hexagon_v1024v512v512_Intrinsic<"HEXAGON_V6_vmpybusv">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpybusv_128B,VD_ftype_VIVI,2)
// tag : V6_vmpybusv_128B
def int_hexagon_V6_vmpybusv_128B :
Hexagon_v2048v1024v1024_Intrinsic<"HEXAGON_V6_vmpybusv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpybusv_acc,VD_ftype_VDVIVI,3)
// tag : V6_vmpybusv_acc
def int_hexagon_V6_vmpybusv_acc :
Hexagon_v1024v1024v512v512_Intrinsic<"HEXAGON_V6_vmpybusv_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpybusv_acc_128B,VD_ftype_VDVIVI,3)
// tag : V6_vmpybusv_acc_128B
def int_hexagon_V6_vmpybusv_acc_128B :
Hexagon_v2048v2048v1024v1024_Intrinsic<"HEXAGON_V6_vmpybusv_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpabusv,VD_ftype_VDVD,2)
// tag : V6_vmpabusv
def int_hexagon_V6_vmpabusv :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vmpabusv">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpabusv_128B,VD_ftype_VDVD,2)
// tag : V6_vmpabusv_128B
def int_hexagon_V6_vmpabusv_128B :
Hexagon_v2048v2048v2048_Intrinsic<"HEXAGON_V6_vmpabusv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpabuuv,VD_ftype_VDVD,2)
// tag : V6_vmpabuuv
def int_hexagon_V6_vmpabuuv :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vmpabuuv">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpabuuv_128B,VD_ftype_VDVD,2)
// tag : V6_vmpabuuv_128B
def int_hexagon_V6_vmpabuuv_128B :
Hexagon_v2048v2048v2048_Intrinsic<"HEXAGON_V6_vmpabuuv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyhv,VD_ftype_VIVI,2)
// tag : V6_vmpyhv
def int_hexagon_V6_vmpyhv :
Hexagon_v1024v512v512_Intrinsic<"HEXAGON_V6_vmpyhv">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyhv_128B,VD_ftype_VIVI,2)
// tag : V6_vmpyhv_128B
def int_hexagon_V6_vmpyhv_128B :
Hexagon_v2048v1024v1024_Intrinsic<"HEXAGON_V6_vmpyhv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyhv_acc,VD_ftype_VDVIVI,3)
// tag : V6_vmpyhv_acc
def int_hexagon_V6_vmpyhv_acc :
Hexagon_v1024v1024v512v512_Intrinsic<"HEXAGON_V6_vmpyhv_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyhv_acc_128B,VD_ftype_VDVIVI,3)
// tag : V6_vmpyhv_acc_128B
def int_hexagon_V6_vmpyhv_acc_128B :
Hexagon_v2048v2048v1024v1024_Intrinsic<"HEXAGON_V6_vmpyhv_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyuhv,VD_ftype_VIVI,2)
// tag : V6_vmpyuhv
def int_hexagon_V6_vmpyuhv :
Hexagon_v1024v512v512_Intrinsic<"HEXAGON_V6_vmpyuhv">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyuhv_128B,VD_ftype_VIVI,2)
// tag : V6_vmpyuhv_128B
def int_hexagon_V6_vmpyuhv_128B :
Hexagon_v2048v1024v1024_Intrinsic<"HEXAGON_V6_vmpyuhv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyuhv_acc,VD_ftype_VDVIVI,3)
// tag : V6_vmpyuhv_acc
def int_hexagon_V6_vmpyuhv_acc :
Hexagon_v1024v1024v512v512_Intrinsic<"HEXAGON_V6_vmpyuhv_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyuhv_acc_128B,VD_ftype_VDVIVI,3)
// tag : V6_vmpyuhv_acc_128B
def int_hexagon_V6_vmpyuhv_acc_128B :
Hexagon_v2048v2048v1024v1024_Intrinsic<"HEXAGON_V6_vmpyuhv_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyhvsrs,VI_ftype_VIVI,2)
// tag : V6_vmpyhvsrs
def int_hexagon_V6_vmpyhvsrs :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vmpyhvsrs">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyhvsrs_128B,VI_ftype_VIVI,2)
// tag : V6_vmpyhvsrs_128B
def int_hexagon_V6_vmpyhvsrs_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vmpyhvsrs_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyhus,VD_ftype_VIVI,2)
// tag : V6_vmpyhus
def int_hexagon_V6_vmpyhus :
Hexagon_v1024v512v512_Intrinsic<"HEXAGON_V6_vmpyhus">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyhus_128B,VD_ftype_VIVI,2)
// tag : V6_vmpyhus_128B
def int_hexagon_V6_vmpyhus_128B :
Hexagon_v2048v1024v1024_Intrinsic<"HEXAGON_V6_vmpyhus_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyhus_acc,VD_ftype_VDVIVI,3)
// tag : V6_vmpyhus_acc
def int_hexagon_V6_vmpyhus_acc :
Hexagon_v1024v1024v512v512_Intrinsic<"HEXAGON_V6_vmpyhus_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyhus_acc_128B,VD_ftype_VDVIVI,3)
// tag : V6_vmpyhus_acc_128B
def int_hexagon_V6_vmpyhus_acc_128B :
Hexagon_v2048v2048v1024v1024_Intrinsic<"HEXAGON_V6_vmpyhus_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyih,VI_ftype_VIVI,2)
// tag : V6_vmpyih
def int_hexagon_V6_vmpyih :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vmpyih">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyih_128B,VI_ftype_VIVI,2)
// tag : V6_vmpyih_128B
def int_hexagon_V6_vmpyih_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vmpyih_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyih_acc,VI_ftype_VIVIVI,3)
// tag : V6_vmpyih_acc
def int_hexagon_V6_vmpyih_acc :
Hexagon_v512v512v512v512_Intrinsic<"HEXAGON_V6_vmpyih_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyih_acc_128B,VI_ftype_VIVIVI,3)
// tag : V6_vmpyih_acc_128B
def int_hexagon_V6_vmpyih_acc_128B :
Hexagon_v1024v1024v1024v1024_Intrinsic<"HEXAGON_V6_vmpyih_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyewuh,VI_ftype_VIVI,2)
// tag : V6_vmpyewuh
def int_hexagon_V6_vmpyewuh :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vmpyewuh">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyewuh_128B,VI_ftype_VIVI,2)
// tag : V6_vmpyewuh_128B
def int_hexagon_V6_vmpyewuh_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vmpyewuh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyowh,VI_ftype_VIVI,2)
// tag : V6_vmpyowh
def int_hexagon_V6_vmpyowh :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vmpyowh">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyowh_128B,VI_ftype_VIVI,2)
// tag : V6_vmpyowh_128B
def int_hexagon_V6_vmpyowh_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vmpyowh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyowh_rnd,VI_ftype_VIVI,2)
// tag : V6_vmpyowh_rnd
def int_hexagon_V6_vmpyowh_rnd :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vmpyowh_rnd">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyowh_rnd_128B,VI_ftype_VIVI,2)
// tag : V6_vmpyowh_rnd_128B
def int_hexagon_V6_vmpyowh_rnd_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vmpyowh_rnd_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyowh_sacc,VI_ftype_VIVIVI,3)
// tag : V6_vmpyowh_sacc
def int_hexagon_V6_vmpyowh_sacc :
Hexagon_v512v512v512v512_Intrinsic<"HEXAGON_V6_vmpyowh_sacc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyowh_sacc_128B,VI_ftype_VIVIVI,3)
// tag : V6_vmpyowh_sacc_128B
def int_hexagon_V6_vmpyowh_sacc_128B :
Hexagon_v1024v1024v1024v1024_Intrinsic<"HEXAGON_V6_vmpyowh_sacc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyowh_rnd_sacc,VI_ftype_VIVIVI,3)
// tag : V6_vmpyowh_rnd_sacc
def int_hexagon_V6_vmpyowh_rnd_sacc :
Hexagon_v512v512v512v512_Intrinsic<"HEXAGON_V6_vmpyowh_rnd_sacc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyowh_rnd_sacc_128B,VI_ftype_VIVIVI,3)
// tag : V6_vmpyowh_rnd_sacc_128B
def int_hexagon_V6_vmpyowh_rnd_sacc_128B :
Hexagon_v1024v1024v1024v1024_Intrinsic<"HEXAGON_V6_vmpyowh_rnd_sacc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyieoh,VI_ftype_VIVI,2)
// tag : V6_vmpyieoh
def int_hexagon_V6_vmpyieoh :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vmpyieoh">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyieoh_128B,VI_ftype_VIVI,2)
// tag : V6_vmpyieoh_128B
def int_hexagon_V6_vmpyieoh_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vmpyieoh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyiewuh,VI_ftype_VIVI,2)
// tag : V6_vmpyiewuh
def int_hexagon_V6_vmpyiewuh :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vmpyiewuh">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyiewuh_128B,VI_ftype_VIVI,2)
// tag : V6_vmpyiewuh_128B
def int_hexagon_V6_vmpyiewuh_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vmpyiewuh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyiowh,VI_ftype_VIVI,2)
// tag : V6_vmpyiowh
def int_hexagon_V6_vmpyiowh :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vmpyiowh">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyiowh_128B,VI_ftype_VIVI,2)
// tag : V6_vmpyiowh_128B
def int_hexagon_V6_vmpyiowh_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vmpyiowh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyiewh_acc,VI_ftype_VIVIVI,3)
// tag : V6_vmpyiewh_acc
def int_hexagon_V6_vmpyiewh_acc :
Hexagon_v512v512v512v512_Intrinsic<"HEXAGON_V6_vmpyiewh_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyiewh_acc_128B,VI_ftype_VIVIVI,3)
// tag : V6_vmpyiewh_acc_128B
def int_hexagon_V6_vmpyiewh_acc_128B :
Hexagon_v1024v1024v1024v1024_Intrinsic<"HEXAGON_V6_vmpyiewh_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyiewuh_acc,VI_ftype_VIVIVI,3)
// tag : V6_vmpyiewuh_acc
def int_hexagon_V6_vmpyiewuh_acc :
Hexagon_v512v512v512v512_Intrinsic<"HEXAGON_V6_vmpyiewuh_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyiewuh_acc_128B,VI_ftype_VIVIVI,3)
// tag : V6_vmpyiewuh_acc_128B
def int_hexagon_V6_vmpyiewuh_acc_128B :
Hexagon_v1024v1024v1024v1024_Intrinsic<"HEXAGON_V6_vmpyiewuh_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyub,VD_ftype_VISI,2)
// tag : V6_vmpyub
def int_hexagon_V6_vmpyub :
Hexagon_v1024v512i_Intrinsic<"HEXAGON_V6_vmpyub">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyub_128B,VD_ftype_VISI,2)
// tag : V6_vmpyub_128B
def int_hexagon_V6_vmpyub_128B :
Hexagon_v2048v1024i_Intrinsic<"HEXAGON_V6_vmpyub_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyub_acc,VD_ftype_VDVISI,3)
// tag : V6_vmpyub_acc
def int_hexagon_V6_vmpyub_acc :
Hexagon_v1024v1024v512i_Intrinsic<"HEXAGON_V6_vmpyub_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyub_acc_128B,VD_ftype_VDVISI,3)
// tag : V6_vmpyub_acc_128B
def int_hexagon_V6_vmpyub_acc_128B :
Hexagon_v2048v2048v1024i_Intrinsic<"HEXAGON_V6_vmpyub_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpybus,VD_ftype_VISI,2)
// tag : V6_vmpybus
def int_hexagon_V6_vmpybus :
Hexagon_v1024v512i_Intrinsic<"HEXAGON_V6_vmpybus">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpybus_128B,VD_ftype_VISI,2)
// tag : V6_vmpybus_128B
def int_hexagon_V6_vmpybus_128B :
Hexagon_v2048v1024i_Intrinsic<"HEXAGON_V6_vmpybus_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpybus_acc,VD_ftype_VDVISI,3)
// tag : V6_vmpybus_acc
def int_hexagon_V6_vmpybus_acc :
Hexagon_v1024v1024v512i_Intrinsic<"HEXAGON_V6_vmpybus_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpybus_acc_128B,VD_ftype_VDVISI,3)
// tag : V6_vmpybus_acc_128B
def int_hexagon_V6_vmpybus_acc_128B :
Hexagon_v2048v2048v1024i_Intrinsic<"HEXAGON_V6_vmpybus_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpabus,VD_ftype_VDSI,2)
// tag : V6_vmpabus
def int_hexagon_V6_vmpabus :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vmpabus">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpabus_128B,VD_ftype_VDSI,2)
// tag : V6_vmpabus_128B
def int_hexagon_V6_vmpabus_128B :
Hexagon_v2048v2048i_Intrinsic<"HEXAGON_V6_vmpabus_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpabus_acc,VD_ftype_VDVDSI,3)
// tag : V6_vmpabus_acc
def int_hexagon_V6_vmpabus_acc :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vmpabus_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpabus_acc_128B,VD_ftype_VDVDSI,3)
// tag : V6_vmpabus_acc_128B
def int_hexagon_V6_vmpabus_acc_128B :
Hexagon_v2048v2048v2048i_Intrinsic<"HEXAGON_V6_vmpabus_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpahb,VD_ftype_VDSI,2)
// tag : V6_vmpahb
def int_hexagon_V6_vmpahb :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vmpahb">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpahb_128B,VD_ftype_VDSI,2)
// tag : V6_vmpahb_128B
def int_hexagon_V6_vmpahb_128B :
Hexagon_v2048v2048i_Intrinsic<"HEXAGON_V6_vmpahb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpahb_acc,VD_ftype_VDVDSI,3)
// tag : V6_vmpahb_acc
def int_hexagon_V6_vmpahb_acc :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vmpahb_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpahb_acc_128B,VD_ftype_VDVDSI,3)
// tag : V6_vmpahb_acc_128B
def int_hexagon_V6_vmpahb_acc_128B :
Hexagon_v2048v2048v2048i_Intrinsic<"HEXAGON_V6_vmpahb_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyh,VD_ftype_VISI,2)
// tag : V6_vmpyh
def int_hexagon_V6_vmpyh :
Hexagon_v1024v512i_Intrinsic<"HEXAGON_V6_vmpyh">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyh_128B,VD_ftype_VISI,2)
// tag : V6_vmpyh_128B
def int_hexagon_V6_vmpyh_128B :
Hexagon_v2048v1024i_Intrinsic<"HEXAGON_V6_vmpyh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyhsat_acc,VD_ftype_VDVISI,3)
// tag : V6_vmpyhsat_acc
def int_hexagon_V6_vmpyhsat_acc :
Hexagon_v1024v1024v512i_Intrinsic<"HEXAGON_V6_vmpyhsat_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyhsat_acc_128B,VD_ftype_VDVISI,3)
// tag : V6_vmpyhsat_acc_128B
def int_hexagon_V6_vmpyhsat_acc_128B :
Hexagon_v2048v2048v1024i_Intrinsic<"HEXAGON_V6_vmpyhsat_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyhss,VI_ftype_VISI,2)
// tag : V6_vmpyhss
def int_hexagon_V6_vmpyhss :
Hexagon_v512v512i_Intrinsic<"HEXAGON_V6_vmpyhss">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyhss_128B,VI_ftype_VISI,2)
// tag : V6_vmpyhss_128B
def int_hexagon_V6_vmpyhss_128B :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vmpyhss_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyhsrs,VI_ftype_VISI,2)
// tag : V6_vmpyhsrs
def int_hexagon_V6_vmpyhsrs :
Hexagon_v512v512i_Intrinsic<"HEXAGON_V6_vmpyhsrs">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyhsrs_128B,VI_ftype_VISI,2)
// tag : V6_vmpyhsrs_128B
def int_hexagon_V6_vmpyhsrs_128B :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vmpyhsrs_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyuh,VD_ftype_VISI,2)
// tag : V6_vmpyuh
def int_hexagon_V6_vmpyuh :
Hexagon_v1024v512i_Intrinsic<"HEXAGON_V6_vmpyuh">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyuh_128B,VD_ftype_VISI,2)
// tag : V6_vmpyuh_128B
def int_hexagon_V6_vmpyuh_128B :
Hexagon_v2048v1024i_Intrinsic<"HEXAGON_V6_vmpyuh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyuh_acc,VD_ftype_VDVISI,3)
// tag : V6_vmpyuh_acc
def int_hexagon_V6_vmpyuh_acc :
Hexagon_v1024v1024v512i_Intrinsic<"HEXAGON_V6_vmpyuh_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyuh_acc_128B,VD_ftype_VDVISI,3)
// tag : V6_vmpyuh_acc_128B
def int_hexagon_V6_vmpyuh_acc_128B :
Hexagon_v2048v2048v1024i_Intrinsic<"HEXAGON_V6_vmpyuh_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyihb,VI_ftype_VISI,2)
// tag : V6_vmpyihb
def int_hexagon_V6_vmpyihb :
Hexagon_v512v512i_Intrinsic<"HEXAGON_V6_vmpyihb">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyihb_128B,VI_ftype_VISI,2)
// tag : V6_vmpyihb_128B
def int_hexagon_V6_vmpyihb_128B :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vmpyihb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyihb_acc,VI_ftype_VIVISI,3)
// tag : V6_vmpyihb_acc
def int_hexagon_V6_vmpyihb_acc :
Hexagon_v512v512v512i_Intrinsic<"HEXAGON_V6_vmpyihb_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyihb_acc_128B,VI_ftype_VIVISI,3)
// tag : V6_vmpyihb_acc_128B
def int_hexagon_V6_vmpyihb_acc_128B :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vmpyihb_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyiwb,VI_ftype_VISI,2)
// tag : V6_vmpyiwb
def int_hexagon_V6_vmpyiwb :
Hexagon_v512v512i_Intrinsic<"HEXAGON_V6_vmpyiwb">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyiwb_128B,VI_ftype_VISI,2)
// tag : V6_vmpyiwb_128B
def int_hexagon_V6_vmpyiwb_128B :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vmpyiwb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyiwb_acc,VI_ftype_VIVISI,3)
// tag : V6_vmpyiwb_acc
def int_hexagon_V6_vmpyiwb_acc :
Hexagon_v512v512v512i_Intrinsic<"HEXAGON_V6_vmpyiwb_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyiwb_acc_128B,VI_ftype_VIVISI,3)
// tag : V6_vmpyiwb_acc_128B
def int_hexagon_V6_vmpyiwb_acc_128B :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vmpyiwb_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyiwh,VI_ftype_VISI,2)
// tag : V6_vmpyiwh
def int_hexagon_V6_vmpyiwh :
Hexagon_v512v512i_Intrinsic<"HEXAGON_V6_vmpyiwh">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyiwh_128B,VI_ftype_VISI,2)
// tag : V6_vmpyiwh_128B
def int_hexagon_V6_vmpyiwh_128B :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vmpyiwh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyiwh_acc,VI_ftype_VIVISI,3)
// tag : V6_vmpyiwh_acc
def int_hexagon_V6_vmpyiwh_acc :
Hexagon_v512v512v512i_Intrinsic<"HEXAGON_V6_vmpyiwh_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyiwh_acc_128B,VI_ftype_VIVISI,3)
// tag : V6_vmpyiwh_acc_128B
def int_hexagon_V6_vmpyiwh_acc_128B :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vmpyiwh_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vand,VI_ftype_VIVI,2)
// tag : V6_vand
def int_hexagon_V6_vand :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vand">;

//
// BUILTIN_INFO(HEXAGON.V6_vand_128B,VI_ftype_VIVI,2)
// tag : V6_vand_128B
def int_hexagon_V6_vand_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vand_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vor,VI_ftype_VIVI,2)
// tag : V6_vor
def int_hexagon_V6_vor :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vor">;

//
// BUILTIN_INFO(HEXAGON.V6_vor_128B,VI_ftype_VIVI,2)
// tag : V6_vor_128B
def int_hexagon_V6_vor_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vor_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vxor,VI_ftype_VIVI,2)
// tag : V6_vxor
def int_hexagon_V6_vxor :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vxor">;

//
// BUILTIN_INFO(HEXAGON.V6_vxor_128B,VI_ftype_VIVI,2)
// tag : V6_vxor_128B
def int_hexagon_V6_vxor_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vxor_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vnot,VI_ftype_VI,1)
// tag : V6_vnot
def int_hexagon_V6_vnot :
Hexagon_v512v512_Intrinsic<"HEXAGON_V6_vnot">;

//
// BUILTIN_INFO(HEXAGON.V6_vnot_128B,VI_ftype_VI,1)
// tag : V6_vnot_128B
def int_hexagon_V6_vnot_128B :
Hexagon_v1024v1024_Intrinsic<"HEXAGON_V6_vnot_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vandqrt,VI_ftype_QVSI,2)
// tag : V6_vandqrt
def int_hexagon_V6_vandqrt :
Hexagon_v512v64ii_Intrinsic<"HEXAGON_V6_vandqrt">;

//
// BUILTIN_INFO(HEXAGON.V6_vandqrt_128B,VI_ftype_QVSI,2)
// tag : V6_vandqrt_128B
def int_hexagon_V6_vandqrt_128B :
Hexagon_v1024v128ii_Intrinsic<"HEXAGON_V6_vandqrt_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vandqrt_acc,VI_ftype_VIQVSI,3)
// tag : V6_vandqrt_acc
def int_hexagon_V6_vandqrt_acc :
Hexagon_v512v512v64ii_Intrinsic<"HEXAGON_V6_vandqrt_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vandqrt_acc_128B,VI_ftype_VIQVSI,3)
// tag : V6_vandqrt_acc_128B
def int_hexagon_V6_vandqrt_acc_128B :
Hexagon_v1024v1024v128ii_Intrinsic<"HEXAGON_V6_vandqrt_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vandvrt,QV_ftype_VISI,2)
// tag : V6_vandvrt
def int_hexagon_V6_vandvrt :
Hexagon_v64iv512i_Intrinsic<"HEXAGON_V6_vandvrt">;

//
// BUILTIN_INFO(HEXAGON.V6_vandvrt_128B,QV_ftype_VISI,2)
// tag : V6_vandvrt_128B
def int_hexagon_V6_vandvrt_128B :
Hexagon_v128iv1024i_Intrinsic<"HEXAGON_V6_vandvrt_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vandvrt_acc,QV_ftype_QVVISI,3)
// tag : V6_vandvrt_acc
def int_hexagon_V6_vandvrt_acc :
Hexagon_v64iv64iv512i_Intrinsic<"HEXAGON_V6_vandvrt_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vandvrt_acc_128B,QV_ftype_QVVISI,3)
// tag : V6_vandvrt_acc_128B
def int_hexagon_V6_vandvrt_acc_128B :
Hexagon_v128iv128iv1024i_Intrinsic<"HEXAGON_V6_vandvrt_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtw,QV_ftype_VIVI,2)
// tag : V6_vgtw
def int_hexagon_V6_vgtw :
Hexagon_v64iv512v512_Intrinsic<"HEXAGON_V6_vgtw">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtw_128B,QV_ftype_VIVI,2)
// tag : V6_vgtw_128B
def int_hexagon_V6_vgtw_128B :
Hexagon_v128iv1024v1024_Intrinsic<"HEXAGON_V6_vgtw_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtw_and,QV_ftype_QVVIVI,3)
// tag : V6_vgtw_and
def int_hexagon_V6_vgtw_and :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_vgtw_and">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtw_and_128B,QV_ftype_QVVIVI,3)
// tag : V6_vgtw_and_128B
def int_hexagon_V6_vgtw_and_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_vgtw_and_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtw_or,QV_ftype_QVVIVI,3)
// tag : V6_vgtw_or
def int_hexagon_V6_vgtw_or :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_vgtw_or">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtw_or_128B,QV_ftype_QVVIVI,3)
// tag : V6_vgtw_or_128B
def int_hexagon_V6_vgtw_or_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_vgtw_or_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtw_xor,QV_ftype_QVVIVI,3)
// tag : V6_vgtw_xor
def int_hexagon_V6_vgtw_xor :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_vgtw_xor">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtw_xor_128B,QV_ftype_QVVIVI,3)
// tag : V6_vgtw_xor_128B
def int_hexagon_V6_vgtw_xor_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_vgtw_xor_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_veqw,QV_ftype_VIVI,2)
// tag : V6_veqw
def int_hexagon_V6_veqw :
Hexagon_v64iv512v512_Intrinsic<"HEXAGON_V6_veqw">;

//
// BUILTIN_INFO(HEXAGON.V6_veqw_128B,QV_ftype_VIVI,2)
// tag : V6_veqw_128B
def int_hexagon_V6_veqw_128B :
Hexagon_v128iv1024v1024_Intrinsic<"HEXAGON_V6_veqw_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_veqw_and,QV_ftype_QVVIVI,3)
// tag : V6_veqw_and
def int_hexagon_V6_veqw_and :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_veqw_and">;

//
// BUILTIN_INFO(HEXAGON.V6_veqw_and_128B,QV_ftype_QVVIVI,3)
// tag : V6_veqw_and_128B
def int_hexagon_V6_veqw_and_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_veqw_and_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_veqw_or,QV_ftype_QVVIVI,3)
// tag : V6_veqw_or
def int_hexagon_V6_veqw_or :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_veqw_or">;

//
// BUILTIN_INFO(HEXAGON.V6_veqw_or_128B,QV_ftype_QVVIVI,3)
// tag : V6_veqw_or_128B
def int_hexagon_V6_veqw_or_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_veqw_or_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_veqw_xor,QV_ftype_QVVIVI,3)
// tag : V6_veqw_xor
def int_hexagon_V6_veqw_xor :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_veqw_xor">;

//
// BUILTIN_INFO(HEXAGON.V6_veqw_xor_128B,QV_ftype_QVVIVI,3)
// tag : V6_veqw_xor_128B
def int_hexagon_V6_veqw_xor_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_veqw_xor_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vgth,QV_ftype_VIVI,2)
// tag : V6_vgth
def int_hexagon_V6_vgth :
Hexagon_v64iv512v512_Intrinsic<"HEXAGON_V6_vgth">;

//
// BUILTIN_INFO(HEXAGON.V6_vgth_128B,QV_ftype_VIVI,2)
// tag : V6_vgth_128B
def int_hexagon_V6_vgth_128B :
Hexagon_v128iv1024v1024_Intrinsic<"HEXAGON_V6_vgth_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vgth_and,QV_ftype_QVVIVI,3)
// tag : V6_vgth_and
def int_hexagon_V6_vgth_and :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_vgth_and">;

//
// BUILTIN_INFO(HEXAGON.V6_vgth_and_128B,QV_ftype_QVVIVI,3)
// tag : V6_vgth_and_128B
def int_hexagon_V6_vgth_and_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_vgth_and_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vgth_or,QV_ftype_QVVIVI,3)
// tag : V6_vgth_or
def int_hexagon_V6_vgth_or :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_vgth_or">;

//
// BUILTIN_INFO(HEXAGON.V6_vgth_or_128B,QV_ftype_QVVIVI,3)
// tag : V6_vgth_or_128B
def int_hexagon_V6_vgth_or_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_vgth_or_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vgth_xor,QV_ftype_QVVIVI,3)
// tag : V6_vgth_xor
def int_hexagon_V6_vgth_xor :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_vgth_xor">;

//
// BUILTIN_INFO(HEXAGON.V6_vgth_xor_128B,QV_ftype_QVVIVI,3)
// tag : V6_vgth_xor_128B
def int_hexagon_V6_vgth_xor_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_vgth_xor_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_veqh,QV_ftype_VIVI,2)
// tag : V6_veqh
def int_hexagon_V6_veqh :
Hexagon_v64iv512v512_Intrinsic<"HEXAGON_V6_veqh">;

//
// BUILTIN_INFO(HEXAGON.V6_veqh_128B,QV_ftype_VIVI,2)
// tag : V6_veqh_128B
def int_hexagon_V6_veqh_128B :
Hexagon_v128iv1024v1024_Intrinsic<"HEXAGON_V6_veqh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_veqh_and,QV_ftype_QVVIVI,3)
// tag : V6_veqh_and
def int_hexagon_V6_veqh_and :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_veqh_and">;

//
// BUILTIN_INFO(HEXAGON.V6_veqh_and_128B,QV_ftype_QVVIVI,3)
// tag : V6_veqh_and_128B
def int_hexagon_V6_veqh_and_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_veqh_and_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_veqh_or,QV_ftype_QVVIVI,3)
// tag : V6_veqh_or
def int_hexagon_V6_veqh_or :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_veqh_or">;

//
// BUILTIN_INFO(HEXAGON.V6_veqh_or_128B,QV_ftype_QVVIVI,3)
// tag : V6_veqh_or_128B
def int_hexagon_V6_veqh_or_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_veqh_or_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_veqh_xor,QV_ftype_QVVIVI,3)
// tag : V6_veqh_xor
def int_hexagon_V6_veqh_xor :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_veqh_xor">;

//
// BUILTIN_INFO(HEXAGON.V6_veqh_xor_128B,QV_ftype_QVVIVI,3)
// tag : V6_veqh_xor_128B
def int_hexagon_V6_veqh_xor_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_veqh_xor_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtb,QV_ftype_VIVI,2)
// tag : V6_vgtb
def int_hexagon_V6_vgtb :
Hexagon_v64iv512v512_Intrinsic<"HEXAGON_V6_vgtb">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtb_128B,QV_ftype_VIVI,2)
// tag : V6_vgtb_128B
def int_hexagon_V6_vgtb_128B :
Hexagon_v128iv1024v1024_Intrinsic<"HEXAGON_V6_vgtb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtb_and,QV_ftype_QVVIVI,3)
// tag : V6_vgtb_and
def int_hexagon_V6_vgtb_and :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_vgtb_and">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtb_and_128B,QV_ftype_QVVIVI,3)
// tag : V6_vgtb_and_128B
def int_hexagon_V6_vgtb_and_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_vgtb_and_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtb_or,QV_ftype_QVVIVI,3)
// tag : V6_vgtb_or
def int_hexagon_V6_vgtb_or :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_vgtb_or">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtb_or_128B,QV_ftype_QVVIVI,3)
// tag : V6_vgtb_or_128B
def int_hexagon_V6_vgtb_or_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_vgtb_or_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtb_xor,QV_ftype_QVVIVI,3)
// tag : V6_vgtb_xor
def int_hexagon_V6_vgtb_xor :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_vgtb_xor">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtb_xor_128B,QV_ftype_QVVIVI,3)
// tag : V6_vgtb_xor_128B
def int_hexagon_V6_vgtb_xor_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_vgtb_xor_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_veqb,QV_ftype_VIVI,2)
// tag : V6_veqb
def int_hexagon_V6_veqb :
Hexagon_v64iv512v512_Intrinsic<"HEXAGON_V6_veqb">;

//
// BUILTIN_INFO(HEXAGON.V6_veqb_128B,QV_ftype_VIVI,2)
// tag : V6_veqb_128B
def int_hexagon_V6_veqb_128B :
Hexagon_v128iv1024v1024_Intrinsic<"HEXAGON_V6_veqb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_veqb_and,QV_ftype_QVVIVI,3)
// tag : V6_veqb_and
def int_hexagon_V6_veqb_and :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_veqb_and">;

//
// BUILTIN_INFO(HEXAGON.V6_veqb_and_128B,QV_ftype_QVVIVI,3)
// tag : V6_veqb_and_128B
def int_hexagon_V6_veqb_and_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_veqb_and_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_veqb_or,QV_ftype_QVVIVI,3)
// tag : V6_veqb_or
def int_hexagon_V6_veqb_or :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_veqb_or">;

//
// BUILTIN_INFO(HEXAGON.V6_veqb_or_128B,QV_ftype_QVVIVI,3)
// tag : V6_veqb_or_128B
def int_hexagon_V6_veqb_or_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_veqb_or_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_veqb_xor,QV_ftype_QVVIVI,3)
// tag : V6_veqb_xor
def int_hexagon_V6_veqb_xor :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_veqb_xor">;

//
// BUILTIN_INFO(HEXAGON.V6_veqb_xor_128B,QV_ftype_QVVIVI,3)
// tag : V6_veqb_xor_128B
def int_hexagon_V6_veqb_xor_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_veqb_xor_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtuw,QV_ftype_VIVI,2)
// tag : V6_vgtuw
def int_hexagon_V6_vgtuw :
Hexagon_v64iv512v512_Intrinsic<"HEXAGON_V6_vgtuw">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtuw_128B,QV_ftype_VIVI,2)
// tag : V6_vgtuw_128B
def int_hexagon_V6_vgtuw_128B :
Hexagon_v128iv1024v1024_Intrinsic<"HEXAGON_V6_vgtuw_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtuw_and,QV_ftype_QVVIVI,3)
// tag : V6_vgtuw_and
def int_hexagon_V6_vgtuw_and :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_vgtuw_and">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtuw_and_128B,QV_ftype_QVVIVI,3)
// tag : V6_vgtuw_and_128B
def int_hexagon_V6_vgtuw_and_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_vgtuw_and_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtuw_or,QV_ftype_QVVIVI,3)
// tag : V6_vgtuw_or
def int_hexagon_V6_vgtuw_or :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_vgtuw_or">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtuw_or_128B,QV_ftype_QVVIVI,3)
// tag : V6_vgtuw_or_128B
def int_hexagon_V6_vgtuw_or_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_vgtuw_or_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtuw_xor,QV_ftype_QVVIVI,3)
// tag : V6_vgtuw_xor
def int_hexagon_V6_vgtuw_xor :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_vgtuw_xor">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtuw_xor_128B,QV_ftype_QVVIVI,3)
// tag : V6_vgtuw_xor_128B
def int_hexagon_V6_vgtuw_xor_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_vgtuw_xor_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtuh,QV_ftype_VIVI,2)
// tag : V6_vgtuh
def int_hexagon_V6_vgtuh :
Hexagon_v64iv512v512_Intrinsic<"HEXAGON_V6_vgtuh">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtuh_128B,QV_ftype_VIVI,2)
// tag : V6_vgtuh_128B
def int_hexagon_V6_vgtuh_128B :
Hexagon_v128iv1024v1024_Intrinsic<"HEXAGON_V6_vgtuh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtuh_and,QV_ftype_QVVIVI,3)
// tag : V6_vgtuh_and
def int_hexagon_V6_vgtuh_and :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_vgtuh_and">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtuh_and_128B,QV_ftype_QVVIVI,3)
// tag : V6_vgtuh_and_128B
def int_hexagon_V6_vgtuh_and_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_vgtuh_and_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtuh_or,QV_ftype_QVVIVI,3)
// tag : V6_vgtuh_or
def int_hexagon_V6_vgtuh_or :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_vgtuh_or">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtuh_or_128B,QV_ftype_QVVIVI,3)
// tag : V6_vgtuh_or_128B
def int_hexagon_V6_vgtuh_or_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_vgtuh_or_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtuh_xor,QV_ftype_QVVIVI,3)
// tag : V6_vgtuh_xor
def int_hexagon_V6_vgtuh_xor :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_vgtuh_xor">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtuh_xor_128B,QV_ftype_QVVIVI,3)
// tag : V6_vgtuh_xor_128B
def int_hexagon_V6_vgtuh_xor_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_vgtuh_xor_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtub,QV_ftype_VIVI,2)
// tag : V6_vgtub
def int_hexagon_V6_vgtub :
Hexagon_v64iv512v512_Intrinsic<"HEXAGON_V6_vgtub">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtub_128B,QV_ftype_VIVI,2)
// tag : V6_vgtub_128B
def int_hexagon_V6_vgtub_128B :
Hexagon_v128iv1024v1024_Intrinsic<"HEXAGON_V6_vgtub_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtub_and,QV_ftype_QVVIVI,3)
// tag : V6_vgtub_and
def int_hexagon_V6_vgtub_and :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_vgtub_and">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtub_and_128B,QV_ftype_QVVIVI,3)
// tag : V6_vgtub_and_128B
def int_hexagon_V6_vgtub_and_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_vgtub_and_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtub_or,QV_ftype_QVVIVI,3)
// tag : V6_vgtub_or
def int_hexagon_V6_vgtub_or :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_vgtub_or">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtub_or_128B,QV_ftype_QVVIVI,3)
// tag : V6_vgtub_or_128B
def int_hexagon_V6_vgtub_or_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_vgtub_or_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtub_xor,QV_ftype_QVVIVI,3)
// tag : V6_vgtub_xor
def int_hexagon_V6_vgtub_xor :
Hexagon_v64iv64iv512v512_Intrinsic<"HEXAGON_V6_vgtub_xor">;

//
// BUILTIN_INFO(HEXAGON.V6_vgtub_xor_128B,QV_ftype_QVVIVI,3)
// tag : V6_vgtub_xor_128B
def int_hexagon_V6_vgtub_xor_128B :
Hexagon_v128iv128iv1024v1024_Intrinsic<"HEXAGON_V6_vgtub_xor_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_pred_or,QV_ftype_QVQV,2)
// tag : V6_pred_or
def int_hexagon_V6_pred_or :
Hexagon_v64iv64iv64i_Intrinsic<"HEXAGON_V6_pred_or">;

//
// BUILTIN_INFO(HEXAGON.V6_pred_or_128B,QV_ftype_QVQV,2)
// tag : V6_pred_or_128B
def int_hexagon_V6_pred_or_128B :
Hexagon_v128iv128iv128i_Intrinsic<"HEXAGON_V6_pred_or_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_pred_and,QV_ftype_QVQV,2)
// tag : V6_pred_and
def int_hexagon_V6_pred_and :
Hexagon_v64iv64iv64i_Intrinsic<"HEXAGON_V6_pred_and">;

//
// BUILTIN_INFO(HEXAGON.V6_pred_and_128B,QV_ftype_QVQV,2)
// tag : V6_pred_and_128B
def int_hexagon_V6_pred_and_128B :
Hexagon_v128iv128iv128i_Intrinsic<"HEXAGON_V6_pred_and_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_pred_not,QV_ftype_QV,1)
// tag : V6_pred_not
def int_hexagon_V6_pred_not :
Hexagon_v64iv64i_Intrinsic<"HEXAGON_V6_pred_not">;

//
// BUILTIN_INFO(HEXAGON.V6_pred_not_128B,QV_ftype_QV,1)
// tag : V6_pred_not_128B
def int_hexagon_V6_pred_not_128B :
Hexagon_v128iv128i_Intrinsic<"HEXAGON_V6_pred_not_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_pred_xor,QV_ftype_QVQV,2)
// tag : V6_pred_xor
def int_hexagon_V6_pred_xor :
Hexagon_v64iv64iv64i_Intrinsic<"HEXAGON_V6_pred_xor">;

//
// BUILTIN_INFO(HEXAGON.V6_pred_xor_128B,QV_ftype_QVQV,2)
// tag : V6_pred_xor_128B
def int_hexagon_V6_pred_xor_128B :
Hexagon_v128iv128iv128i_Intrinsic<"HEXAGON_V6_pred_xor_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_pred_and_n,QV_ftype_QVQV,2)
// tag : V6_pred_and_n
def int_hexagon_V6_pred_and_n :
Hexagon_v64iv64iv64i_Intrinsic<"HEXAGON_V6_pred_and_n">;

//
// BUILTIN_INFO(HEXAGON.V6_pred_and_n_128B,QV_ftype_QVQV,2)
// tag : V6_pred_and_n_128B
def int_hexagon_V6_pred_and_n_128B :
Hexagon_v128iv128iv128i_Intrinsic<"HEXAGON_V6_pred_and_n_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_pred_or_n,QV_ftype_QVQV,2)
// tag : V6_pred_or_n
def int_hexagon_V6_pred_or_n :
Hexagon_v64iv64iv64i_Intrinsic<"HEXAGON_V6_pred_or_n">;

//
// BUILTIN_INFO(HEXAGON.V6_pred_or_n_128B,QV_ftype_QVQV,2)
// tag : V6_pred_or_n_128B
def int_hexagon_V6_pred_or_n_128B :
Hexagon_v128iv128iv128i_Intrinsic<"HEXAGON_V6_pred_or_n_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_pred_scalar2,QV_ftype_SI,1)
// tag : V6_pred_scalar2
def int_hexagon_V6_pred_scalar2 :
Hexagon_v64ii_Intrinsic<"HEXAGON_V6_pred_scalar2">;

//
// BUILTIN_INFO(HEXAGON.V6_pred_scalar2_128B,QV_ftype_SI,1)
// tag : V6_pred_scalar2_128B
def int_hexagon_V6_pred_scalar2_128B :
Hexagon_v128ii_Intrinsic<"HEXAGON_V6_pred_scalar2_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmux,VI_ftype_QVVIVI,3)
// tag : V6_vmux
def int_hexagon_V6_vmux :
Hexagon_v512v64iv512v512_Intrinsic<"HEXAGON_V6_vmux">;

//
// BUILTIN_INFO(HEXAGON.V6_vmux_128B,VI_ftype_QVVIVI,3)
// tag : V6_vmux_128B
def int_hexagon_V6_vmux_128B :
Hexagon_v1024v128iv1024v1024_Intrinsic<"HEXAGON_V6_vmux_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vswap,VD_ftype_QVVIVI,3)
// tag : V6_vswap
def int_hexagon_V6_vswap :
Hexagon_v1024v64iv512v512_Intrinsic<"HEXAGON_V6_vswap">;

//
// BUILTIN_INFO(HEXAGON.V6_vswap_128B,VD_ftype_QVVIVI,3)
// tag : V6_vswap_128B
def int_hexagon_V6_vswap_128B :
Hexagon_v2048v128iv1024v1024_Intrinsic<"HEXAGON_V6_vswap_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmaxub,VI_ftype_VIVI,2)
// tag : V6_vmaxub
def int_hexagon_V6_vmaxub :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vmaxub">;

//
// BUILTIN_INFO(HEXAGON.V6_vmaxub_128B,VI_ftype_VIVI,2)
// tag : V6_vmaxub_128B
def int_hexagon_V6_vmaxub_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vmaxub_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vminub,VI_ftype_VIVI,2)
// tag : V6_vminub
def int_hexagon_V6_vminub :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vminub">;

//
// BUILTIN_INFO(HEXAGON.V6_vminub_128B,VI_ftype_VIVI,2)
// tag : V6_vminub_128B
def int_hexagon_V6_vminub_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vminub_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmaxuh,VI_ftype_VIVI,2)
// tag : V6_vmaxuh
def int_hexagon_V6_vmaxuh :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vmaxuh">;

//
// BUILTIN_INFO(HEXAGON.V6_vmaxuh_128B,VI_ftype_VIVI,2)
// tag : V6_vmaxuh_128B
def int_hexagon_V6_vmaxuh_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vmaxuh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vminuh,VI_ftype_VIVI,2)
// tag : V6_vminuh
def int_hexagon_V6_vminuh :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vminuh">;

//
// BUILTIN_INFO(HEXAGON.V6_vminuh_128B,VI_ftype_VIVI,2)
// tag : V6_vminuh_128B
def int_hexagon_V6_vminuh_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vminuh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmaxh,VI_ftype_VIVI,2)
// tag : V6_vmaxh
def int_hexagon_V6_vmaxh :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vmaxh">;

//
// BUILTIN_INFO(HEXAGON.V6_vmaxh_128B,VI_ftype_VIVI,2)
// tag : V6_vmaxh_128B
def int_hexagon_V6_vmaxh_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vmaxh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vminh,VI_ftype_VIVI,2)
// tag : V6_vminh
def int_hexagon_V6_vminh :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vminh">;

//
// BUILTIN_INFO(HEXAGON.V6_vminh_128B,VI_ftype_VIVI,2)
// tag : V6_vminh_128B
def int_hexagon_V6_vminh_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vminh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmaxw,VI_ftype_VIVI,2)
// tag : V6_vmaxw
def int_hexagon_V6_vmaxw :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vmaxw">;

//
// BUILTIN_INFO(HEXAGON.V6_vmaxw_128B,VI_ftype_VIVI,2)
// tag : V6_vmaxw_128B
def int_hexagon_V6_vmaxw_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vmaxw_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vminw,VI_ftype_VIVI,2)
// tag : V6_vminw
def int_hexagon_V6_vminw :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vminw">;

//
// BUILTIN_INFO(HEXAGON.V6_vminw_128B,VI_ftype_VIVI,2)
// tag : V6_vminw_128B
def int_hexagon_V6_vminw_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vminw_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsathub,VI_ftype_VIVI,2)
// tag : V6_vsathub
def int_hexagon_V6_vsathub :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vsathub">;

//
// BUILTIN_INFO(HEXAGON.V6_vsathub_128B,VI_ftype_VIVI,2)
// tag : V6_vsathub_128B
def int_hexagon_V6_vsathub_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vsathub_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsatwh,VI_ftype_VIVI,2)
// tag : V6_vsatwh
def int_hexagon_V6_vsatwh :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vsatwh">;

//
// BUILTIN_INFO(HEXAGON.V6_vsatwh_128B,VI_ftype_VIVI,2)
// tag : V6_vsatwh_128B
def int_hexagon_V6_vsatwh_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vsatwh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vshuffeb,VI_ftype_VIVI,2)
// tag : V6_vshuffeb
def int_hexagon_V6_vshuffeb :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vshuffeb">;

//
// BUILTIN_INFO(HEXAGON.V6_vshuffeb_128B,VI_ftype_VIVI,2)
// tag : V6_vshuffeb_128B
def int_hexagon_V6_vshuffeb_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vshuffeb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vshuffob,VI_ftype_VIVI,2)
// tag : V6_vshuffob
def int_hexagon_V6_vshuffob :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vshuffob">;

//
// BUILTIN_INFO(HEXAGON.V6_vshuffob_128B,VI_ftype_VIVI,2)
// tag : V6_vshuffob_128B
def int_hexagon_V6_vshuffob_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vshuffob_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vshufeh,VI_ftype_VIVI,2)
// tag : V6_vshufeh
def int_hexagon_V6_vshufeh :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vshufeh">;

//
// BUILTIN_INFO(HEXAGON.V6_vshufeh_128B,VI_ftype_VIVI,2)
// tag : V6_vshufeh_128B
def int_hexagon_V6_vshufeh_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vshufeh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vshufoh,VI_ftype_VIVI,2)
// tag : V6_vshufoh
def int_hexagon_V6_vshufoh :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vshufoh">;

//
// BUILTIN_INFO(HEXAGON.V6_vshufoh_128B,VI_ftype_VIVI,2)
// tag : V6_vshufoh_128B
def int_hexagon_V6_vshufoh_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vshufoh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vshuffvdd,VD_ftype_VIVISI,3)
// tag : V6_vshuffvdd
def int_hexagon_V6_vshuffvdd :
Hexagon_v1024v512v512i_Intrinsic<"HEXAGON_V6_vshuffvdd">;

//
// BUILTIN_INFO(HEXAGON.V6_vshuffvdd_128B,VD_ftype_VIVISI,3)
// tag : V6_vshuffvdd_128B
def int_hexagon_V6_vshuffvdd_128B :
Hexagon_v2048v1024v1024i_Intrinsic<"HEXAGON_V6_vshuffvdd_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdealvdd,VD_ftype_VIVISI,3)
// tag : V6_vdealvdd
def int_hexagon_V6_vdealvdd :
Hexagon_v1024v512v512i_Intrinsic<"HEXAGON_V6_vdealvdd">;

//
// BUILTIN_INFO(HEXAGON.V6_vdealvdd_128B,VD_ftype_VIVISI,3)
// tag : V6_vdealvdd_128B
def int_hexagon_V6_vdealvdd_128B :
Hexagon_v2048v1024v1024i_Intrinsic<"HEXAGON_V6_vdealvdd_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vshufoeh,VD_ftype_VIVI,2)
// tag : V6_vshufoeh
def int_hexagon_V6_vshufoeh :
Hexagon_v1024v512v512_Intrinsic<"HEXAGON_V6_vshufoeh">;

//
// BUILTIN_INFO(HEXAGON.V6_vshufoeh_128B,VD_ftype_VIVI,2)
// tag : V6_vshufoeh_128B
def int_hexagon_V6_vshufoeh_128B :
Hexagon_v2048v1024v1024_Intrinsic<"HEXAGON_V6_vshufoeh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vshufoeb,VD_ftype_VIVI,2)
// tag : V6_vshufoeb
def int_hexagon_V6_vshufoeb :
Hexagon_v1024v512v512_Intrinsic<"HEXAGON_V6_vshufoeb">;

//
// BUILTIN_INFO(HEXAGON.V6_vshufoeb_128B,VD_ftype_VIVI,2)
// tag : V6_vshufoeb_128B
def int_hexagon_V6_vshufoeb_128B :
Hexagon_v2048v1024v1024_Intrinsic<"HEXAGON_V6_vshufoeb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdealh,VI_ftype_VI,1)
// tag : V6_vdealh
def int_hexagon_V6_vdealh :
Hexagon_v512v512_Intrinsic<"HEXAGON_V6_vdealh">;

//
// BUILTIN_INFO(HEXAGON.V6_vdealh_128B,VI_ftype_VI,1)
// tag : V6_vdealh_128B
def int_hexagon_V6_vdealh_128B :
Hexagon_v1024v1024_Intrinsic<"HEXAGON_V6_vdealh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdealb,VI_ftype_VI,1)
// tag : V6_vdealb
def int_hexagon_V6_vdealb :
Hexagon_v512v512_Intrinsic<"HEXAGON_V6_vdealb">;

//
// BUILTIN_INFO(HEXAGON.V6_vdealb_128B,VI_ftype_VI,1)
// tag : V6_vdealb_128B
def int_hexagon_V6_vdealb_128B :
Hexagon_v1024v1024_Intrinsic<"HEXAGON_V6_vdealb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdealb4w,VI_ftype_VIVI,2)
// tag : V6_vdealb4w
def int_hexagon_V6_vdealb4w :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vdealb4w">;

//
// BUILTIN_INFO(HEXAGON.V6_vdealb4w_128B,VI_ftype_VIVI,2)
// tag : V6_vdealb4w_128B
def int_hexagon_V6_vdealb4w_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vdealb4w_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vshuffh,VI_ftype_VI,1)
// tag : V6_vshuffh
def int_hexagon_V6_vshuffh :
Hexagon_v512v512_Intrinsic<"HEXAGON_V6_vshuffh">;

//
// BUILTIN_INFO(HEXAGON.V6_vshuffh_128B,VI_ftype_VI,1)
// tag : V6_vshuffh_128B
def int_hexagon_V6_vshuffh_128B :
Hexagon_v1024v1024_Intrinsic<"HEXAGON_V6_vshuffh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vshuffb,VI_ftype_VI,1)
// tag : V6_vshuffb
def int_hexagon_V6_vshuffb :
Hexagon_v512v512_Intrinsic<"HEXAGON_V6_vshuffb">;

//
// BUILTIN_INFO(HEXAGON.V6_vshuffb_128B,VI_ftype_VI,1)
// tag : V6_vshuffb_128B
def int_hexagon_V6_vshuffb_128B :
Hexagon_v1024v1024_Intrinsic<"HEXAGON_V6_vshuffb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_extractw,SI_ftype_VISI,2)
// tag : V6_extractw
def int_hexagon_V6_extractw :
Hexagon_iv512i_Intrinsic<"HEXAGON_V6_extractw">;

//
// BUILTIN_INFO(HEXAGON.V6_extractw_128B,SI_ftype_VISI,2)
// tag : V6_extractw_128B
def int_hexagon_V6_extractw_128B :
Hexagon_iv1024i_Intrinsic<"HEXAGON_V6_extractw_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vinsertwr,VI_ftype_VISI,2)
// tag : V6_vinsertwr
def int_hexagon_V6_vinsertwr :
Hexagon_v512v512i_Intrinsic<"HEXAGON_V6_vinsertwr">;

//
// BUILTIN_INFO(HEXAGON.V6_vinsertwr_128B,VI_ftype_VISI,2)
// tag : V6_vinsertwr_128B
def int_hexagon_V6_vinsertwr_128B :
Hexagon_v1024v1024i_Intrinsic<"HEXAGON_V6_vinsertwr_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_lvsplatw,VI_ftype_SI,1)
// tag : V6_lvsplatw
def int_hexagon_V6_lvsplatw :
Hexagon_v512i_Intrinsic<"HEXAGON_V6_lvsplatw">;

//
// BUILTIN_INFO(HEXAGON.V6_lvsplatw_128B,VI_ftype_SI,1)
// tag : V6_lvsplatw_128B
def int_hexagon_V6_lvsplatw_128B :
Hexagon_v1024i_Intrinsic<"HEXAGON_V6_lvsplatw_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vassign,VI_ftype_VI,1)
// tag : V6_vassign
def int_hexagon_V6_vassign :
Hexagon_v512v512_Intrinsic<"HEXAGON_V6_vassign">;

//
// BUILTIN_INFO(HEXAGON.V6_vassign_128B,VI_ftype_VI,1)
// tag : V6_vassign_128B
def int_hexagon_V6_vassign_128B :
Hexagon_v1024v1024_Intrinsic<"HEXAGON_V6_vassign_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vcombine,VD_ftype_VIVI,2)
// tag : V6_vcombine
def int_hexagon_V6_vcombine :
Hexagon_v1024v512v512_Intrinsic<"HEXAGON_V6_vcombine">;

//
// BUILTIN_INFO(HEXAGON.V6_vcombine_128B,VD_ftype_VIVI,2)
// tag : V6_vcombine_128B
def int_hexagon_V6_vcombine_128B :
Hexagon_v2048v1024v1024_Intrinsic<"HEXAGON_V6_vcombine_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdelta,VI_ftype_VIVI,2)
// tag : V6_vdelta
def int_hexagon_V6_vdelta :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vdelta">;

//
// BUILTIN_INFO(HEXAGON.V6_vdelta_128B,VI_ftype_VIVI,2)
// tag : V6_vdelta_128B
def int_hexagon_V6_vdelta_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vdelta_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vrdelta,VI_ftype_VIVI,2)
// tag : V6_vrdelta
def int_hexagon_V6_vrdelta :
Hexagon_v512v512v512_Intrinsic<"HEXAGON_V6_vrdelta">;

//
// BUILTIN_INFO(HEXAGON.V6_vrdelta_128B,VI_ftype_VIVI,2)
// tag : V6_vrdelta_128B
def int_hexagon_V6_vrdelta_128B :
Hexagon_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vrdelta_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vcl0w,VI_ftype_VI,1)
// tag : V6_vcl0w
def int_hexagon_V6_vcl0w :
Hexagon_v512v512_Intrinsic<"HEXAGON_V6_vcl0w">;

//
// BUILTIN_INFO(HEXAGON.V6_vcl0w_128B,VI_ftype_VI,1)
// tag : V6_vcl0w_128B
def int_hexagon_V6_vcl0w_128B :
Hexagon_v1024v1024_Intrinsic<"HEXAGON_V6_vcl0w_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vcl0h,VI_ftype_VI,1)
// tag : V6_vcl0h
def int_hexagon_V6_vcl0h :
Hexagon_v512v512_Intrinsic<"HEXAGON_V6_vcl0h">;

//
// BUILTIN_INFO(HEXAGON.V6_vcl0h_128B,VI_ftype_VI,1)
// tag : V6_vcl0h_128B
def int_hexagon_V6_vcl0h_128B :
Hexagon_v1024v1024_Intrinsic<"HEXAGON_V6_vcl0h_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vnormamtw,VI_ftype_VI,1)
// tag : V6_vnormamtw
def int_hexagon_V6_vnormamtw :
Hexagon_v512v512_Intrinsic<"HEXAGON_V6_vnormamtw">;

//
// BUILTIN_INFO(HEXAGON.V6_vnormamtw_128B,VI_ftype_VI,1)
// tag : V6_vnormamtw_128B
def int_hexagon_V6_vnormamtw_128B :
Hexagon_v1024v1024_Intrinsic<"HEXAGON_V6_vnormamtw_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vnormamth,VI_ftype_VI,1)
// tag : V6_vnormamth
def int_hexagon_V6_vnormamth :
Hexagon_v512v512_Intrinsic<"HEXAGON_V6_vnormamth">;

//
// BUILTIN_INFO(HEXAGON.V6_vnormamth_128B,VI_ftype_VI,1)
// tag : V6_vnormamth_128B
def int_hexagon_V6_vnormamth_128B :
Hexagon_v1024v1024_Intrinsic<"HEXAGON_V6_vnormamth_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vpopcounth,VI_ftype_VI,1)
// tag : V6_vpopcounth
def int_hexagon_V6_vpopcounth :
Hexagon_v512v512_Intrinsic<"HEXAGON_V6_vpopcounth">;

//
// BUILTIN_INFO(HEXAGON.V6_vpopcounth_128B,VI_ftype_VI,1)
// tag : V6_vpopcounth_128B
def int_hexagon_V6_vpopcounth_128B :
Hexagon_v1024v1024_Intrinsic<"HEXAGON_V6_vpopcounth_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vlutvvb,VI_ftype_VIVISI,3)
// tag : V6_vlutvvb
def int_hexagon_V6_vlutvvb :
Hexagon_v512v512v512i_Intrinsic<"HEXAGON_V6_vlutvvb">;

//
// BUILTIN_INFO(HEXAGON.V6_vlutvvb_128B,VI_ftype_VIVISI,3)
// tag : V6_vlutvvb_128B
def int_hexagon_V6_vlutvvb_128B :
Hexagon_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vlutvvb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vlutvvb_oracc,VI_ftype_VIVIVISI,4)
// tag : V6_vlutvvb_oracc
def int_hexagon_V6_vlutvvb_oracc :
Hexagon_v512v512v512v512i_Intrinsic<"HEXAGON_V6_vlutvvb_oracc">;

//
// BUILTIN_INFO(HEXAGON.V6_vlutvvb_oracc_128B,VI_ftype_VIVIVISI,4)
// tag : V6_vlutvvb_oracc_128B
def int_hexagon_V6_vlutvvb_oracc_128B :
Hexagon_v1024v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vlutvvb_oracc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vlutvwh,VD_ftype_VIVISI,3)
// tag : V6_vlutvwh
def int_hexagon_V6_vlutvwh :
Hexagon_v1024v512v512i_Intrinsic<"HEXAGON_V6_vlutvwh">;

//
// BUILTIN_INFO(HEXAGON.V6_vlutvwh_128B,VD_ftype_VIVISI,3)
// tag : V6_vlutvwh_128B
def int_hexagon_V6_vlutvwh_128B :
Hexagon_v2048v1024v1024i_Intrinsic<"HEXAGON_V6_vlutvwh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vlutvwh_oracc,VD_ftype_VDVIVISI,4)
// tag : V6_vlutvwh_oracc
def int_hexagon_V6_vlutvwh_oracc :
Hexagon_v1024v1024v512v512i_Intrinsic<"HEXAGON_V6_vlutvwh_oracc">;

//
// BUILTIN_INFO(HEXAGON.V6_vlutvwh_oracc_128B,VD_ftype_VDVIVISI,4)
// tag : V6_vlutvwh_oracc_128B
def int_hexagon_V6_vlutvwh_oracc_128B :
Hexagon_v2048v2048v1024v1024i_Intrinsic<"HEXAGON_V6_vlutvwh_oracc_128B">;

//
// Masked vector stores
//
def int_hexagon_V6_vS32b_qpred_ai :
Hexagon_vv64ivmemv512_Intrinsic<"HEXAGON_V6_vS32b_qpred_ai">;

def int_hexagon_V6_vS32b_nqpred_ai :
Hexagon_vv64ivmemv512_Intrinsic<"HEXAGON_V6_vS32b_nqpred_ai">;

def int_hexagon_V6_vS32b_nt_qpred_ai :
Hexagon_vv64ivmemv512_Intrinsic<"HEXAGON_V6_vS32b_nt_qpred_ai">;

def int_hexagon_V6_vS32b_nt_nqpred_ai :
Hexagon_vv64ivmemv512_Intrinsic<"HEXAGON_V6_vS32b_nt_nqpred_ai">;

def int_hexagon_V6_vS32b_qpred_ai_128B :
Hexagon_vv128ivmemv1024_Intrinsic<"HEXAGON_V6_vS32b_qpred_ai_128B">;

def int_hexagon_V6_vS32b_nqpred_ai_128B :
Hexagon_vv128ivmemv1024_Intrinsic<"HEXAGON_V6_vS32b_nqpred_ai_128B">;

def int_hexagon_V6_vS32b_nt_qpred_ai_128B :
Hexagon_vv128ivmemv1024_Intrinsic<"HEXAGON_V6_vS32b_nt_qpred_ai_128B">;

def int_hexagon_V6_vS32b_nt_nqpred_ai_128B :
Hexagon_vv128ivmemv1024_Intrinsic<"HEXAGON_V6_vS32b_nt_nqpred_ai_128B">;

def int_hexagon_V6_vmaskedstoreq :
Hexagon_vv64ivmemv512_Intrinsic<"HEXAGON_V6_vmaskedstoreq">;

def int_hexagon_V6_vmaskedstorenq :
Hexagon_vv64ivmemv512_Intrinsic<"HEXAGON_V6_vmaskedstorenq">;

def int_hexagon_V6_vmaskedstorentq :
Hexagon_vv64ivmemv512_Intrinsic<"HEXAGON_V6_vmaskedstorentq">;

def int_hexagon_V6_vmaskedstorentnq :
Hexagon_vv64ivmemv512_Intrinsic<"HEXAGON_V6_vmaskedstorentnq">;

def int_hexagon_V6_vmaskedstoreq_128B :
Hexagon_vv128ivmemv1024_Intrinsic<"HEXAGON_V6_vmaskedstoreq_128B">;

def int_hexagon_V6_vmaskedstorenq_128B :
Hexagon_vv128ivmemv1024_Intrinsic<"HEXAGON_V6_vmaskedstorenq_128B">;

def int_hexagon_V6_vmaskedstorentq_128B :
Hexagon_vv128ivmemv1024_Intrinsic<"HEXAGON_V6_vmaskedstorentq_128B">;

def int_hexagon_V6_vmaskedstorentnq_128B :
Hexagon_vv128ivmemv1024_Intrinsic<"HEXAGON_V6_vmaskedstorentnq_128B">;

multiclass Hexagon_custom_circ_ld_Intrinsic<LLVMType ElTy> {
  def NAME#_pci : Hexagon_NonGCC_Intrinsic<
    [ElTy, llvm_ptr_ty],
    [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_ptr_ty],
    [IntrArgMemOnly, NoCapture<3>]>;
  def NAME#_pcr : Hexagon_NonGCC_Intrinsic<
    [ElTy, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty, llvm_ptr_ty],
    [IntrArgMemOnly, NoCapture<2>]>;
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
    [IntrArgMemOnly, NoCapture<4>]>;
  def NAME#_pcr : Hexagon_NonGCC_Intrinsic<
    [llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty, ElTy, llvm_ptr_ty],
    [IntrArgMemOnly, NoCapture<3>]>;
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


///
/// HexagonV62 intrinsics
///

//
// Hexagon_LLiLLiLLi_Intrinsic<string GCCIntSuffix>
// tag : M6_vabsdiffb
class Hexagon_LLiLLiLLi_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty,llvm_i64_ty],
                          [IntrNoMem]>;

//
// Hexagon_LLii_Intrinsic<string GCCIntSuffix>
// tag : S6_vsplatrbp
class Hexagon_LLii_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v512v512i_Intrinsic<string GCCIntSuffix>
// tag : V6_vlsrb
class Hexagon_V62_v512v512i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v16i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v1024v1024i_Intrinsic<string GCCIntSuffix>
// tag : V6_vlsrb_128B
class Hexagon_V62_v1024v1024i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v512v512v512i_Intrinsic<string GCCIntSuffix>
// tag : V6_vasrwuhrndsat
class Hexagon_V62_v512v512v512i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v16i32_ty,llvm_v16i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v1024v1024v1024i_Intrinsic<string GCCIntSuffix>
// tag : V6_vasrwuhrndsat_128B
class Hexagon_V62_v1024v1024v1024i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_v32i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v512v512v512_Intrinsic<string GCCIntSuffix>
// tag : V6_vrounduwuh
class Hexagon_V62_v512v512v512_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v16i32_ty,llvm_v16i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v1024v1024v1024_Intrinsic<string GCCIntSuffix>
// tag : V6_vrounduwuh_128B
class Hexagon_V62_v1024v1024v1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_v32i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v2048v2048v2048_Intrinsic<string GCCIntSuffix>
// tag : V6_vadduwsat_dv_128B
class Hexagon_V62_v2048v2048v2048_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v64i32_ty,llvm_v64i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v1024v1024v512v512_Intrinsic<string GCCIntSuffix>
// tag : V6_vaddhw_acc
class Hexagon_V62_v1024v1024v512v512_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_v16i32_ty,llvm_v16i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v2048v2048v1024v1024_Intrinsic<string GCCIntSuffix>
// tag : V6_vaddhw_acc_128B
class Hexagon_V62_v2048v2048v1024v1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v64i32_ty,llvm_v32i32_ty,llvm_v32i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v1024v512v512_Intrinsic<string GCCIntSuffix>
// tag : V6_vmpyewuh_64
class Hexagon_V62_v1024v512v512_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v16i32_ty,llvm_v16i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v2048v1024v1024_Intrinsic<string GCCIntSuffix>
// tag : V6_vmpyewuh_64_128B
class Hexagon_V62_v2048v1024v1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v32i32_ty,llvm_v32i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v2048v2048i_Intrinsic<string GCCIntSuffix>
// tag : V6_vmpauhb_128B
class Hexagon_V62_v2048v2048i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v64i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v2048v2048v2048i_Intrinsic<string GCCIntSuffix>
// tag : V6_vmpauhb_acc_128B
class Hexagon_V62_v2048v2048v2048i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v64i32_ty,llvm_v64i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v512v64ii_Intrinsic<string GCCIntSuffix>
// tag : V6_vandnqrt
class Hexagon_V62_v512v64ii_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v512i1_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v1024v128ii_Intrinsic<string GCCIntSuffix>
// tag : V6_vandnqrt_128B
class Hexagon_V62_v1024v128ii_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v1024i1_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v512v512v64ii_Intrinsic<string GCCIntSuffix>
// tag : V6_vandnqrt_acc
class Hexagon_V62_v512v512v64ii_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v16i32_ty,llvm_v512i1_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v1024v1024v128ii_Intrinsic<string GCCIntSuffix>
// tag : V6_vandnqrt_acc_128B
class Hexagon_V62_v1024v1024v128ii_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_v1024i1_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v512v64iv512_Intrinsic<string GCCIntSuffix>
// tag : V6_vandvqv
class Hexagon_V62_v512v64iv512_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v512i1_ty,llvm_v16i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v1024v128iv1024_Intrinsic<string GCCIntSuffix>
// tag : V6_vandvqv_128B
class Hexagon_V62_v1024v128iv1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v1024i1_ty,llvm_v32i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v64ii_Intrinsic<string GCCIntSuffix>
// tag : V6_pred_scalar2v2
class Hexagon_V62_v64ii_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v512i1_ty], [llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v128ii_Intrinsic<string GCCIntSuffix>
// tag : V6_pred_scalar2v2_128B
class Hexagon_V62_v128ii_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v1024i1_ty], [llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v64iv64iv64i_Intrinsic<string GCCIntSuffix>
// tag : V6_shuffeqw
class Hexagon_V62_v64iv64iv64i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v512i1_ty], [llvm_v512i1_ty,llvm_v512i1_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v128iv128iv128i_Intrinsic<string GCCIntSuffix>
// tag : V6_shuffeqw_128B
class Hexagon_V62_v128iv128iv128i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v1024i1_ty], [llvm_v1024i1_ty,llvm_v1024i1_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v512i_Intrinsic<string GCCIntSuffix>
// tag : V6_lvsplath
class Hexagon_V62_v512i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v1024i_Intrinsic<string GCCIntSuffix>
// tag : V6_lvsplath_128B
class Hexagon_V62_v1024i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v512v512v512v512i_Intrinsic<string GCCIntSuffix>
// tag : V6_vlutvvb_oracci
class Hexagon_V62_v512v512v512v512i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v16i32_ty,llvm_v16i32_ty,llvm_v16i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v1024v1024v1024v1024i_Intrinsic<string GCCIntSuffix>
// tag : V6_vlutvvb_oracci_128B
class Hexagon_V62_v1024v1024v1024v1024i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_v32i32_ty,llvm_v32i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v1024v512v512i_Intrinsic<string GCCIntSuffix>
// tag : V6_vlutvwhi
class Hexagon_V62_v1024v512v512i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v16i32_ty,llvm_v16i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v2048v1024v1024i_Intrinsic<string GCCIntSuffix>
// tag : V6_vlutvwhi_128B
class Hexagon_V62_v2048v1024v1024i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v32i32_ty,llvm_v32i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v1024v1024v512v512i_Intrinsic<string GCCIntSuffix>
// tag : V6_vlutvwh_oracci
class Hexagon_V62_v1024v1024v512v512i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_v16i32_ty,llvm_v16i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V62_v2048v2048v1024v1024i_Intrinsic<string GCCIntSuffix>
// tag : V6_vlutvwh_oracci_128B
class Hexagon_V62_v2048v2048v1024v1024i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v64i32_ty,llvm_v32i32_ty,llvm_v32i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

// Hexagon_v512v64iv512v512v64i_Intrinsic<string GCCIntSuffix>
// tag: V6_vaddcarry
class Hexagon_v512v64iv512v512v64i_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty, llvm_v512i1_ty], [llvm_v16i32_ty,llvm_v16i32_ty,llvm_v512i1_ty],
                          [IntrNoMem]>;

// Hexagon_v1024v128iv1024v1024v128i_Intrinsic<string GCCIntSuffix>
// tag: V6_vaddcarry_128B
class Hexagon_v1024v128iv1024v1024v128i_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty, llvm_v1024i1_ty], [llvm_v32i32_ty,llvm_v32i32_ty,llvm_v1024i1_ty],
                          [IntrNoMem]>;


//
// BUILTIN_INFO(HEXAGON.M6_vabsdiffb,DI_ftype_DIDI,2)
// tag : M6_vabsdiffb
def int_hexagon_M6_vabsdiffb :
Hexagon_LLiLLiLLi_Intrinsic<"HEXAGON_M6_vabsdiffb">;

//
// BUILTIN_INFO(HEXAGON.M6_vabsdiffub,DI_ftype_DIDI,2)
// tag : M6_vabsdiffub
def int_hexagon_M6_vabsdiffub :
Hexagon_LLiLLiLLi_Intrinsic<"HEXAGON_M6_vabsdiffub">;

//
// BUILTIN_INFO(HEXAGON.S6_vtrunehb_ppp,DI_ftype_DIDI,2)
// tag : S6_vtrunehb_ppp
def int_hexagon_S6_vtrunehb_ppp :
Hexagon_LLiLLiLLi_Intrinsic<"HEXAGON_S6_vtrunehb_ppp">;

//
// BUILTIN_INFO(HEXAGON.S6_vtrunohb_ppp,DI_ftype_DIDI,2)
// tag : S6_vtrunohb_ppp
def int_hexagon_S6_vtrunohb_ppp :
Hexagon_LLiLLiLLi_Intrinsic<"HEXAGON_S6_vtrunohb_ppp">;

//
// BUILTIN_INFO(HEXAGON.S6_vsplatrbp,DI_ftype_SI,1)
// tag : S6_vsplatrbp
def int_hexagon_S6_vsplatrbp :
Hexagon_LLii_Intrinsic<"HEXAGON_S6_vsplatrbp">;

//
// BUILTIN_INFO(HEXAGON.V6_vlsrb,VI_ftype_VISI,2)
// tag : V6_vlsrb
def int_hexagon_V6_vlsrb :
Hexagon_V62_v512v512i_Intrinsic<"HEXAGON_V6_vlsrb">;

//
// BUILTIN_INFO(HEXAGON.V6_vlsrb_128B,VI_ftype_VISI,2)
// tag : V6_vlsrb_128B
def int_hexagon_V6_vlsrb_128B :
Hexagon_V62_v1024v1024i_Intrinsic<"HEXAGON_V6_vlsrb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrwuhrndsat,VI_ftype_VIVISI,3)
// tag : V6_vasrwuhrndsat
def int_hexagon_V6_vasrwuhrndsat :
Hexagon_V62_v512v512v512i_Intrinsic<"HEXAGON_V6_vasrwuhrndsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrwuhrndsat_128B,VI_ftype_VIVISI,3)
// tag : V6_vasrwuhrndsat_128B
def int_hexagon_V6_vasrwuhrndsat_128B :
Hexagon_V62_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vasrwuhrndsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vasruwuhrndsat,VI_ftype_VIVISI,3)
// tag : V6_vasruwuhrndsat
def int_hexagon_V6_vasruwuhrndsat :
Hexagon_V62_v512v512v512i_Intrinsic<"HEXAGON_V6_vasruwuhrndsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vasruwuhrndsat_128B,VI_ftype_VIVISI,3)
// tag : V6_vasruwuhrndsat_128B
def int_hexagon_V6_vasruwuhrndsat_128B :
Hexagon_V62_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vasruwuhrndsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrhbsat,VI_ftype_VIVISI,3)
// tag : V6_vasrhbsat
def int_hexagon_V6_vasrhbsat :
Hexagon_V62_v512v512v512i_Intrinsic<"HEXAGON_V6_vasrhbsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrhbsat_128B,VI_ftype_VIVISI,3)
// tag : V6_vasrhbsat_128B
def int_hexagon_V6_vasrhbsat_128B :
Hexagon_V62_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vasrhbsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vrounduwuh,VI_ftype_VIVI,2)
// tag : V6_vrounduwuh
def int_hexagon_V6_vrounduwuh :
Hexagon_V62_v512v512v512_Intrinsic<"HEXAGON_V6_vrounduwuh">;

//
// BUILTIN_INFO(HEXAGON.V6_vrounduwuh_128B,VI_ftype_VIVI,2)
// tag : V6_vrounduwuh_128B
def int_hexagon_V6_vrounduwuh_128B :
Hexagon_V62_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vrounduwuh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vrounduhub,VI_ftype_VIVI,2)
// tag : V6_vrounduhub
def int_hexagon_V6_vrounduhub :
Hexagon_V62_v512v512v512_Intrinsic<"HEXAGON_V6_vrounduhub">;

//
// BUILTIN_INFO(HEXAGON.V6_vrounduhub_128B,VI_ftype_VIVI,2)
// tag : V6_vrounduhub_128B
def int_hexagon_V6_vrounduhub_128B :
Hexagon_V62_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vrounduhub_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vadduwsat,VI_ftype_VIVI,2)
// tag : V6_vadduwsat
def int_hexagon_V6_vadduwsat :
Hexagon_V62_v512v512v512_Intrinsic<"HEXAGON_V6_vadduwsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vadduwsat_128B,VI_ftype_VIVI,2)
// tag : V6_vadduwsat_128B
def int_hexagon_V6_vadduwsat_128B :
Hexagon_V62_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vadduwsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vadduwsat_dv,VD_ftype_VDVD,2)
// tag : V6_vadduwsat_dv
def int_hexagon_V6_vadduwsat_dv :
Hexagon_V62_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vadduwsat_dv">;

//
// BUILTIN_INFO(HEXAGON.V6_vadduwsat_dv_128B,VD_ftype_VDVD,2)
// tag : V6_vadduwsat_dv_128B
def int_hexagon_V6_vadduwsat_dv_128B :
Hexagon_V62_v2048v2048v2048_Intrinsic<"HEXAGON_V6_vadduwsat_dv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubuwsat,VI_ftype_VIVI,2)
// tag : V6_vsubuwsat
def int_hexagon_V6_vsubuwsat :
Hexagon_V62_v512v512v512_Intrinsic<"HEXAGON_V6_vsubuwsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubuwsat_128B,VI_ftype_VIVI,2)
// tag : V6_vsubuwsat_128B
def int_hexagon_V6_vsubuwsat_128B :
Hexagon_V62_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vsubuwsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubuwsat_dv,VD_ftype_VDVD,2)
// tag : V6_vsubuwsat_dv
def int_hexagon_V6_vsubuwsat_dv :
Hexagon_V62_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vsubuwsat_dv">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubuwsat_dv_128B,VD_ftype_VDVD,2)
// tag : V6_vsubuwsat_dv_128B
def int_hexagon_V6_vsubuwsat_dv_128B :
Hexagon_V62_v2048v2048v2048_Intrinsic<"HEXAGON_V6_vsubuwsat_dv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddbsat,VI_ftype_VIVI,2)
// tag : V6_vaddbsat
def int_hexagon_V6_vaddbsat :
Hexagon_V62_v512v512v512_Intrinsic<"HEXAGON_V6_vaddbsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddbsat_128B,VI_ftype_VIVI,2)
// tag : V6_vaddbsat_128B
def int_hexagon_V6_vaddbsat_128B :
Hexagon_V62_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vaddbsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddbsat_dv,VD_ftype_VDVD,2)
// tag : V6_vaddbsat_dv
def int_hexagon_V6_vaddbsat_dv :
Hexagon_V62_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vaddbsat_dv">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddbsat_dv_128B,VD_ftype_VDVD,2)
// tag : V6_vaddbsat_dv_128B
def int_hexagon_V6_vaddbsat_dv_128B :
Hexagon_V62_v2048v2048v2048_Intrinsic<"HEXAGON_V6_vaddbsat_dv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubbsat,VI_ftype_VIVI,2)
// tag : V6_vsubbsat
def int_hexagon_V6_vsubbsat :
Hexagon_V62_v512v512v512_Intrinsic<"HEXAGON_V6_vsubbsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubbsat_128B,VI_ftype_VIVI,2)
// tag : V6_vsubbsat_128B
def int_hexagon_V6_vsubbsat_128B :
Hexagon_V62_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vsubbsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubbsat_dv,VD_ftype_VDVD,2)
// tag : V6_vsubbsat_dv
def int_hexagon_V6_vsubbsat_dv :
Hexagon_V62_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vsubbsat_dv">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubbsat_dv_128B,VD_ftype_VDVD,2)
// tag : V6_vsubbsat_dv_128B
def int_hexagon_V6_vsubbsat_dv_128B :
Hexagon_V62_v2048v2048v2048_Intrinsic<"HEXAGON_V6_vsubbsat_dv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddububb_sat,VI_ftype_VIVI,2)
// tag : V6_vaddububb_sat
def int_hexagon_V6_vaddububb_sat :
Hexagon_V62_v512v512v512_Intrinsic<"HEXAGON_V6_vaddububb_sat">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddububb_sat_128B,VI_ftype_VIVI,2)
// tag : V6_vaddububb_sat_128B
def int_hexagon_V6_vaddububb_sat_128B :
Hexagon_V62_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vaddububb_sat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubububb_sat,VI_ftype_VIVI,2)
// tag : V6_vsubububb_sat
def int_hexagon_V6_vsubububb_sat :
Hexagon_V62_v512v512v512_Intrinsic<"HEXAGON_V6_vsubububb_sat">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubububb_sat_128B,VI_ftype_VIVI,2)
// tag : V6_vsubububb_sat_128B
def int_hexagon_V6_vsubububb_sat_128B :
Hexagon_V62_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vsubububb_sat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddhw_acc,VD_ftype_VDVIVI,3)
// tag : V6_vaddhw_acc
def int_hexagon_V6_vaddhw_acc :
Hexagon_V62_v1024v1024v512v512_Intrinsic<"HEXAGON_V6_vaddhw_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddhw_acc_128B,VD_ftype_VDVIVI,3)
// tag : V6_vaddhw_acc_128B
def int_hexagon_V6_vaddhw_acc_128B :
Hexagon_V62_v2048v2048v1024v1024_Intrinsic<"HEXAGON_V6_vaddhw_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vadduhw_acc,VD_ftype_VDVIVI,3)
// tag : V6_vadduhw_acc
def int_hexagon_V6_vadduhw_acc :
Hexagon_V62_v1024v1024v512v512_Intrinsic<"HEXAGON_V6_vadduhw_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vadduhw_acc_128B,VD_ftype_VDVIVI,3)
// tag : V6_vadduhw_acc_128B
def int_hexagon_V6_vadduhw_acc_128B :
Hexagon_V62_v2048v2048v1024v1024_Intrinsic<"HEXAGON_V6_vadduhw_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddubh_acc,VD_ftype_VDVIVI,3)
// tag : V6_vaddubh_acc
def int_hexagon_V6_vaddubh_acc :
Hexagon_V62_v1024v1024v512v512_Intrinsic<"HEXAGON_V6_vaddubh_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddubh_acc_128B,VD_ftype_VDVIVI,3)
// tag : V6_vaddubh_acc_128B
def int_hexagon_V6_vaddubh_acc_128B :
Hexagon_V62_v2048v2048v1024v1024_Intrinsic<"HEXAGON_V6_vaddubh_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyewuh_64,VD_ftype_VIVI,2)
// tag : V6_vmpyewuh_64
def int_hexagon_V6_vmpyewuh_64 :
Hexagon_V62_v1024v512v512_Intrinsic<"HEXAGON_V6_vmpyewuh_64">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyewuh_64_128B,VD_ftype_VIVI,2)
// tag : V6_vmpyewuh_64_128B
def int_hexagon_V6_vmpyewuh_64_128B :
Hexagon_V62_v2048v1024v1024_Intrinsic<"HEXAGON_V6_vmpyewuh_64_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyowh_64_acc,VD_ftype_VDVIVI,3)
// tag : V6_vmpyowh_64_acc
def int_hexagon_V6_vmpyowh_64_acc :
Hexagon_V62_v1024v1024v512v512_Intrinsic<"HEXAGON_V6_vmpyowh_64_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyowh_64_acc_128B,VD_ftype_VDVIVI,3)
// tag : V6_vmpyowh_64_acc_128B
def int_hexagon_V6_vmpyowh_64_acc_128B :
Hexagon_V62_v2048v2048v1024v1024_Intrinsic<"HEXAGON_V6_vmpyowh_64_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpauhb,VD_ftype_VDSI,2)
// tag : V6_vmpauhb
def int_hexagon_V6_vmpauhb :
Hexagon_V62_v1024v1024i_Intrinsic<"HEXAGON_V6_vmpauhb">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpauhb_128B,VD_ftype_VDSI,2)
// tag : V6_vmpauhb_128B
def int_hexagon_V6_vmpauhb_128B :
Hexagon_V62_v2048v2048i_Intrinsic<"HEXAGON_V6_vmpauhb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpauhb_acc,VD_ftype_VDVDSI,3)
// tag : V6_vmpauhb_acc
def int_hexagon_V6_vmpauhb_acc :
Hexagon_V62_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vmpauhb_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpauhb_acc_128B,VD_ftype_VDVDSI,3)
// tag : V6_vmpauhb_acc_128B
def int_hexagon_V6_vmpauhb_acc_128B :
Hexagon_V62_v2048v2048v2048i_Intrinsic<"HEXAGON_V6_vmpauhb_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyiwub,VI_ftype_VISI,2)
// tag : V6_vmpyiwub
def int_hexagon_V6_vmpyiwub :
Hexagon_V62_v512v512i_Intrinsic<"HEXAGON_V6_vmpyiwub">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyiwub_128B,VI_ftype_VISI,2)
// tag : V6_vmpyiwub_128B
def int_hexagon_V6_vmpyiwub_128B :
Hexagon_V62_v1024v1024i_Intrinsic<"HEXAGON_V6_vmpyiwub_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyiwub_acc,VI_ftype_VIVISI,3)
// tag : V6_vmpyiwub_acc
def int_hexagon_V6_vmpyiwub_acc :
Hexagon_V62_v512v512v512i_Intrinsic<"HEXAGON_V6_vmpyiwub_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyiwub_acc_128B,VI_ftype_VIVISI,3)
// tag : V6_vmpyiwub_acc_128B
def int_hexagon_V6_vmpyiwub_acc_128B :
Hexagon_V62_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vmpyiwub_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vandnqrt,VI_ftype_QVSI,2)
// tag : V6_vandnqrt
def int_hexagon_V6_vandnqrt :
Hexagon_V62_v512v64ii_Intrinsic<"HEXAGON_V6_vandnqrt">;

//
// BUILTIN_INFO(HEXAGON.V6_vandnqrt_128B,VI_ftype_QVSI,2)
// tag : V6_vandnqrt_128B
def int_hexagon_V6_vandnqrt_128B :
Hexagon_V62_v1024v128ii_Intrinsic<"HEXAGON_V6_vandnqrt_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vandnqrt_acc,VI_ftype_VIQVSI,3)
// tag : V6_vandnqrt_acc
def int_hexagon_V6_vandnqrt_acc :
Hexagon_V62_v512v512v64ii_Intrinsic<"HEXAGON_V6_vandnqrt_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vandnqrt_acc_128B,VI_ftype_VIQVSI,3)
// tag : V6_vandnqrt_acc_128B
def int_hexagon_V6_vandnqrt_acc_128B :
Hexagon_V62_v1024v1024v128ii_Intrinsic<"HEXAGON_V6_vandnqrt_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vandvqv,VI_ftype_QVVI,2)
// tag : V6_vandvqv
def int_hexagon_V6_vandvqv :
Hexagon_V62_v512v64iv512_Intrinsic<"HEXAGON_V6_vandvqv">;

//
// BUILTIN_INFO(HEXAGON.V6_vandvqv_128B,VI_ftype_QVVI,2)
// tag : V6_vandvqv_128B
def int_hexagon_V6_vandvqv_128B :
Hexagon_V62_v1024v128iv1024_Intrinsic<"HEXAGON_V6_vandvqv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vandvnqv,VI_ftype_QVVI,2)
// tag : V6_vandvnqv
def int_hexagon_V6_vandvnqv :
Hexagon_V62_v512v64iv512_Intrinsic<"HEXAGON_V6_vandvnqv">;

//
// BUILTIN_INFO(HEXAGON.V6_vandvnqv_128B,VI_ftype_QVVI,2)
// tag : V6_vandvnqv_128B
def int_hexagon_V6_vandvnqv_128B :
Hexagon_V62_v1024v128iv1024_Intrinsic<"HEXAGON_V6_vandvnqv_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_pred_scalar2v2,QV_ftype_SI,1)
// tag : V6_pred_scalar2v2
def int_hexagon_V6_pred_scalar2v2 :
Hexagon_V62_v64ii_Intrinsic<"HEXAGON_V6_pred_scalar2v2">;

//
// BUILTIN_INFO(HEXAGON.V6_pred_scalar2v2_128B,QV_ftype_SI,1)
// tag : V6_pred_scalar2v2_128B
def int_hexagon_V6_pred_scalar2v2_128B :
Hexagon_V62_v128ii_Intrinsic<"HEXAGON_V6_pred_scalar2v2_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_shuffeqw,QV_ftype_QVQV,2)
// tag : V6_shuffeqw
def int_hexagon_V6_shuffeqw :
Hexagon_V62_v64iv64iv64i_Intrinsic<"HEXAGON_V6_shuffeqw">;

//
// BUILTIN_INFO(HEXAGON.V6_shuffeqw_128B,QV_ftype_QVQV,2)
// tag : V6_shuffeqw_128B
def int_hexagon_V6_shuffeqw_128B :
Hexagon_V62_v128iv128iv128i_Intrinsic<"HEXAGON_V6_shuffeqw_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_shuffeqh,QV_ftype_QVQV,2)
// tag : V6_shuffeqh
def int_hexagon_V6_shuffeqh :
Hexagon_V62_v64iv64iv64i_Intrinsic<"HEXAGON_V6_shuffeqh">;

//
// BUILTIN_INFO(HEXAGON.V6_shuffeqh_128B,QV_ftype_QVQV,2)
// tag : V6_shuffeqh_128B
def int_hexagon_V6_shuffeqh_128B :
Hexagon_V62_v128iv128iv128i_Intrinsic<"HEXAGON_V6_shuffeqh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmaxb,VI_ftype_VIVI,2)
// tag : V6_vmaxb
def int_hexagon_V6_vmaxb :
Hexagon_V62_v512v512v512_Intrinsic<"HEXAGON_V6_vmaxb">;

//
// BUILTIN_INFO(HEXAGON.V6_vmaxb_128B,VI_ftype_VIVI,2)
// tag : V6_vmaxb_128B
def int_hexagon_V6_vmaxb_128B :
Hexagon_V62_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vmaxb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vminb,VI_ftype_VIVI,2)
// tag : V6_vminb
def int_hexagon_V6_vminb :
Hexagon_V62_v512v512v512_Intrinsic<"HEXAGON_V6_vminb">;

//
// BUILTIN_INFO(HEXAGON.V6_vminb_128B,VI_ftype_VIVI,2)
// tag : V6_vminb_128B
def int_hexagon_V6_vminb_128B :
Hexagon_V62_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vminb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsatuwuh,VI_ftype_VIVI,2)
// tag : V6_vsatuwuh
def int_hexagon_V6_vsatuwuh :
Hexagon_V62_v512v512v512_Intrinsic<"HEXAGON_V6_vsatuwuh">;

//
// BUILTIN_INFO(HEXAGON.V6_vsatuwuh_128B,VI_ftype_VIVI,2)
// tag : V6_vsatuwuh_128B
def int_hexagon_V6_vsatuwuh_128B :
Hexagon_V62_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vsatuwuh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_lvsplath,VI_ftype_SI,1)
// tag : V6_lvsplath
def int_hexagon_V6_lvsplath :
Hexagon_V62_v512i_Intrinsic<"HEXAGON_V6_lvsplath">;

//
// BUILTIN_INFO(HEXAGON.V6_lvsplath_128B,VI_ftype_SI,1)
// tag : V6_lvsplath_128B
def int_hexagon_V6_lvsplath_128B :
Hexagon_V62_v1024i_Intrinsic<"HEXAGON_V6_lvsplath_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_lvsplatb,VI_ftype_SI,1)
// tag : V6_lvsplatb
def int_hexagon_V6_lvsplatb :
Hexagon_V62_v512i_Intrinsic<"HEXAGON_V6_lvsplatb">;

//
// BUILTIN_INFO(HEXAGON.V6_lvsplatb_128B,VI_ftype_SI,1)
// tag : V6_lvsplatb_128B
def int_hexagon_V6_lvsplatb_128B :
Hexagon_V62_v1024i_Intrinsic<"HEXAGON_V6_lvsplatb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddclbw,VI_ftype_VIVI,2)
// tag : V6_vaddclbw
def int_hexagon_V6_vaddclbw :
Hexagon_V62_v512v512v512_Intrinsic<"HEXAGON_V6_vaddclbw">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddclbw_128B,VI_ftype_VIVI,2)
// tag : V6_vaddclbw_128B
def int_hexagon_V6_vaddclbw_128B :
Hexagon_V62_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vaddclbw_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddclbh,VI_ftype_VIVI,2)
// tag : V6_vaddclbh
def int_hexagon_V6_vaddclbh :
Hexagon_V62_v512v512v512_Intrinsic<"HEXAGON_V6_vaddclbh">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddclbh_128B,VI_ftype_VIVI,2)
// tag : V6_vaddclbh_128B
def int_hexagon_V6_vaddclbh_128B :
Hexagon_V62_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vaddclbh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vlutvvbi,VI_ftype_VIVISI,3)
// tag : V6_vlutvvbi
def int_hexagon_V6_vlutvvbi :
Hexagon_V62_v512v512v512i_Intrinsic<"HEXAGON_V6_vlutvvbi">;

//
// BUILTIN_INFO(HEXAGON.V6_vlutvvbi_128B,VI_ftype_VIVISI,3)
// tag : V6_vlutvvbi_128B
def int_hexagon_V6_vlutvvbi_128B :
Hexagon_V62_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vlutvvbi_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vlutvvb_oracci,VI_ftype_VIVIVISI,4)
// tag : V6_vlutvvb_oracci
def int_hexagon_V6_vlutvvb_oracci :
Hexagon_V62_v512v512v512v512i_Intrinsic<"HEXAGON_V6_vlutvvb_oracci">;

//
// BUILTIN_INFO(HEXAGON.V6_vlutvvb_oracci_128B,VI_ftype_VIVIVISI,4)
// tag : V6_vlutvvb_oracci_128B
def int_hexagon_V6_vlutvvb_oracci_128B :
Hexagon_V62_v1024v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vlutvvb_oracci_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vlutvwhi,VD_ftype_VIVISI,3)
// tag : V6_vlutvwhi
def int_hexagon_V6_vlutvwhi :
Hexagon_V62_v1024v512v512i_Intrinsic<"HEXAGON_V6_vlutvwhi">;

//
// BUILTIN_INFO(HEXAGON.V6_vlutvwhi_128B,VD_ftype_VIVISI,3)
// tag : V6_vlutvwhi_128B
def int_hexagon_V6_vlutvwhi_128B :
Hexagon_V62_v2048v1024v1024i_Intrinsic<"HEXAGON_V6_vlutvwhi_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vlutvwh_oracci,VD_ftype_VDVIVISI,4)
// tag : V6_vlutvwh_oracci
def int_hexagon_V6_vlutvwh_oracci :
Hexagon_V62_v1024v1024v512v512i_Intrinsic<"HEXAGON_V6_vlutvwh_oracci">;

//
// BUILTIN_INFO(HEXAGON.V6_vlutvwh_oracci_128B,VD_ftype_VDVIVISI,4)
// tag : V6_vlutvwh_oracci_128B
def int_hexagon_V6_vlutvwh_oracci_128B :
Hexagon_V62_v2048v2048v1024v1024i_Intrinsic<"HEXAGON_V6_vlutvwh_oracci_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vlutvvb_nm,VI_ftype_VIVISI,3)
// tag : V6_vlutvvb_nm
def int_hexagon_V6_vlutvvb_nm :
Hexagon_V62_v512v512v512i_Intrinsic<"HEXAGON_V6_vlutvvb_nm">;

//
// BUILTIN_INFO(HEXAGON.V6_vlutvvb_nm_128B,VI_ftype_VIVISI,3)
// tag : V6_vlutvvb_nm_128B
def int_hexagon_V6_vlutvvb_nm_128B :
Hexagon_V62_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vlutvvb_nm_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vlutvwh_nm,VD_ftype_VIVISI,3)
// tag : V6_vlutvwh_nm
def int_hexagon_V6_vlutvwh_nm :
Hexagon_V62_v1024v512v512i_Intrinsic<"HEXAGON_V6_vlutvwh_nm">;

//
// BUILTIN_INFO(HEXAGON.V6_vlutvwh_nm_128B,VD_ftype_VIVISI,3)
// tag : V6_vlutvwh_nm_128B
def int_hexagon_V6_vlutvwh_nm_128B :
Hexagon_V62_v2048v1024v1024i_Intrinsic<"HEXAGON_V6_vlutvwh_nm_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddcarry,VI_ftype_VIVIQV,3)
// tag: V6_vaddcarry
def int_hexagon_V6_vaddcarry :
Hexagon_v512v64iv512v512v64i_Intrinsic<"HEXAGON_v6_vaddcarry">;

//
// BUILTIN_INFO(HEXAGON.V6_vaddcarry_128B,VI_ftype_VIVIQV,3)
// tag: V6_vaddcarry_128B
def int_hexagon_V6_vaddcarry_128B :
Hexagon_v1024v128iv1024v1024v128i_Intrinsic<"HEXAGON_v6_vaddcarry_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubcarry,VI_ftype_VIVIQV,3)
// tag: V6_vsubcarry
def int_hexagon_V6_vsubcarry :
Hexagon_v512v64iv512v512v64i_Intrinsic<"HEXAGON_v6_vsubcarry">;

//
// BUILTIN_INFO(HEXAGON.V6_vsubcarry_128B,VI_ftype_VIVIQV,3)
// tag: V6_vsubcarry_128B
def int_hexagon_V6_vsubcarry_128B :
Hexagon_v1024v128iv1024v1024v128i_Intrinsic<"HEXAGON_v6_vsubcarry_128B">;


///
/// HexagonV65 intrinsics
///

//
// Hexagon_V65_iLLiLLi_Intrinsic<string GCCIntSuffix>
// tag : A6_vcmpbeq_notany
class Hexagon_V65_iLLiLLi_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i64_ty,llvm_i64_ty],
                          [IntrNoMem]>;

//
// Hexagon_V65_v1024v512LLi_Intrinsic<string GCCIntSuffix>
// tag : V6_vrmpyub_rtt
class Hexagon_V65_v1024v512LLi_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v16i32_ty,llvm_i64_ty],
                          [IntrNoMem]>;

//
// Hexagon_V65_v2048v1024LLi_Intrinsic<string GCCIntSuffix>
// tag : V6_vrmpyub_rtt_128B
class Hexagon_V65_v2048v1024LLi_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v32i32_ty,llvm_i64_ty],
                          [IntrNoMem]>;

//
// Hexagon_V65_v1024v1024v512LLi_Intrinsic<string GCCIntSuffix>
// tag : V6_vrmpyub_rtt_acc
class Hexagon_V65_v1024v1024v512LLi_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_v16i32_ty,llvm_i64_ty],
                          [IntrNoMem]>;

//
// Hexagon_V65_v2048v2048v1024LLi_Intrinsic<string GCCIntSuffix>
// tag : V6_vrmpyub_rtt_acc_128B
class Hexagon_V65_v2048v2048v1024LLi_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v64i32_ty,llvm_v32i32_ty,llvm_i64_ty],
                          [IntrNoMem]>;

//
// Hexagon_V65_v512v512v512i_Intrinsic<string GCCIntSuffix>
// tag : V6_vasruwuhsat
class Hexagon_V65_v512v512v512i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v16i32_ty,llvm_v16i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V65_v1024v1024v1024i_Intrinsic<string GCCIntSuffix>
// tag : V6_vasruwuhsat_128B
class Hexagon_V65_v1024v1024v1024i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_v32i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V65_v512v512v512_Intrinsic<string GCCIntSuffix>
// tag : V6_vavguw
class Hexagon_V65_v512v512v512_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v16i32_ty,llvm_v16i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V65_v1024v1024v1024_Intrinsic<string GCCIntSuffix>
// tag : V6_vavguw_128B
class Hexagon_V65_v1024v1024v1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_v32i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V65_v512v512_Intrinsic<string GCCIntSuffix>
// tag : V6_vabsb
class Hexagon_V65_v512v512_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v16i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V65_v1024v1024_Intrinsic<string GCCIntSuffix>
// tag : V6_vabsb_128B
class Hexagon_V65_v1024v1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V65_v1024v1024i_Intrinsic<string GCCIntSuffix>
// tag : V6_vmpabuu
class Hexagon_V65_v1024v1024i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V65_v2048v2048i_Intrinsic<string GCCIntSuffix>
// tag : V6_vmpabuu_128B
class Hexagon_V65_v2048v2048i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v64i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V65_v2048v2048v2048i_Intrinsic<string GCCIntSuffix>
// tag : V6_vmpabuu_acc_128B
class Hexagon_V65_v2048v2048v2048i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v64i32_ty,llvm_v64i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V65_v1024v1024v512i_Intrinsic<string GCCIntSuffix>
// tag : V6_vmpyh_acc
class Hexagon_V65_v1024v1024v512i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_v16i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V65_v2048v2048v1024i_Intrinsic<string GCCIntSuffix>
// tag : V6_vmpyh_acc_128B
class Hexagon_V65_v2048v2048v1024i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [llvm_v64i32_ty,llvm_v32i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V65_v512v512v512LLi_Intrinsic<string GCCIntSuffix>
// tag : V6_vmpahhsat
class Hexagon_V65_v512v512v512LLi_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v16i32_ty,llvm_v16i32_ty,llvm_i64_ty],
                          [IntrNoMem]>;

//
// Hexagon_V65_v1024v1024v1024LLi_Intrinsic<string GCCIntSuffix>
// tag : V6_vmpahhsat_128B
class Hexagon_V65_v1024v1024v1024LLi_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_v32i32_ty,llvm_i64_ty],
                          [IntrNoMem]>;

//
// Hexagon_V65_v512v512LLi_Intrinsic<string GCCIntSuffix>
// tag : V6_vlut4
class Hexagon_V65_v512v512LLi_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v16i32_ty,llvm_i64_ty],
                          [IntrNoMem]>;

//
// Hexagon_V65_v1024v1024LLi_Intrinsic<string GCCIntSuffix>
// tag : V6_vlut4_128B
class Hexagon_V65_v1024v1024LLi_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_i64_ty],
                          [IntrNoMem]>;

//
// Hexagon_V65_v512v512i_Intrinsic<string GCCIntSuffix>
// tag : V6_vmpyuhe
class Hexagon_V65_v512v512i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v16i32_ty,llvm_i32_ty],
                          [IntrNoMem]>;

//
// Hexagon_V65_v512v64i_Intrinsic<string GCCIntSuffix>
// tag : V6_vprefixqb
class Hexagon_V65_v512v64i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v16i32_ty], [llvm_v512i1_ty],
                          [IntrNoMem]>;

//
// Hexagon_V65_v1024v128i_Intrinsic<string GCCIntSuffix>
// tag : V6_vprefixqb_128B
class Hexagon_V65_v1024v128i_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v32i32_ty], [llvm_v1024i1_ty],
                          [IntrNoMem]>;

//
// BUILTIN_INFO(HEXAGON.A6_vcmpbeq_notany,QI_ftype_DIDI,2)
// tag : A6_vcmpbeq_notany
def int_hexagon_A6_vcmpbeq_notany :
Hexagon_V65_iLLiLLi_Intrinsic<"HEXAGON_A6_vcmpbeq_notany">;

//
// BUILTIN_INFO(HEXAGON.A6_vcmpbeq_notany_128B,QI_ftype_DIDI,2)
// tag : A6_vcmpbeq_notany_128B
def int_hexagon_A6_vcmpbeq_notany_128B :
Hexagon_V65_iLLiLLi_Intrinsic<"HEXAGON_A6_vcmpbeq_notany_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpyub_rtt,VD_ftype_VIDI,2)
// tag : V6_vrmpyub_rtt
def int_hexagon_V6_vrmpyub_rtt :
Hexagon_V65_v1024v512LLi_Intrinsic<"HEXAGON_V6_vrmpyub_rtt">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpyub_rtt_128B,VD_ftype_VIDI,2)
// tag : V6_vrmpyub_rtt_128B
def int_hexagon_V6_vrmpyub_rtt_128B :
Hexagon_V65_v2048v1024LLi_Intrinsic<"HEXAGON_V6_vrmpyub_rtt_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpyub_rtt_acc,VD_ftype_VDVIDI,3)
// tag : V6_vrmpyub_rtt_acc
def int_hexagon_V6_vrmpyub_rtt_acc :
Hexagon_V65_v1024v1024v512LLi_Intrinsic<"HEXAGON_V6_vrmpyub_rtt_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpyub_rtt_acc_128B,VD_ftype_VDVIDI,3)
// tag : V6_vrmpyub_rtt_acc_128B
def int_hexagon_V6_vrmpyub_rtt_acc_128B :
Hexagon_V65_v2048v2048v1024LLi_Intrinsic<"HEXAGON_V6_vrmpyub_rtt_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpybub_rtt,VD_ftype_VIDI,2)
// tag : V6_vrmpybub_rtt
def int_hexagon_V6_vrmpybub_rtt :
Hexagon_V65_v1024v512LLi_Intrinsic<"HEXAGON_V6_vrmpybub_rtt">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpybub_rtt_128B,VD_ftype_VIDI,2)
// tag : V6_vrmpybub_rtt_128B
def int_hexagon_V6_vrmpybub_rtt_128B :
Hexagon_V65_v2048v1024LLi_Intrinsic<"HEXAGON_V6_vrmpybub_rtt_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpybub_rtt_acc,VD_ftype_VDVIDI,3)
// tag : V6_vrmpybub_rtt_acc
def int_hexagon_V6_vrmpybub_rtt_acc :
Hexagon_V65_v1024v1024v512LLi_Intrinsic<"HEXAGON_V6_vrmpybub_rtt_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vrmpybub_rtt_acc_128B,VD_ftype_VDVIDI,3)
// tag : V6_vrmpybub_rtt_acc_128B
def int_hexagon_V6_vrmpybub_rtt_acc_128B :
Hexagon_V65_v2048v2048v1024LLi_Intrinsic<"HEXAGON_V6_vrmpybub_rtt_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vasruwuhsat,VI_ftype_VIVISI,3)
// tag : V6_vasruwuhsat
def int_hexagon_V6_vasruwuhsat :
Hexagon_V65_v512v512v512i_Intrinsic<"HEXAGON_V6_vasruwuhsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vasruwuhsat_128B,VI_ftype_VIVISI,3)
// tag : V6_vasruwuhsat_128B
def int_hexagon_V6_vasruwuhsat_128B :
Hexagon_V65_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vasruwuhsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vasruhubsat,VI_ftype_VIVISI,3)
// tag : V6_vasruhubsat
def int_hexagon_V6_vasruhubsat :
Hexagon_V65_v512v512v512i_Intrinsic<"HEXAGON_V6_vasruhubsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vasruhubsat_128B,VI_ftype_VIVISI,3)
// tag : V6_vasruhubsat_128B
def int_hexagon_V6_vasruhubsat_128B :
Hexagon_V65_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vasruhubsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vasruhubrndsat,VI_ftype_VIVISI,3)
// tag : V6_vasruhubrndsat
def int_hexagon_V6_vasruhubrndsat :
Hexagon_V65_v512v512v512i_Intrinsic<"HEXAGON_V6_vasruhubrndsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vasruhubrndsat_128B,VI_ftype_VIVISI,3)
// tag : V6_vasruhubrndsat_128B
def int_hexagon_V6_vasruhubrndsat_128B :
Hexagon_V65_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vasruhubrndsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vaslh_acc,VI_ftype_VIVISI,3)
// tag : V6_vaslh_acc
def int_hexagon_V6_vaslh_acc :
Hexagon_V65_v512v512v512i_Intrinsic<"HEXAGON_V6_vaslh_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vaslh_acc_128B,VI_ftype_VIVISI,3)
// tag : V6_vaslh_acc_128B
def int_hexagon_V6_vaslh_acc_128B :
Hexagon_V65_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vaslh_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrh_acc,VI_ftype_VIVISI,3)
// tag : V6_vasrh_acc
def int_hexagon_V6_vasrh_acc :
Hexagon_V65_v512v512v512i_Intrinsic<"HEXAGON_V6_vasrh_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vasrh_acc_128B,VI_ftype_VIVISI,3)
// tag : V6_vasrh_acc_128B
def int_hexagon_V6_vasrh_acc_128B :
Hexagon_V65_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vasrh_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vavguw,VI_ftype_VIVI,2)
// tag : V6_vavguw
def int_hexagon_V6_vavguw :
Hexagon_V65_v512v512v512_Intrinsic<"HEXAGON_V6_vavguw">;

//
// BUILTIN_INFO(HEXAGON.V6_vavguw_128B,VI_ftype_VIVI,2)
// tag : V6_vavguw_128B
def int_hexagon_V6_vavguw_128B :
Hexagon_V65_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vavguw_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vavguwrnd,VI_ftype_VIVI,2)
// tag : V6_vavguwrnd
def int_hexagon_V6_vavguwrnd :
Hexagon_V65_v512v512v512_Intrinsic<"HEXAGON_V6_vavguwrnd">;

//
// BUILTIN_INFO(HEXAGON.V6_vavguwrnd_128B,VI_ftype_VIVI,2)
// tag : V6_vavguwrnd_128B
def int_hexagon_V6_vavguwrnd_128B :
Hexagon_V65_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vavguwrnd_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vavgb,VI_ftype_VIVI,2)
// tag : V6_vavgb
def int_hexagon_V6_vavgb :
Hexagon_V65_v512v512v512_Intrinsic<"HEXAGON_V6_vavgb">;

//
// BUILTIN_INFO(HEXAGON.V6_vavgb_128B,VI_ftype_VIVI,2)
// tag : V6_vavgb_128B
def int_hexagon_V6_vavgb_128B :
Hexagon_V65_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vavgb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vavgbrnd,VI_ftype_VIVI,2)
// tag : V6_vavgbrnd
def int_hexagon_V6_vavgbrnd :
Hexagon_V65_v512v512v512_Intrinsic<"HEXAGON_V6_vavgbrnd">;

//
// BUILTIN_INFO(HEXAGON.V6_vavgbrnd_128B,VI_ftype_VIVI,2)
// tag : V6_vavgbrnd_128B
def int_hexagon_V6_vavgbrnd_128B :
Hexagon_V65_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vavgbrnd_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vnavgb,VI_ftype_VIVI,2)
// tag : V6_vnavgb
def int_hexagon_V6_vnavgb :
Hexagon_V65_v512v512v512_Intrinsic<"HEXAGON_V6_vnavgb">;

//
// BUILTIN_INFO(HEXAGON.V6_vnavgb_128B,VI_ftype_VIVI,2)
// tag : V6_vnavgb_128B
def int_hexagon_V6_vnavgb_128B :
Hexagon_V65_v1024v1024v1024_Intrinsic<"HEXAGON_V6_vnavgb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vabsb,VI_ftype_VI,1)
// tag : V6_vabsb
def int_hexagon_V6_vabsb :
Hexagon_V65_v512v512_Intrinsic<"HEXAGON_V6_vabsb">;

//
// BUILTIN_INFO(HEXAGON.V6_vabsb_128B,VI_ftype_VI,1)
// tag : V6_vabsb_128B
def int_hexagon_V6_vabsb_128B :
Hexagon_V65_v1024v1024_Intrinsic<"HEXAGON_V6_vabsb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vabsb_sat,VI_ftype_VI,1)
// tag : V6_vabsb_sat
def int_hexagon_V6_vabsb_sat :
Hexagon_V65_v512v512_Intrinsic<"HEXAGON_V6_vabsb_sat">;

//
// BUILTIN_INFO(HEXAGON.V6_vabsb_sat_128B,VI_ftype_VI,1)
// tag : V6_vabsb_sat_128B
def int_hexagon_V6_vabsb_sat_128B :
Hexagon_V65_v1024v1024_Intrinsic<"HEXAGON_V6_vabsb_sat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpabuu,VD_ftype_VDSI,2)
// tag : V6_vmpabuu
def int_hexagon_V6_vmpabuu :
Hexagon_V65_v1024v1024i_Intrinsic<"HEXAGON_V6_vmpabuu">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpabuu_128B,VD_ftype_VDSI,2)
// tag : V6_vmpabuu_128B
def int_hexagon_V6_vmpabuu_128B :
Hexagon_V65_v2048v2048i_Intrinsic<"HEXAGON_V6_vmpabuu_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpabuu_acc,VD_ftype_VDVDSI,3)
// tag : V6_vmpabuu_acc
def int_hexagon_V6_vmpabuu_acc :
Hexagon_V65_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vmpabuu_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpabuu_acc_128B,VD_ftype_VDVDSI,3)
// tag : V6_vmpabuu_acc_128B
def int_hexagon_V6_vmpabuu_acc_128B :
Hexagon_V65_v2048v2048v2048i_Intrinsic<"HEXAGON_V6_vmpabuu_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyh_acc,VD_ftype_VDVISI,3)
// tag : V6_vmpyh_acc
def int_hexagon_V6_vmpyh_acc :
Hexagon_V65_v1024v1024v512i_Intrinsic<"HEXAGON_V6_vmpyh_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyh_acc_128B,VD_ftype_VDVISI,3)
// tag : V6_vmpyh_acc_128B
def int_hexagon_V6_vmpyh_acc_128B :
Hexagon_V65_v2048v2048v1024i_Intrinsic<"HEXAGON_V6_vmpyh_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpahhsat,VI_ftype_VIVIDI,3)
// tag : V6_vmpahhsat
def int_hexagon_V6_vmpahhsat :
Hexagon_V65_v512v512v512LLi_Intrinsic<"HEXAGON_V6_vmpahhsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpahhsat_128B,VI_ftype_VIVIDI,3)
// tag : V6_vmpahhsat_128B
def int_hexagon_V6_vmpahhsat_128B :
Hexagon_V65_v1024v1024v1024LLi_Intrinsic<"HEXAGON_V6_vmpahhsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpauhuhsat,VI_ftype_VIVIDI,3)
// tag : V6_vmpauhuhsat
def int_hexagon_V6_vmpauhuhsat :
Hexagon_V65_v512v512v512LLi_Intrinsic<"HEXAGON_V6_vmpauhuhsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpauhuhsat_128B,VI_ftype_VIVIDI,3)
// tag : V6_vmpauhuhsat_128B
def int_hexagon_V6_vmpauhuhsat_128B :
Hexagon_V65_v1024v1024v1024LLi_Intrinsic<"HEXAGON_V6_vmpauhuhsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpsuhuhsat,VI_ftype_VIVIDI,3)
// tag : V6_vmpsuhuhsat
def int_hexagon_V6_vmpsuhuhsat :
Hexagon_V65_v512v512v512LLi_Intrinsic<"HEXAGON_V6_vmpsuhuhsat">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpsuhuhsat_128B,VI_ftype_VIVIDI,3)
// tag : V6_vmpsuhuhsat_128B
def int_hexagon_V6_vmpsuhuhsat_128B :
Hexagon_V65_v1024v1024v1024LLi_Intrinsic<"HEXAGON_V6_vmpsuhuhsat_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vlut4,VI_ftype_VIDI,2)
// tag : V6_vlut4
def int_hexagon_V6_vlut4 :
Hexagon_V65_v512v512LLi_Intrinsic<"HEXAGON_V6_vlut4">;

//
// BUILTIN_INFO(HEXAGON.V6_vlut4_128B,VI_ftype_VIDI,2)
// tag : V6_vlut4_128B
def int_hexagon_V6_vlut4_128B :
Hexagon_V65_v1024v1024LLi_Intrinsic<"HEXAGON_V6_vlut4_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyuhe,VI_ftype_VISI,2)
// tag : V6_vmpyuhe
def int_hexagon_V6_vmpyuhe :
Hexagon_V65_v512v512i_Intrinsic<"HEXAGON_V6_vmpyuhe">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyuhe_128B,VI_ftype_VISI,2)
// tag : V6_vmpyuhe_128B
def int_hexagon_V6_vmpyuhe_128B :
Hexagon_V65_v1024v1024i_Intrinsic<"HEXAGON_V6_vmpyuhe_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyuhe_acc,VI_ftype_VIVISI,3)
// tag : V6_vmpyuhe_acc
def int_hexagon_V6_vmpyuhe_acc :
Hexagon_V65_v512v512v512i_Intrinsic<"HEXAGON_V6_vmpyuhe_acc">;

//
// BUILTIN_INFO(HEXAGON.V6_vmpyuhe_acc_128B,VI_ftype_VIVISI,3)
// tag : V6_vmpyuhe_acc_128B
def int_hexagon_V6_vmpyuhe_acc_128B :
Hexagon_V65_v1024v1024v1024i_Intrinsic<"HEXAGON_V6_vmpyuhe_acc_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vprefixqb,VI_ftype_QV,1)
// tag : V6_vprefixqb
def int_hexagon_V6_vprefixqb :
Hexagon_V65_v512v64i_Intrinsic<"HEXAGON_V6_vprefixqb">;

//
// BUILTIN_INFO(HEXAGON.V6_vprefixqb_128B,VI_ftype_QV,1)
// tag : V6_vprefixqb_128B
def int_hexagon_V6_vprefixqb_128B :
Hexagon_V65_v1024v128i_Intrinsic<"HEXAGON_V6_vprefixqb_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vprefixqh,VI_ftype_QV,1)
// tag : V6_vprefixqh
def int_hexagon_V6_vprefixqh :
Hexagon_V65_v512v64i_Intrinsic<"HEXAGON_V6_vprefixqh">;

//
// BUILTIN_INFO(HEXAGON.V6_vprefixqh_128B,VI_ftype_QV,1)
// tag : V6_vprefixqh_128B
def int_hexagon_V6_vprefixqh_128B :
Hexagon_V65_v1024v128i_Intrinsic<"HEXAGON_V6_vprefixqh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vprefixqw,VI_ftype_QV,1)
// tag : V6_vprefixqw
def int_hexagon_V6_vprefixqw :
Hexagon_V65_v512v64i_Intrinsic<"HEXAGON_V6_vprefixqw">;

//
// BUILTIN_INFO(HEXAGON.V6_vprefixqw_128B,VI_ftype_QV,1)
// tag : V6_vprefixqw_128B
def int_hexagon_V6_vprefixqw_128B :
Hexagon_V65_v1024v128i_Intrinsic<"HEXAGON_V6_vprefixqw_128B">;


// The scatter/gather ones below will not be generated from iset.py. Make sure
// you don't overwrite these.
class Hexagon_V65_vvmemiiv512_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [], [llvm_ptr_ty,llvm_i32_ty,llvm_i32_ty,
                               llvm_v16i32_ty],
                          [IntrArgMemOnly]>;

class Hexagon_V65_vvmemiiv1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [], [llvm_ptr_ty,llvm_i32_ty,llvm_i32_ty,
                               llvm_v32i32_ty],
                          [IntrArgMemOnly]>;

class Hexagon_V65_vvmemiiv2048_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [], [llvm_ptr_ty,llvm_i32_ty,llvm_i32_ty,
                               llvm_v64i32_ty],
                          [IntrArgMemOnly]>;

class Hexagon_V65_vvmemv64iiiv512_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [], [llvm_ptr_ty,llvm_v512i1_ty,llvm_i32_ty,
                               llvm_i32_ty,llvm_v16i32_ty],
                          [IntrArgMemOnly]>;

class Hexagon_V65_vvmemv128iiiv1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [], [llvm_ptr_ty,llvm_v1024i1_ty,llvm_i32_ty,
                               llvm_i32_ty,llvm_v32i32_ty],
                          [IntrArgMemOnly]>;

class Hexagon_V65_vvmemv64iiiv1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [], [llvm_ptr_ty,llvm_v512i1_ty,llvm_i32_ty,
                               llvm_i32_ty,llvm_v32i32_ty],
                          [IntrArgMemOnly]>;

class Hexagon_V65_vvmemv128iiiv2048_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [], [llvm_ptr_ty,llvm_v1024i1_ty,llvm_i32_ty,
                               llvm_i32_ty,llvm_v64i32_ty],
                          [IntrArgMemOnly]>;

def int_hexagon_V6_vgathermw :
Hexagon_V65_vvmemiiv512_Intrinsic<"HEXAGON_V6_vgathermw">;

def int_hexagon_V6_vgathermw_128B :
Hexagon_V65_vvmemiiv1024_Intrinsic<"HEXAGON_V6_vgathermw_128B">;

def int_hexagon_V6_vgathermh :
Hexagon_V65_vvmemiiv512_Intrinsic<"HEXAGON_V6_vgathermh">;

def int_hexagon_V6_vgathermh_128B :
Hexagon_V65_vvmemiiv1024_Intrinsic<"HEXAGON_V6_vgathermh_128B">;

def int_hexagon_V6_vgathermhw :
Hexagon_V65_vvmemiiv1024_Intrinsic<"HEXAGON_V6_vgathermhw">;

def int_hexagon_V6_vgathermhw_128B :
Hexagon_V65_vvmemiiv2048_Intrinsic<"HEXAGON_V6_vgathermhw_128B">;

def int_hexagon_V6_vgathermwq :
Hexagon_V65_vvmemv64iiiv512_Intrinsic<"HEXAGON_V6_vgathermwq">;

def int_hexagon_V6_vgathermwq_128B :
Hexagon_V65_vvmemv128iiiv1024_Intrinsic<"HEXAGON_V6_vgathermwq_128B">;

def int_hexagon_V6_vgathermhq :
Hexagon_V65_vvmemv64iiiv512_Intrinsic<"HEXAGON_V6_vgathermhq">;

def int_hexagon_V6_vgathermhq_128B :
Hexagon_V65_vvmemv128iiiv1024_Intrinsic<"HEXAGON_V6_vgathermhq_128B">;

def int_hexagon_V6_vgathermhwq :
Hexagon_V65_vvmemv64iiiv1024_Intrinsic<"HEXAGON_V6_vgathermhwq">;

def int_hexagon_V6_vgathermhwq_128B :
Hexagon_V65_vvmemv128iiiv2048_Intrinsic<"HEXAGON_V6_vgathermhwq_128B">;

class Hexagon_V65_viiv512v512_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [], [llvm_i32_ty,llvm_i32_ty,
                                           llvm_v16i32_ty,llvm_v16i32_ty],
                          [IntrWriteMem]>;

class Hexagon_V65_viiv1024v1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [], [llvm_i32_ty,llvm_i32_ty,
                                           llvm_v32i32_ty,llvm_v32i32_ty],
                          [IntrWriteMem]>;

class Hexagon_V65_vv64iiiv512v512_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [], [llvm_v512i1_ty,llvm_i32_ty,
                                           llvm_i32_ty,llvm_v16i32_ty,
                                           llvm_v16i32_ty],
                          [IntrWriteMem]>;

class Hexagon_V65_vv128iiiv1024v1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [], [llvm_v1024i1_ty,llvm_i32_ty,
                                           llvm_i32_ty,llvm_v32i32_ty,
                                           llvm_v32i32_ty],
                          [IntrWriteMem]>;

class Hexagon_V65_viiv1024v512_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [], [llvm_i32_ty,llvm_i32_ty,
                                           llvm_v32i32_ty,llvm_v16i32_ty],
                          [IntrWriteMem]>;

class Hexagon_V65_viiv2048v1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [], [llvm_i32_ty,llvm_i32_ty,
                                           llvm_v64i32_ty,llvm_v32i32_ty],
                          [IntrWriteMem]>;

class Hexagon_V65_vv64iiiv1024v512_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [], [llvm_v512i1_ty,llvm_i32_ty,
                                           llvm_i32_ty,llvm_v32i32_ty,
                                           llvm_v16i32_ty],
                          [IntrWriteMem]>;

class Hexagon_V65_vv128iiiv2048v1024_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [], [llvm_v1024i1_ty,llvm_i32_ty,
                                           llvm_i32_ty,llvm_v64i32_ty,
                                           llvm_v32i32_ty],
                          [IntrWriteMem]>;

class Hexagon_V65_v2048_Intrinsic<string GCCIntSuffix>
 : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_v64i32_ty], [],
                          [IntrNoMem]>;

//
// BUILTIN_INFO(HEXAGON.V6_vscattermw,v_ftype_SISIVIVI,4)
// tag : V6_vscattermw
def int_hexagon_V6_vscattermw :
Hexagon_V65_viiv512v512_Intrinsic<"HEXAGON_V6_vscattermw">;

//
// BUILTIN_INFO(HEXAGON.V6_vscattermw_128B,v_ftype_SISIVIVI,4)
// tag : V6_vscattermw_128B
def int_hexagon_V6_vscattermw_128B :
Hexagon_V65_viiv1024v1024_Intrinsic<"HEXAGON_V6_vscattermw_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vscattermh,v_ftype_SISIVIVI,4)
// tag : V6_vscattermh
def int_hexagon_V6_vscattermh :
Hexagon_V65_viiv512v512_Intrinsic<"HEXAGON_V6_vscattermh">;

//
// BUILTIN_INFO(HEXAGON.V6_vscattermh_128B,v_ftype_SISIVIVI,4)
// tag : V6_vscattermh_128B
def int_hexagon_V6_vscattermh_128B :
Hexagon_V65_viiv1024v1024_Intrinsic<"HEXAGON_V6_vscattermh_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vscattermw_add,v_ftype_SISIVIVI,4)
// tag : V6_vscattermw_add
def int_hexagon_V6_vscattermw_add :
Hexagon_V65_viiv512v512_Intrinsic<"HEXAGON_V6_vscattermw_add">;

//
// BUILTIN_INFO(HEXAGON.V6_vscattermw_add_128B,v_ftype_SISIVIVI,4)
// tag : V6_vscattermw_add_128B
def int_hexagon_V6_vscattermw_add_128B :
Hexagon_V65_viiv1024v1024_Intrinsic<"HEXAGON_V6_vscattermw_add_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vscattermh_add,v_ftype_SISIVIVI,4)
// tag : V6_vscattermh_add
def int_hexagon_V6_vscattermh_add :
Hexagon_V65_viiv512v512_Intrinsic<"HEXAGON_V6_vscattermh_add">;

//
// BUILTIN_INFO(HEXAGON.V6_vscattermh_add_128B,v_ftype_SISIVIVI,4)
// tag : V6_vscattermh_add_128B
def int_hexagon_V6_vscattermh_add_128B :
Hexagon_V65_viiv1024v1024_Intrinsic<"HEXAGON_V6_vscattermh_add_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vscattermwq,v_ftype_QVSISIVIVI,5)
// tag : V6_vscattermwq
def int_hexagon_V6_vscattermwq :
Hexagon_V65_vv64iiiv512v512_Intrinsic<"HEXAGON_V6_vscattermwq">;

//
// BUILTIN_INFO(HEXAGON.V6_vscattermwq_128B,v_ftype_QVSISIVIVI,5)
// tag : V6_vscattermwq_128B
def int_hexagon_V6_vscattermwq_128B :
Hexagon_V65_vv128iiiv1024v1024_Intrinsic<"HEXAGON_V6_vscattermwq_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vscattermhq,v_ftype_QVSISIVIVI,5)
// tag : V6_vscattermhq
def int_hexagon_V6_vscattermhq :
Hexagon_V65_vv64iiiv512v512_Intrinsic<"HEXAGON_V6_vscattermhq">;

//
// BUILTIN_INFO(HEXAGON.V6_vscattermhq_128B,v_ftype_QVSISIVIVI,5)
// tag : V6_vscattermhq_128B
def int_hexagon_V6_vscattermhq_128B :
Hexagon_V65_vv128iiiv1024v1024_Intrinsic<"HEXAGON_V6_vscattermhq_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vscattermhw,v_ftype_SISIVDVI,4)
// tag : V6_vscattermhw
def int_hexagon_V6_vscattermhw :
Hexagon_V65_viiv1024v512_Intrinsic<"HEXAGON_V6_vscattermhw">;

//
// BUILTIN_INFO(HEXAGON.V6_vscattermhw_128B,v_ftype_SISIVDVI,4)
// tag : V6_vscattermhw_128B
def int_hexagon_V6_vscattermhw_128B :
Hexagon_V65_viiv2048v1024_Intrinsic<"HEXAGON_V6_vscattermhw_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vscattermhwq,v_ftype_QVSISIVDVI,5)
// tag : V6_vscattermhwq
def int_hexagon_V6_vscattermhwq :
Hexagon_V65_vv64iiiv1024v512_Intrinsic<"HEXAGON_V6_vscattermhwq">;

//
// BUILTIN_INFO(HEXAGON.V6_vscattermhwq_128B,v_ftype_QVSISIVDVI,5)
// tag : V6_vscattermhwq_128B
def int_hexagon_V6_vscattermhwq_128B :
Hexagon_V65_vv128iiiv2048v1024_Intrinsic<"HEXAGON_V6_vscattermhwq_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vscattermhw_add,v_ftype_SISIVDVI,4)
// tag : V6_vscattermhw_add
def int_hexagon_V6_vscattermhw_add :
Hexagon_V65_viiv1024v512_Intrinsic<"HEXAGON_V6_vscattermhw_add">;

//
// BUILTIN_INFO(HEXAGON.V6_vscattermhw_add_128B,v_ftype_SISIVDVI,4)
// tag : V6_vscattermhw_add_128B
def int_hexagon_V6_vscattermhw_add_128B :
Hexagon_V65_viiv2048v1024_Intrinsic<"HEXAGON_V6_vscattermhw_add_128B">;

//
// BUILTIN_INFO(HEXAGON.V6_vdd0,VD_ftype_,0)
// tag : V6_vdd0
def int_hexagon_V6_vdd0 :
Hexagon_v1024_Intrinsic<"HEXAGON_V6_vdd0">;

//
// BUILTIN_INFO(HEXAGON.V6_vdd0_128B,VD_ftype_,0)
// tag : V6_vdd0_128B
def int_hexagon_V6_vdd0_128B :
Hexagon_V65_v2048_Intrinsic<"HEXAGON_V6_vdd0_128B">;
