//===- Intrinsics.td - Defines all LLVM intrinsics ---------*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines properties of all LLVM intrinsics.
//
//===----------------------------------------------------------------------===//

include "llvm/CodeGen/ValueTypes.td"

//===----------------------------------------------------------------------===//
//  Properties we keep track of for intrinsics.
//===----------------------------------------------------------------------===//

class IntrinsicProperty;

// Intr*Mem - Memory properties.  An intrinsic is allowed to have exactly one of
// these properties set.  They are listed from the most aggressive (best to use
// if correct) to the least aggressive.  If no property is set, the worst case 
// is assumed (IntrWriteMem).

// IntrNoMem - The intrinsic does not access memory or have any other side
// effects.  It may be CSE'd deleted if dead, etc.
def IntrNoMem : IntrinsicProperty;

// IntrReadArgMem - This intrinsic reads only from memory that one of its
// arguments points to, but may read an unspecified amount.
def IntrReadArgMem : IntrinsicProperty;

// IntrReadMem - This intrinsic reads from unspecified memory, so it cannot be
// moved across stores.  However, it can be reordered otherwise and can be
// deleted if dead.
def IntrReadMem : IntrinsicProperty;

// IntrWriteArgMem - This intrinsic reads and writes only from memory that one
// of its arguments points to, but may access an unspecified amount.  It has no
// other side effects.  This may only be used if the intrinsic doesn't "capture"
// the argument pointer (e.g. storing it someplace).
def IntrWriteArgMem : IntrinsicProperty;

// IntrWriteMem - This intrinsic may read or modify unspecified memory or has 
// other side effects.  It cannot be modified by the optimizer.  This is the
// default if the intrinsic has no other Intr*Mem property.
def IntrWriteMem : IntrinsicProperty;

// Commutative - This intrinsic is commutative: X op Y == Y op X.
def Commutative : IntrinsicProperty;

// NoCapture - The specified argument pointer is not captured by the intrinsic.
class NoCapture<int argNo> : IntrinsicProperty {
  int ArgNo = argNo;
}

//===----------------------------------------------------------------------===//
// Types used by intrinsics.
//===----------------------------------------------------------------------===//

class LLVMType<ValueType vt> {
  ValueType VT = vt;
}

class LLVMPointerType<LLVMType elty>
  : LLVMType<iPTR>{
  LLVMType ElTy = elty;
} 

class LLVMAnyPointerType<LLVMType elty>
  : LLVMType<iPTRAny>{
  LLVMType ElTy = elty;
} 

class LLVMMatchType<int num>
  : LLVMType<OtherVT>{
  int Number = num;
}

// Match the type of another intrinsic parameter that is expected to be 
// an integral vector type, but change the element size to be twice as wide
// or half as wide as the other type.  This is only useful when the intrinsic
// is overloaded, so the matched type should be declared as iAny.
class LLVMExtendedElementVectorType<int num> : LLVMMatchType<num>;
class LLVMTruncatedElementVectorType<int num> : LLVMMatchType<num>;

def llvm_void_ty       : LLVMType<isVoid>;
def llvm_anyint_ty     : LLVMType<iAny>;
def llvm_anyfloat_ty   : LLVMType<fAny>;
def llvm_i1_ty         : LLVMType<i1>;
def llvm_i8_ty         : LLVMType<i8>;
def llvm_i16_ty        : LLVMType<i16>;
def llvm_i32_ty        : LLVMType<i32>;
def llvm_i64_ty        : LLVMType<i64>;
def llvm_float_ty      : LLVMType<f32>;
def llvm_double_ty     : LLVMType<f64>;
def llvm_f80_ty        : LLVMType<f80>;
def llvm_f128_ty       : LLVMType<f128>;
def llvm_ppcf128_ty    : LLVMType<ppcf128>;
def llvm_ptr_ty        : LLVMPointerType<llvm_i8_ty>;             // i8*
def llvm_ptrptr_ty     : LLVMPointerType<llvm_ptr_ty>;            // i8**
def llvm_anyptr_ty     : LLVMAnyPointerType<llvm_i8_ty>;          // (space)i8*
def llvm_empty_ty      : LLVMType<OtherVT>;                       // { }
def llvm_descriptor_ty : LLVMPointerType<llvm_empty_ty>;          // { }*

def llvm_v16i8_ty      : LLVMType<v16i8>;    // 16 x i8
def llvm_v8i16_ty      : LLVMType<v8i16>;    //  8 x i16
def llvm_v2i64_ty      : LLVMType<v2i64>;    //  2 x i64
def llvm_v2i32_ty      : LLVMType<v2i32>;    //  2 x i32
def llvm_v1i64_ty      : LLVMType<v1i64>;    //  1 x i64
def llvm_v4i32_ty      : LLVMType<v4i32>;    //  4 x i32
def llvm_v4f32_ty      : LLVMType<v4f32>;    //  4 x float
def llvm_v2f64_ty      : LLVMType<v2f64>;    //  2 x double

// MMX Vector Types
def llvm_v8i8_ty       : LLVMType<v8i8>;     //  8 x i8
def llvm_v4i16_ty      : LLVMType<v4i16>;    //  4 x i16

def llvm_vararg_ty     : LLVMType<isVoid>;   // this means vararg here

//===----------------------------------------------------------------------===//
// Intrinsic Definitions.
//===----------------------------------------------------------------------===//

// Intrinsic class - This is used to define one LLVM intrinsic.  The name of the
// intrinsic definition should start with "int_", then match the LLVM intrinsic
// name with the "llvm." prefix removed, and all "."s turned into "_"s.  For
// example, llvm.bswap.i16 -> int_bswap_i16.
//
//  * RetTypes is a list containing the return types expected for the
//    intrinsic.
//  * ParamTypes is a list containing the parameter types expected for the
//    intrinsic.
//  * Properties can be set to describe the behavior of the intrinsic.
//
class Intrinsic<list<LLVMType> ret_types,
                list<LLVMType> param_types = [],
                list<IntrinsicProperty> properties = [],
                string name = ""> {
  string LLVMName = name;
  string TargetPrefix = "";   // Set to a prefix for target-specific intrinsics.
  list<LLVMType> RetTypes = ret_types;
  list<LLVMType> ParamTypes = param_types;
  list<IntrinsicProperty> Properties = properties;
}

/// GCCBuiltin - If this intrinsic exactly corresponds to a GCC builtin, this
/// specifies the name of the builtin.  This provides automatic CBE and CFE
/// support.
class GCCBuiltin<string name> {
  string GCCBuiltinName = name;
}


//===--------------- Variable Argument Handling Intrinsics ----------------===//
//  

def int_vastart : Intrinsic<[llvm_void_ty], [llvm_ptr_ty], [], "llvm.va_start">;
def int_vacopy  : Intrinsic<[llvm_void_ty], [llvm_ptr_ty, llvm_ptr_ty], [],
                            "llvm.va_copy">;
def int_vaend   : Intrinsic<[llvm_void_ty], [llvm_ptr_ty], [], "llvm.va_end">;

//===------------------- Garbage Collection Intrinsics --------------------===//
//  
def int_gcroot  : Intrinsic<[llvm_void_ty],
                            [llvm_ptrptr_ty, llvm_ptr_ty]>;
def int_gcread  : Intrinsic<[llvm_ptr_ty],
                            [llvm_ptr_ty, llvm_ptrptr_ty],
                            [IntrReadArgMem]>;
def int_gcwrite : Intrinsic<[llvm_void_ty],
                            [llvm_ptr_ty, llvm_ptr_ty, llvm_ptrptr_ty],
                            [IntrWriteArgMem]>;

//===--------------------- Code Generator Intrinsics ----------------------===//
//  
def int_returnaddress : Intrinsic<[llvm_ptr_ty], [llvm_i32_ty], [IntrNoMem]>;
def int_frameaddress  : Intrinsic<[llvm_ptr_ty], [llvm_i32_ty], [IntrNoMem]>;

// Note: we treat stacksave/stackrestore as writemem because we don't otherwise
// model their dependencies on allocas.
def int_stacksave     : Intrinsic<[llvm_ptr_ty]>,
                        GCCBuiltin<"__builtin_stack_save">;
def int_stackrestore  : Intrinsic<[llvm_void_ty], [llvm_ptr_ty]>,
                        GCCBuiltin<"__builtin_stack_restore">;

// IntrWriteArgMem is more pessimistic than strictly necessary for prefetch,
// however it does conveniently prevent the prefetch from being reordered
// with respect to nearby accesses to the same memory.
def int_prefetch      : Intrinsic<[llvm_void_ty],
                                  [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
                                  [IntrWriteArgMem]>;
def int_pcmarker      : Intrinsic<[llvm_void_ty], [llvm_i32_ty]>;

def int_readcyclecounter : Intrinsic<[llvm_i64_ty]>;

// Stack Protector Intrinsic - The stackprotector intrinsic writes the stack
// guard to the correct place on the stack frame.
def int_stackprotector : Intrinsic<[llvm_void_ty],
                                   [llvm_ptr_ty, llvm_ptrptr_ty],
                                   [IntrWriteMem]>;

//===------------------- Standard C Library Intrinsics --------------------===//
//

def int_memcpy  : Intrinsic<[llvm_void_ty],
                             [llvm_ptr_ty, llvm_ptr_ty, llvm_anyint_ty,
                              llvm_i32_ty],
                            [IntrWriteArgMem, NoCapture<0>, NoCapture<1>]>;
def int_memmove : Intrinsic<[llvm_void_ty],
                            [llvm_ptr_ty, llvm_ptr_ty, llvm_anyint_ty,
                             llvm_i32_ty],
                            [IntrWriteArgMem, NoCapture<0>, NoCapture<1>]>;
def int_memset  : Intrinsic<[llvm_void_ty],
                            [llvm_ptr_ty, llvm_i8_ty, llvm_anyint_ty,
                             llvm_i32_ty],
                            [IntrWriteArgMem, NoCapture<0>]>;

// These functions do not actually read memory, but they are sensitive to the
// rounding mode.  This needs to be modelled separately; in the meantime
// declaring them as reading memory is conservatively correct.
let Properties = [IntrReadMem] in {
  def int_sqrt : Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>]>;
  def int_powi : Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>, llvm_i32_ty]>;
  def int_sin  : Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>]>;
  def int_cos  : Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>]>; 
  def int_pow  : Intrinsic<[llvm_anyfloat_ty],
                           [LLVMMatchType<0>, LLVMMatchType<0>]>;
  def int_log  : Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>]>;
  def int_log10: Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>]>;
  def int_log2 : Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>]>;
  def int_exp  : Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>]>;
  def int_exp2 : Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>]>;
}

// NOTE: these are internal interfaces.
def int_setjmp     : Intrinsic<[llvm_i32_ty],  [llvm_ptr_ty]>;
def int_longjmp    : Intrinsic<[llvm_void_ty], [llvm_ptr_ty, llvm_i32_ty]>;
def int_sigsetjmp  : Intrinsic<[llvm_i32_ty] , [llvm_ptr_ty, llvm_i32_ty]>;
def int_siglongjmp : Intrinsic<[llvm_void_ty], [llvm_ptr_ty, llvm_i32_ty]>;

//===-------------------- Bit Manipulation Intrinsics ---------------------===//
//

// None of these intrinsics accesses memory at all.
let Properties = [IntrNoMem] in {
  def int_bswap: Intrinsic<[llvm_anyint_ty], [LLVMMatchType<0>]>;
  def int_ctpop: Intrinsic<[llvm_anyint_ty], [LLVMMatchType<0>]>;
  def int_ctlz : Intrinsic<[llvm_anyint_ty], [LLVMMatchType<0>]>;
  def int_cttz : Intrinsic<[llvm_anyint_ty], [LLVMMatchType<0>]>;
  def int_part_select : Intrinsic<[llvm_anyint_ty],
                                  [LLVMMatchType<0>, llvm_i32_ty, llvm_i32_ty]>;
  def int_part_set : Intrinsic<[llvm_anyint_ty],
                               [LLVMMatchType<0>, llvm_anyint_ty,
                                llvm_i32_ty, llvm_i32_ty]>;
}

//===------------------------ Debugger Intrinsics -------------------------===//
//

def int_dbg_stoppoint    : Intrinsic<[llvm_void_ty],
                                     [llvm_i32_ty, llvm_i32_ty, 
                                      llvm_descriptor_ty]>;
def int_dbg_region_start : Intrinsic<[llvm_void_ty], [llvm_descriptor_ty]>;
def int_dbg_region_end   : Intrinsic<[llvm_void_ty], [llvm_descriptor_ty]>;
def int_dbg_func_start   : Intrinsic<[llvm_void_ty], [llvm_descriptor_ty]>;
def int_dbg_declare      : Intrinsic<[llvm_void_ty],
                                     [llvm_descriptor_ty, llvm_descriptor_ty]>;

//===------------------ Exception Handling Intrinsics----------------------===//
//
def int_eh_exception    : Intrinsic<[llvm_ptr_ty]>;
def int_eh_selector_i32 : Intrinsic<[llvm_i32_ty],
                                    [llvm_ptr_ty, llvm_ptr_ty, llvm_vararg_ty]>;
def int_eh_selector_i64 : Intrinsic<[llvm_i64_ty],
                                    [llvm_ptr_ty, llvm_ptr_ty, llvm_vararg_ty]>;

def int_eh_typeid_for_i32 : Intrinsic<[llvm_i32_ty], [llvm_ptr_ty]>;
def int_eh_typeid_for_i64 : Intrinsic<[llvm_i64_ty], [llvm_ptr_ty]>;

def int_eh_return_i32 : Intrinsic<[llvm_void_ty], [llvm_i32_ty, llvm_ptr_ty]>;
def int_eh_return_i64 : Intrinsic<[llvm_void_ty], [llvm_i64_ty, llvm_ptr_ty]>;

def int_eh_unwind_init: Intrinsic<[llvm_void_ty]>,
                        GCCBuiltin<"__builtin_unwind_init">;

def int_eh_dwarf_cfa  : Intrinsic<[llvm_ptr_ty], [llvm_i32_ty]>;

//===---------------- Generic Variable Attribute Intrinsics----------------===//
//
def int_var_annotation : Intrinsic<[llvm_void_ty],
                                   [llvm_ptr_ty, llvm_ptr_ty,
                                    llvm_ptr_ty, llvm_i32_ty], 
                                   [], "llvm.var.annotation">;
def int_ptr_annotation : Intrinsic<[LLVMAnyPointerType<llvm_anyint_ty>],
                                   [LLVMMatchType<0>, llvm_ptr_ty, llvm_ptr_ty,
                                    llvm_i32_ty], 
                                   [], "llvm.ptr.annotation">;
def int_annotation : Intrinsic<[llvm_anyint_ty],
                               [LLVMMatchType<0>, llvm_ptr_ty,
                                llvm_ptr_ty, llvm_i32_ty],
                               [], "llvm.annotation">;

//===------------------------ Trampoline Intrinsics -----------------------===//
//
def int_init_trampoline : Intrinsic<[llvm_ptr_ty],
                                    [llvm_ptr_ty, llvm_ptr_ty, llvm_ptr_ty],
                                    []>,
                          GCCBuiltin<"__builtin_init_trampoline">;

//===------------------------ Overflow Intrinsics -------------------------===//
//

// Expose the carry flag from add operations on two integrals.
def int_sadd_with_overflow : Intrinsic<[llvm_anyint_ty, llvm_i1_ty],
                                       [LLVMMatchType<0>, LLVMMatchType<0>]>;
def int_uadd_with_overflow : Intrinsic<[llvm_anyint_ty, llvm_i1_ty],
                                       [LLVMMatchType<0>, LLVMMatchType<0>]>;

def int_ssub_with_overflow : Intrinsic<[llvm_anyint_ty, llvm_i1_ty],
                                       [LLVMMatchType<0>, LLVMMatchType<0>]>;
def int_usub_with_overflow : Intrinsic<[llvm_anyint_ty, llvm_i1_ty],
                                       [LLVMMatchType<0>, LLVMMatchType<0>]>;

def int_smul_with_overflow : Intrinsic<[llvm_anyint_ty, llvm_i1_ty],
                                       [LLVMMatchType<0>, LLVMMatchType<0>]>;
def int_umul_with_overflow : Intrinsic<[llvm_anyint_ty, llvm_i1_ty],
                                       [LLVMMatchType<0>, LLVMMatchType<0>]>;

//===------------------------- Atomic Intrinsics --------------------------===//
//
def int_memory_barrier : Intrinsic<[llvm_void_ty],
                                   [llvm_i1_ty, llvm_i1_ty,
                                    llvm_i1_ty, llvm_i1_ty, llvm_i1_ty], []>,
                                    GCCBuiltin<"__builtin_llvm_memory_barrier">;

def int_atomic_cmp_swap : Intrinsic<[llvm_anyint_ty],
                                    [LLVMAnyPointerType<LLVMMatchType<0>>,
                                     LLVMMatchType<0>, LLVMMatchType<0>],
                                    [IntrWriteArgMem]>,
                           GCCBuiltin<"__sync_val_compare_and_swap">;
def int_atomic_load_add : Intrinsic<[llvm_anyint_ty],
                                    [LLVMAnyPointerType<LLVMMatchType<0>>,
                                     LLVMMatchType<0>],
                                    [IntrWriteArgMem]>,
                           GCCBuiltin<"__sync_fetch_and_add">;
def int_atomic_swap     : Intrinsic<[llvm_anyint_ty],
                                    [LLVMAnyPointerType<LLVMMatchType<0>>,
                                     LLVMMatchType<0>],
                                    [IntrWriteArgMem]>,
                           GCCBuiltin<"__sync_lock_test_and_set">;
def int_atomic_load_sub : Intrinsic<[llvm_anyint_ty],
                                    [LLVMAnyPointerType<LLVMMatchType<0>>,
                                     LLVMMatchType<0>],
                                    [IntrWriteArgMem]>,
                           GCCBuiltin<"__sync_fetch_and_sub">;
def int_atomic_load_and : Intrinsic<[llvm_anyint_ty],
                                    [LLVMAnyPointerType<LLVMMatchType<0>>,
                                     LLVMMatchType<0>],
                                    [IntrWriteArgMem]>,
                           GCCBuiltin<"__sync_fetch_and_and">;
def int_atomic_load_or   : Intrinsic<[llvm_anyint_ty],
                                     [LLVMAnyPointerType<LLVMMatchType<0>>,
                                      LLVMMatchType<0>],
                                     [IntrWriteArgMem]>,
                           GCCBuiltin<"__sync_fetch_and_or">;
def int_atomic_load_xor : Intrinsic<[llvm_anyint_ty],
                                    [LLVMAnyPointerType<LLVMMatchType<0>>,
                                     LLVMMatchType<0>],
                                    [IntrWriteArgMem]>,
                           GCCBuiltin<"__sync_fetch_and_xor">;
def int_atomic_load_nand : Intrinsic<[llvm_anyint_ty],
                                     [LLVMAnyPointerType<LLVMMatchType<0>>,
                                      LLVMMatchType<0>],
                                     [IntrWriteArgMem]>,
                           GCCBuiltin<"__sync_fetch_and_nand">;
def int_atomic_load_min  : Intrinsic<[llvm_anyint_ty],
                                     [LLVMAnyPointerType<LLVMMatchType<0>>,
                                      LLVMMatchType<0>],
                                     [IntrWriteArgMem]>,
                           GCCBuiltin<"__sync_fetch_and_min">;
def int_atomic_load_max  : Intrinsic<[llvm_anyint_ty],
                                     [LLVMAnyPointerType<LLVMMatchType<0>>,
                                      LLVMMatchType<0>],
                                     [IntrWriteArgMem]>,
                           GCCBuiltin<"__sync_fetch_and_max">;
def int_atomic_load_umin : Intrinsic<[llvm_anyint_ty],
                                     [LLVMAnyPointerType<LLVMMatchType<0>>,
                                      LLVMMatchType<0>],
                                     [IntrWriteArgMem]>,
                           GCCBuiltin<"__sync_fetch_and_umin">;
def int_atomic_load_umax : Intrinsic<[llvm_anyint_ty],
                                     [LLVMAnyPointerType<LLVMMatchType<0>>,
                                      LLVMMatchType<0>],
                                     [IntrWriteArgMem]>,
                           GCCBuiltin<"__sync_fetch_and_umax">;
                                  
//===-------------------------- Other Intrinsics --------------------------===//
//
def int_flt_rounds : Intrinsic<[llvm_i32_ty]>,
                     GCCBuiltin<"__builtin_flt_rounds">;
def int_trap : Intrinsic<[llvm_void_ty]>,
               GCCBuiltin<"__builtin_trap">;

// These convert intrinsics are to support various conversions between
// various types with rounding and saturation. NOTE: avoid using these
// intrinsics as they might be removed sometime in the future and
// most targets don't support them.
def int_convertff  : Intrinsic<[llvm_anyfloat_ty],
                               [llvm_anyfloat_ty, llvm_i32_ty, llvm_i32_ty]>;
def int_convertfsi : Intrinsic<[llvm_anyfloat_ty],
                               [llvm_anyint_ty, llvm_i32_ty, llvm_i32_ty]>;
def int_convertfui : Intrinsic<[llvm_anyfloat_ty],
                               [llvm_anyint_ty, llvm_i32_ty, llvm_i32_ty]>;
def int_convertsif : Intrinsic<[llvm_anyint_ty],
                               [llvm_anyfloat_ty, llvm_i32_ty, llvm_i32_ty]>;
def int_convertuif : Intrinsic<[llvm_anyint_ty],
                               [llvm_anyfloat_ty, llvm_i32_ty, llvm_i32_ty]>;
def int_convertss  : Intrinsic<[llvm_anyint_ty],
                               [llvm_anyint_ty, llvm_i32_ty, llvm_i32_ty]>;
def int_convertsu  : Intrinsic<[llvm_anyint_ty],
                               [llvm_anyint_ty, llvm_i32_ty, llvm_i32_ty]>;
def int_convertus  : Intrinsic<[llvm_anyint_ty],
                               [llvm_anyint_ty, llvm_i32_ty, llvm_i32_ty]>;
def int_convertuu  : Intrinsic<[llvm_anyint_ty],
                               [llvm_anyint_ty, llvm_i32_ty, llvm_i32_ty]>;

//===----------------------------------------------------------------------===//
// Target-specific intrinsics
//===----------------------------------------------------------------------===//

include "llvm/IntrinsicsPowerPC.td"
include "llvm/IntrinsicsX86.td"
include "llvm/IntrinsicsARM.td"
include "llvm/IntrinsicsCellSPU.td"
include "llvm/IntrinsicsAlpha.td"
include "llvm/IntrinsicsXCore.td"
