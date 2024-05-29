//- DXIL.td - Describe DXIL operation -------------------------*- tablegen -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This is a target description file for DXIL operations.
///
//===----------------------------------------------------------------------===//

include "llvm/IR/Intrinsics.td"

// Abstraction of DXIL Operation class.
// It encapsulates an associated function signature viz.,
// returnTy(param1Ty, param2Ty, ...) represented as a list of LLVMTypes.
// DXIL Ops that belong to a DXILOpClass record the signature of that
// DXILOpClass

class DXILOpClass<list<LLVMType> OpSig> {
  list<LLVMType> OpSignature = OpSig;
}

// Concrete definitions of DXIL Op Classes
// Refer to the design document
// (https://github.com/llvm/llvm-project/blob/main/llvm/docs/DirectX/DXILOpTableGenDesign.rst)
// for details about the use of DXIL Op Class name.

// NOTE: The following list is not complete. Classes need to be defined as new DXIL Ops
// are added.
defset list<DXILOpClass> OpClasses = {
  def acceptHitAndEndSearch : DXILOpClass<[llvm_void_ty]>;
  def allocateRayQuery : DXILOpClass<[llvm_i32_ty, llvm_i32_ty]>;
  def attributeAtVertex : DXILOpClass<[llvm_any_ty, llvm_i32_ty, llvm_i32_ty, llvm_i8_ty, llvm_i8_ty]>;
  def barrier : DXILOpClass<[llvm_void_ty, llvm_i32_ty]>;
  def barrierByMemoryType : DXILOpClass<[llvm_void_ty, llvm_i32_ty, llvm_i32_ty]>;
  def binary : DXILOpClass<[llvm_any_ty, LLVMMatchType<0>, LLVMMatchType<0>]>;
  def binaryWithCarryOrBorrow : DXILOpClass<[llvm_i32_ty, llvm_any_ty, LLVMMatchType<0>]>;
  def dot2 : DXILOpClass<!listsplat(llvm_anyfloat_ty, 5)>;
  def dot3 : DXILOpClass<!listsplat(llvm_anyfloat_ty, 7)>;
  def dot4 : DXILOpClass<!listsplat(llvm_anyfloat_ty, 9)>;
  def flattenedThreadIdInGroup : DXILOpClass<[llvm_i32_ty]>;
  def groupId : DXILOpClass<[llvm_i32_ty, llvm_i32_ty]>;
  def isSpecialFloat : DXILOpClass<[llvm_i1_ty, llvm_anyfloat_ty]>;
  def tertiary : DXILOpClass<[llvm_any_ty, LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>]>;
  def threadId : DXILOpClass<[llvm_i32_ty, llvm_i32_ty]>;
  def threadIdInGroup : DXILOpClass<[llvm_i32_ty, llvm_i32_ty]>;
  def unary : DXILOpClass<[llvm_any_ty, LLVMMatchType<0>]>;

  // This is a sentinel definition. Hence placed at the end of the list
  // and not as part of the above alphabetically sorted valid definitions.
  // Additionally it is capitalized unlike all the others.
  def UnknownOpClass: DXILOpClass<[]>;
}

// Abstraction DXIL Operation
class DXILOpPropertiesBase {
  int OpCode = 0;                      // Opcode of DXIL Operation
  DXILOpClass OpClass = UnknownOpClass;// Class of DXIL Operation.
  Intrinsic LLVMIntrinsic = ?;         // LLVM Intrinsic DXIL Operation maps to
  list<LLVMType> OpOverloadTypes = ?; // Valid overload type
                                       // of DXIL Operation
  string Doc = "";                     // A short description of the operation
}

class DXILOpProperties<int opCode,
                    Intrinsic intrinsic,
                    list<LLVMType> overloadTypes,
                    string doc> : DXILOpPropertiesBase {
  int OpCode = opCode;
  Intrinsic LLVMIntrinsic = intrinsic;
  list<LLVMType> OpOverloadTypes = overloadTypes;
  string Doc = doc;
}

// Concrete definitions of DXIL Operation Properties to corresponding LLVM intrinsic

// IsSpecialFloat Class
let OpClass = isSpecialFloat in {
  def IsInf : DXILOpProperties<9,  int_dx_isinf, [llvm_half_ty, llvm_float_ty],
                           "Determines if the specified value is infinite.">;
}

let OpClass = unary in {
  def Abs : DXILOpProperties<6, int_fabs, [llvm_half_ty, llvm_float_ty, llvm_double_ty],
                          "Returns the absolute value of the input.">;

  def Cos  : DXILOpProperties<12, int_cos, [llvm_half_ty, llvm_float_ty],
                          "Returns cosine(theta) for theta in radians.">;
  def Sin  : DXILOpProperties<13, int_sin, [llvm_half_ty, llvm_float_ty],
                           "Returns sine(theta) for theta in radians.">;
  def Tan  : DXILOpProperties<14, int_tan, [llvm_half_ty, llvm_float_ty],
                           "Returns tangent(theta) for theta in radians.">;
  def Exp2 : DXILOpProperties<21, int_exp2, [llvm_half_ty, llvm_float_ty],
                           "Returns the base 2 exponential, or 2**x, of the"
                           " specified value. exp2(x) = 2**x.">;
  def Frac : DXILOpProperties<22, int_dx_frac, [llvm_half_ty, llvm_float_ty],
                            "Returns a fraction from 0 to 1 that represents the"
                            " decimal part of the input.">;
  def Log2 : DXILOpProperties<23, int_log2, [llvm_half_ty, llvm_float_ty],
                           "Returns the base-2 logarithm of the specified value.">;
  def Sqrt : DXILOpProperties<24, int_sqrt, [llvm_half_ty, llvm_float_ty],
                           "Returns the square root of the specified floating-point"
                           "value, per component.">;
  def RSqrt : DXILOpProperties<25, int_dx_rsqrt, [llvm_half_ty, llvm_float_ty],
                            "Returns the reciprocal of the square root of the"
                            " specified value. rsqrt(x) = 1 / sqrt(x).">;
  def Round : DXILOpProperties<26, int_roundeven, [llvm_half_ty, llvm_float_ty],
                            "Returns the input rounded to the nearest integer"
                            "within a floating-point type.">;
  def Floor : DXILOpProperties<27, int_floor, [llvm_half_ty, llvm_float_ty],
                            "Returns the largest integer that is less than or equal to the input.">;
  def Ceil  : DXILOpProperties<28, int_ceil, [llvm_half_ty, llvm_float_ty],
                            "Returns the smallest integer that is greater than or equal to the input.">;
  def Trunc : DXILOpProperties<29, int_trunc, [llvm_half_ty, llvm_float_ty],
                            "Returns the specified value truncated to the integer component.">;
  def Rbits : DXILOpProperties<30, int_bitreverse, [llvm_i16_ty, llvm_i32_ty, llvm_i64_ty],
                            "Returns the specified value with its bits reversed.">;
}

let OpClass = binary in {
// Float overloads
  def FMax : DXILOpProperties<35, int_maxnum, [llvm_half_ty, llvm_float_ty, llvm_double_ty],
                           "Float maximum. FMax(a,b) = a > b ? a : b">;
  def FMin : DXILOpProperties<36, int_minnum, [llvm_half_ty, llvm_float_ty, llvm_double_ty],
                           "Float minimum. FMin(a,b) = a < b ? a : b">;
// Int overloads
  def SMax : DXILOpProperties<37, int_smax, [llvm_i16_ty, llvm_i32_ty, llvm_i64_ty],
                           "Signed integer maximum. SMax(a,b) = a > b ? a : b">;
  def SMin : DXILOpProperties<38, int_smin, [llvm_i16_ty, llvm_i32_ty, llvm_i64_ty],
                           "Signed integer minimum. SMin(a,b) = a < b ? a : b">;
  def UMax : DXILOpProperties<39, int_umax, [llvm_i16_ty, llvm_i32_ty, llvm_i64_ty],
                           "Unsigned integer maximum. UMax(a,b) = a > b ? a : b">;
  def UMin : DXILOpProperties<40, int_umin, [llvm_i16_ty, llvm_i32_ty, llvm_i64_ty],
                           "Unsigned integer minimum. UMin(a,b) = a < b ? a : b">;
}

let OpClass = tertiary in {
  def FMad : DXILOpProperties<46, int_fmuladd, [llvm_half_ty, llvm_float_ty, llvm_double_ty],
                            "Floating point arithmetic multiply/add operation."
                            " fmad(m,a,b) = m * a + b.">;
// Int overloads
def IMad : DXILOpProperties<48, int_dx_imad, [llvm_i16_ty, llvm_i32_ty, llvm_i64_ty],
                         "Signed integer arithmetic multiply/add operation."
                          " imad(m,a,b) = m * a + b.">;
def UMad : DXILOpProperties<49, int_dx_umad, [llvm_i16_ty, llvm_i32_ty, llvm_i64_ty],
                        "Unsigned integer arithmetic multiply/add operation."
                        " umad(m,a, = m * a + b.">;
}

let OpClass = dot2 in
  def Dot2 : DXILOpProperties<54, int_dx_dot2, [llvm_half_ty, llvm_float_ty],
                          "dot product of two float vectors Dot(a,b) = a[0]*b[0] +"
                          " ... + a[n]*b[n] where n is between 0 and 1">;
let OpClass = dot3 in
  def Dot3 : DXILOpProperties<55, int_dx_dot3, [llvm_half_ty, llvm_float_ty],
                           "dot product of two float vectors Dot(a,b) = a[0]*b[0] +"
                           " ... + a[n]*b[n] where n is between 0 and 2">;
let OpClass = dot4 in
   def Dot4 : DXILOpProperties<56, int_dx_dot4, [llvm_half_ty, llvm_float_ty],
                            "dot product of two float vectors Dot(a,b) = a[0]*b[0] +"
                            " ... + a[n]*b[n] where n is between 0 and 3">;
let OpClass =  threadId in
  def ThreadId : DXILOpProperties<93, int_dx_thread_id, [llvm_i32_ty],
                              "Reads the thread ID">;
let OpClass =  groupId in
  def GroupId  : DXILOpProperties<94, int_dx_group_id, [llvm_i32_ty],
                               "Reads the group ID (SV_GroupID)">;
let OpClass =  threadIdInGroup in
  def ThreadIdInGroup : DXILOpProperties<95, int_dx_thread_id_in_group, [llvm_i32_ty],
                                     "Reads the thread ID within the group "
                                     "(SV_GroupThreadID)">;
let OpClass = flattenedThreadIdInGroup in
  def FlattenedThreadIdInGroup : DXILOpProperties<96, int_dx_flattened_thread_id_in_group,
                                               [llvm_i32_ty],
                                                "Provides a flattened index for a given"
                                                " thread within a given group (SV_GroupIndex)">;
