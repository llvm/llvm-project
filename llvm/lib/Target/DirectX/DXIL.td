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
include "llvm/IR/Attributes.td"

// Abstract representation of the class a DXIL Operation belongs to.
class DXILOpClass<string name> {
  string Name = name;
}

// Abstract representation of the category a DXIL Operation belongs to
class DXILOpCategory<string name> {
  string Name = name;
}

def UnaryClass : DXILOpClass<"Unary">;
def BinaryClass : DXILOpClass<"Binary">;
def FlattenedThreadIdInGroupClass : DXILOpClass<"FlattenedThreadIdInGroup">;
def ThreadIdInGroupClass : DXILOpClass<"ThreadIdInGroup">;
def ThreadIdClass : DXILOpClass<"ThreadId">;
def GroupIdClass : DXILOpClass<"GroupId">;

def BinaryUintCategory : DXILOpCategory<"Binary uint">;
def UnaryFloatCategory : DXILOpCategory<"Unary float">;
def ComputeIDCategory : DXILOpCategory<"Compute/Mesh/Amplification shader">;

// Represent as any pointer type with an option to change to a qualified pointer
// type with address space specified.
def dxil_handle_ty  : LLVMAnyPointerType;
def dxil_cbuffer_ty : LLVMAnyPointerType;
def dxil_resource_ty : LLVMAnyPointerType;

// The parameter description for a DXIL operation
class DXILOpParameter<int pos, LLVMType type, string name, string doc,
                 bit isConstant = 0, string enumName = "",
                 int maxValue = 0> {
  int Pos = pos;               // Position in parameter list
  LLVMType ParamType = type;   // Parameter type
  string Name = name;          // Short, unique parameter name
  string Doc = doc;            // Description of this parameter
  bit IsConstant = isConstant; // Whether this parameter requires a constant value in the IR
  string EnumName = enumName;  // Name of the enum type, if applicable
  int MaxValue = maxValue;     // Maximum value for this parameter, if applicable
}

// A representation for a DXIL operation
class DXILOperationDesc {
  string OpName = "";         // Name of DXIL operation
  int OpCode = 0;             // Unique non-negative integer associated with the operation
  DXILOpClass  OpClass;       // Class of the operation
  DXILOpCategory OpCategory;  // Category of the operation
  string Doc = "";            // Description of the operation
  list<DXILOpParameter> Params = []; // Parameter list of the operation
  list<LLVMType> OverloadTypes = [];  // Overload types, if applicable
  EnumAttr Attribute;         // Operation Attribute. Leverage attributes defined in Attributes.td
                              // ReadNone - operation does not access memory.
                              // ReadOnly - only reads from memory.
                              // "ReadMemory"   - reads memory
  bit IsDerivative = 0;       // Whether this is some kind of derivative
  bit IsGradient = 0;         // Whether this requires a gradient calculation
  bit IsFeedback = 0;         // Whether this is a sampler feedback operation
  bit IsWave = 0;             // Whether this requires in-wave, cross-lane functionality
  bit NeedsUniformInputs = 0; // Whether this operation requires that all
                              // of its inputs are uniform across the wave
  // Group DXIL operation for stats - e.g., to accumulate the number of atomic/float/uint/int/...
  // operations used in the program.
  list<string> StatsGroup = [];
}

class DXILOperation<string name, int opCode, DXILOpClass opClass, DXILOpCategory opCategory, string doc,
              list<LLVMType> oloadTypes, EnumAttr attrs, list<DXILOpParameter> params,
              list<string> statsGroup = []> : DXILOperationDesc {
  let OpName = name;
  let OpCode = opCode;
  let Doc = doc;
  let Params = params;
  let OpClass = opClass;
  let OpCategory = opCategory;
  let OverloadTypes = oloadTypes;
  let Attribute = attrs;
  let StatsGroup = statsGroup;
}

// LLVM intrinsic that DXIL operation maps to.
class LLVMIntrinsic<Intrinsic llvm_intrinsic_> { Intrinsic llvm_intrinsic = llvm_intrinsic_; }

def Sin : DXILOperation<"Sin", 13, UnaryClass, UnaryFloatCategory, "returns sine(theta) for theta in radians.",
  [llvm_half_ty, llvm_float_ty], ReadNone,
  [
    DXILOpParameter<0, llvm_anyfloat_ty, "", "operation result">,
    DXILOpParameter<1, llvm_i32_ty, "opcode", "DXIL opcode">,
    DXILOpParameter<2, llvm_anyfloat_ty, "value", "input value">
  ],
  ["floats"]>,
  LLVMIntrinsic<int_sin>;

def UMax : DXILOperation< "UMax", 39, BinaryClass, BinaryUintCategory, "unsigned integer maximum. UMax(a,b) = a > b ? a : b",
    [llvm_i16_ty, llvm_i32_ty, llvm_i64_ty], ReadNone,
  [
    DXILOpParameter<0, llvm_anyint_ty, "", "operation result">,
    DXILOpParameter<1, llvm_i32_ty, "opcode", "DXIL opcode">,
    DXILOpParameter<2, llvm_anyint_ty, "a", "input value">,
    DXILOpParameter<3, llvm_anyint_ty, "b", "input value">
  ],
  ["uints"]>,
  LLVMIntrinsic<int_umax>;

def ThreadId : DXILOperation< "ThreadId", 93, ThreadIdClass, ComputeIDCategory, "reads the thread ID", [llvm_i32_ty], ReadNone,
  [
    DXILOpParameter<0, llvm_i32_ty, "", "thread ID component">,
    DXILOpParameter<1, llvm_i32_ty, "opcode", "DXIL opcode">,
    DXILOpParameter<2, llvm_i32_ty, "component", "component to read (x,y,z)">
  ]>,
  LLVMIntrinsic<int_dx_thread_id>;

def GroupId : DXILOperation< "GroupId", 94, GroupIdClass, ComputeIDCategory, "reads the group ID (SV_GroupID)", [llvm_i32_ty], ReadNone,
  [
    DXILOpParameter<0, llvm_i32_ty, "", "group ID component">,
    DXILOpParameter<1, llvm_i32_ty, "opcode", "DXIL opcode">,
    DXILOpParameter<2, llvm_i32_ty, "component", "component to read">
  ]>,
  LLVMIntrinsic<int_dx_group_id>;

def ThreadIdInGroup : DXILOperation< "ThreadIdInGroup", 95, ThreadIdInGroupClass, ComputeIDCategory,
  "reads the thread ID within the group (SV_GroupThreadID)", [llvm_i32_ty], ReadNone,
  [
    DXILOpParameter<0, llvm_i32_ty, "", "thread ID in group component">,
    DXILOpParameter<1, llvm_i32_ty, "opcode", "DXIL opcode">,
    DXILOpParameter<2, llvm_i32_ty, "component", "component to read (x,y,z)">
  ]>,
  LLVMIntrinsic<int_dx_thread_id_in_group>;

def FlattenedThreadIdInGroup : DXILOperation< "FlattenedThreadIdInGroup", 96, FlattenedThreadIdInGroupClass, ComputeIDCategory,
   "provides a flattened index for a given thread within a given group (SV_GroupIndex)", [llvm_i32_ty], ReadNone,
  [
    DXILOpParameter<0, llvm_i32_ty, "", "result">,
    DXILOpParameter<1, llvm_i32_ty, "opcode", "DXIL opcode">
  ]>,
  LLVMIntrinsic<int_dx_flattened_thread_id_in_group>;
