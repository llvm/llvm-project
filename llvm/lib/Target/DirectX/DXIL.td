//- DXIL.td - Describe DXIL operation -------------------------*- tablegen -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This is a target description file for DXIL operation.
///
//===----------------------------------------------------------------------===//

include "llvm/IR/Intrinsics.td"

// Abstract DXIL Op Class representation
class DXILClass<string _name> {
  string name = _name;
}

// Abstract DXIL Op Category representation
class DXILCategory<string _name> {
  string name = _name;
}

// Abstract Type representation
class Type<string _desc> {
  string desc = _desc;
}

// Basic Types
def float_t : Type<"float">;
def rfloat_t : Type<"float with rounding">;
def int_t : Type<"int">;
def uint_t : Type<"unit">;
def uint_cb_t : Type<"uint with carry or borrow">;
def uint_two_outs_t : Type<"uint with two outputs">;

// DXIL Op Classes
def unary : DXILClass<"Unary">;
def binary : DXILClass<"Binary">;
def flattened_threadIdIn_group : DXILClass<"FlattenedThreadIdInGroup">;
def threadId_in_group : DXILClass<"ThreadIdInGroup">;
def thread_id : DXILClass<"ThreadId">;
def group_id : DXILClass<"GroupId">;

// DXIL Op Categories
def binary_uint : DXILCategory<"Binary uint">;
def unary_float : DXILCategory<"Unary float">;
def compute_id : DXILCategory<"Compute/Mesh/Amplification shader">;


// The parameter description for a DXIL instruction
class dxil_param<int _pos, string type, string _name, string _doc,
                 bit _is_const = 0, string _enum_name = "",
                 int _max_value = 0> {
  int pos = _pos;           // position in parameter list
  string llvm_type = type; // llvm type name, $o for overload, $r for resource
                           // type, $cb for legacy cbuffer, $u4 for u4 struct
  string name = _name;      // short, unique name
  string doc = _doc;        // the documentation description of this parameter
  bit is_const =
      _is_const; // whether this argument requires a constant value in the IR
  string enum_name = _enum_name; // the name of the enum type if applicable
  int max_value =
      _max_value; // the maximum value for this parameter if applicable
}

// A representation for a DXIL instruction
class dxil_inst {
  // TODO : Appears redundant. dxil_op should serve the same purpose
  string name = ""; // short, unique name

  string dxil_op = "";       // name of DXIL operation
  int dxil_opid = 0;         // ID of DXIL operation
  DXILClass  op_class;      // name of the opcode class
  DXILCategory category;    // classification for this instruction
  string doc = "";           // the documentation description of this instruction
  list<dxil_param> ops = []; // the operands that this instruction takes
  string oload_types = "";   // overload types if applicable
  string fn_attr = "";       // attribute shorthands: rn=does not access
                             // memory,ro=only reads from memory,
  bit is_deriv = 0;          // whether this is some kind of derivative
  bit is_gradient = 0;       // whether this requires a gradient calculation
  bit is_feedback = 0;       // whether this is a sampler feedback op
  bit is_wave = 0; // whether this requires in-wave, cross-lane functionality
  bit requires_uniform_inputs = 0; // whether this operation requires that all
                                   // of its inputs are uniform across the wave
  // Group dxil operation for stats.
  // Like how many atomic/float/uint/int/... instructions used in the program.
  list<string> stats_group = [];
}

class dxil_op<string _name, int code_id, DXILClass code_class, DXILCategory op_category, string _doc,
              string _oload_types, string _fn_attr, list<dxil_param> op_params,
              list<string> _stats_group = []> : dxil_inst {
  let name = _name;
  let dxil_op = _name;
  let dxil_opid = code_id;
  let doc = _doc;
  let ops = op_params;
  let op_class = code_class;
  let category = op_category;
  let oload_types = _oload_types;
  let fn_attr = _fn_attr;
  let stats_group = _stats_group;
}

class dxil_op_min<string _name, int code_id, string _doc> : dxil_inst {
  let name = _name;
  let dxil_op = _name;
  let dxil_opid = code_id;
  let doc = _doc;
}


// LLVM intrinsic to map dxil op to.
class dxil_map_intrinsic<Intrinsic llvm_intrinsic_> { Intrinsic llvm_intrinsic = llvm_intrinsic_; }

// Unary class DXIL Ops
let op_class = unary, ops = [
      dxil_param<0, "$o", "", "operation result">,
      dxil_param<1, "i32", "opcode", "DXIL opcode">,
      dxil_param<2, "$o", "value", "input value">
    ], fn_attr = "rn" in
// Unary float DXIL Ops
  let category = unary_float, oload_types = "hf", stats_group = ["floats"] in {
    def Cos : dxil_op_min<"Cos", 12, "returns cosine(theta) for theta in radians.">,
              dxil_map_intrinsic<int_cos>;

    def Sin : dxil_op_min<"Sin", 13, "returns sine(theta) for theta in radians.">,
              dxil_map_intrinsic<int_sin>;
  }

// Binary class DXIL Ops
let op_class = binary, ops = [
    dxil_param<0,  "$o",  "",  "operation result">,
    dxil_param<1,  "i32",  "opcode",  "DXIL opcode">,
    dxil_param<2,  "$o",  "a",  "input value">,
    dxil_param<3,  "$o",  "b",  "input value">
  ], fn_attr = "rn" in
// Binary uint DXIL Ops
  let category = binary_uint, oload_types = "wil", stats_group = ["uints"] in
    def UMax : dxil_op_min<"UMax", 39, "unsigned integer maximum. UMax(a,b) = a > b ? a : b">,
                   dxil_map_intrinsic<int_umax>;

// Compute ID class DXIL Ops

def ThreadId :dxil_op< "ThreadId", 93,  thread_id, compute_id, "reads the thread ID", "i",  "rn",
  [
    dxil_param<0,  "i32",  "",  "thread ID component">,
    dxil_param<1,  "i32",  "opcode",  "DXIL opcode">,
    dxil_param<2,  "i32",  "component",  "component to read (x,y,z)">
  ]>,
  dxil_map_intrinsic<int_dx_thread_id>;

def GroupId :dxil_op< "GroupId", 94,  group_id, compute_id, "reads the group ID (SV_GroupID)", "i",  "rn",
  [
    dxil_param<0,  "i32",  "",  "group ID component">,
    dxil_param<1,  "i32",  "opcode",  "DXIL opcode">,
    dxil_param<2,  "i32",  "component",  "component to read">
  ]>,
  dxil_map_intrinsic<int_dx_group_id>;

def ThreadIdInGroup :dxil_op< "ThreadIdInGroup", 95,  threadId_in_group, compute_id,
  "reads the thread ID within the group (SV_GroupThreadID)", "i",  "rn",
  [
    dxil_param<0,  "i32",  "",  "thread ID in group component">,
    dxil_param<1,  "i32",  "opcode",  "DXIL opcode">,
    dxil_param<2,  "i32",  "component",  "component to read (x,y,z)">
  ]>,
  dxil_map_intrinsic<int_dx_thread_id_in_group>;

def FlattenedThreadIdInGroup :dxil_op< "FlattenedThreadIdInGroup", 96,  flattened_threadIdIn_group, compute_id,
   "provides a flattened index for a given thread within a given group (SV_GroupIndex)", "i",  "rn",
  [
    dxil_param<0,  "i32",  "",  "result">,
    dxil_param<1,  "i32",  "opcode",  "DXIL opcode">
  ]>,
  dxil_map_intrinsic<int_dx_flattened_thread_id_in_group>;
