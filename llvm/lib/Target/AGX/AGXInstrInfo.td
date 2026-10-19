//===-- AGXInstrInfo.td - AGX instruction information ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "AGXInstrFormats.td"
include "AGXOperands.td"

//===----------------------------------------------------------------------===//
// Target nodes
//===----------------------------------------------------------------------===//

def SDT_AGXCFCond : SDTypeProfile<0, 2, [SDTCisVT<0, i32>, SDTCisVT<1, OtherVT>]>;
def SDT_AGXCFTarget : SDTypeProfile<0, 1, [SDTCisVT<0, OtherVT>]>;

// Kernel return.
def AGXstop : SDNode<"AGXISD::STOP", SDTNone,
                     [SDNPHasChain, SDNPOptInGlue]>;

// Structured control flow. The branch target is the block executed when no
// thread remains active (or, for loops, the loop header).
def AGXif : SDNode<"AGXISD::IF", SDT_AGXCFCond, [SDNPHasChain]>;
def AGXelse : SDNode<"AGXISD::ELSE", SDT_AGXCFTarget, [SDNPHasChain]>;
def AGXloop : SDNode<"AGXISD::LOOP", SDT_AGXCFCond, [SDNPHasChain]>;
def AGXend_cf : SDNode<"AGXISD::END_CF", SDTNone,
                       [SDNPHasChain, SDNPSideEffect]>;
def AGXloop_begin : SDNode<"AGXISD::LOOP_BEGIN", SDTNone,
                           [SDNPHasChain, SDNPSideEffect]>;

// SIMD-group ballots of a 0/1 value.
def AGXballot : SDNode<"AGXISD::BALLOT", SDTIntUnaryOp>;
def AGXquad_ballot : SDNode<"AGXISD::QUAD_BALLOT", SDTIntUnaryOp>;

// sin(x) for x in quadrants: sin_pt_1, sin_pt_2 and a multiply.
def AGXsin_quadrants : SDNode<"AGXISD::SIN_QUADRANTS", SDTFPUnaryOp>;

//===----------------------------------------------------------------------===//
// Complex patterns
//===----------------------------------------------------------------------===//

// Integer ALU source: a register, a uniform register or an 8-bit immediate.
def AGXIntSrc16 : ComplexPattern<i16, 1, "SelectIntSrc", [], [], -2>;
def AGXIntSrc32 : ComplexPattern<i32, 1, "SelectIntSrc", [], [], -2>;
def AGXIntSrc64 : ComplexPattern<i64, 1, "SelectIntSrc", [], [], -2>;

// Floating point source with abs/neg modifiers: (value, mods). The value may
// be an 8-bit floating point immediate.
def AGXFPSrc16 : ComplexPattern<f16, 2, "SelectFPSrc", [], [], -2>;
def AGXFPSrc32 : ComplexPattern<f32, 2, "SelectFPSrc", [], [], -2>;

// Device memory addresses: (base, index, signedness, shift) for an element
// size of 1, 2 or 4 bytes.
def AGXDeviceAddr8 : ComplexPattern<i64, 4, "SelectDeviceAddr8", [], [], -10>;
def AGXDeviceAddr16 : ComplexPattern<i64, 4, "SelectDeviceAddr16", [], [], -10>;
def AGXDeviceAddr32 : ComplexPattern<i64, 4, "SelectDeviceAddr32", [], [], -10>;

// Threadgroup memory addresses: (base, index) for an element size of 1, 2 or
// 4 bytes.
def AGXLocalAddr8 : ComplexPattern<i32, 2, "SelectLocalAddr8", [], [], -10>;
def AGXLocalAddr16 : ComplexPattern<i32, 2, "SelectLocalAddr16", [], [], -10>;
def AGXLocalAddr32 : ComplexPattern<i32, 2, "SelectLocalAddr32", [], [], -10>;

// Stack addresses: a byte offset (frame index, register or immediate).
def AGXStackAddr : ComplexPattern<i32, 1, "SelectStackAddr", [frameindex],
                                  [], -10>;

//===----------------------------------------------------------------------===//
// Address space fragments
//===----------------------------------------------------------------------===//

class AddressSpaceList<list<int> AS> {
  list<int> AddrSpaces = AS;
}

// Generic (0), device (1) and constant (4) pointers all address device
// memory.
defvar DeviceAS = [0, 1, 4];
defvar LocalAS = [3];
defvar PrivateAS = [5];

// The address space check is only emitted for fragments flagged as loads or
// stores.
multiclass AGXLoadFrags<list<int> AS> {
  let AddressSpaces = AS, IsLoad = 1 in {
  def _load : PatFrag<(ops node:$ptr), (load node:$ptr)>;
  def _extloadi8 : PatFrag<(ops node:$ptr), (extloadi8 node:$ptr)>;
  def _zextloadi8 : PatFrag<(ops node:$ptr), (zextloadi8 node:$ptr)>;
  def _extloadi16 : PatFrag<(ops node:$ptr), (extloadi16 node:$ptr)>;
  def _zextloadi16 : PatFrag<(ops node:$ptr), (zextloadi16 node:$ptr)>;
  }
  let AddressSpaces = AS, IsStore = 1 in {
  def _store : PatFrag<(ops node:$val, node:$ptr), (store node:$val, node:$ptr)>;
  def _truncstorei8 : PatFrag<(ops node:$val, node:$ptr),
                              (truncstorei8 node:$val, node:$ptr)>;
  def _truncstorei16 : PatFrag<(ops node:$val, node:$ptr),
                               (truncstorei16 node:$val, node:$ptr)>;
  }
}

defm device : AGXLoadFrags<DeviceAS>;
defm local : AGXLoadFrags<LocalAS>;
defm private : AGXLoadFrags<PrivateAS>;

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

include "G13Instructions.td"
include "AGXInstructions.td"
include "AGXSearchableTables.td"
