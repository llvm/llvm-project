//===-- AMDGPUInstrInfo.td - AMDGPU DAG nodes --------------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains DAG node defintions for the AMDGPU target.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AMDGPU DAG Profiles
//===----------------------------------------------------------------------===//

def AMDGPUDTIntTernaryOp : SDTypeProfile<1, 3, [
  SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisInt<0>, SDTCisInt<3>
]>;

//===----------------------------------------------------------------------===//
// AMDGPU DAG Nodes
//

// This argument to this node is a dword address.
def AMDGPUdwordaddr : SDNode<"AMDGPUISD::DWORDADDR", SDTIntUnaryOp>;

// out = a - floor(a)
def AMDGPUfract : SDNode<"AMDGPUISD::FRACT", SDTFPUnaryOp>;

// out = max(a, b) a and b are floats
def AMDGPUfmax : SDNode<"AMDGPUISD::FMAX", SDTFPBinOp,
  [SDNPCommutative, SDNPAssociative]
>;

// out = max(a, b) a and b are signed ints
def AMDGPUsmax : SDNode<"AMDGPUISD::SMAX", SDTIntBinOp,
  [SDNPCommutative, SDNPAssociative]
>;

// out = max(a, b) a and b are unsigned ints
def AMDGPUumax : SDNode<"AMDGPUISD::UMAX", SDTIntBinOp,
  [SDNPCommutative, SDNPAssociative]
>;

// out = min(a, b) a and b are floats
def AMDGPUfmin : SDNode<"AMDGPUISD::FMIN", SDTFPBinOp,
  [SDNPCommutative, SDNPAssociative]
>;

// out = min(a, b) a snd b are signed ints
def AMDGPUsmin : SDNode<"AMDGPUISD::SMIN", SDTIntBinOp,
  [SDNPCommutative, SDNPAssociative]
>;

// out = min(a, b) a and b are unsigned ints
def AMDGPUumin : SDNode<"AMDGPUISD::UMIN", SDTIntBinOp,
  [SDNPCommutative, SDNPAssociative]
>;

// urecip - This operation is a helper for integer division, it returns the
// result of 1 / a as a fractional unsigned integer.
// out = (2^32 / a) + e
// e is rounding error
def AMDGPUurecip : SDNode<"AMDGPUISD::URECIP", SDTIntUnaryOp>;

def AMDGPUregister_load : SDNode<"AMDGPUISD::REGISTER_LOAD",
                          SDTypeProfile<1, 2, [SDTCisPtrTy<1>, SDTCisInt<2>]>,
                          [SDNPHasChain, SDNPMayLoad]>;

def AMDGPUregister_store : SDNode<"AMDGPUISD::REGISTER_STORE",
                           SDTypeProfile<0, 3, [SDTCisPtrTy<1>, SDTCisInt<2>]>,
                           [SDNPHasChain, SDNPMayStore]>;

// MSKOR instructions are atomic memory instructions used mainly for storing
// 8-bit and 16-bit values.  The definition is:
//
// MSKOR(dst, mask, src) MEM[dst] = ((MEM[dst] & ~mask) | src)
//
// src0: vec4(src, 0, 0, mask)
// src1: dst - rat offset (aka pointer) in dwords  
def AMDGPUstore_mskor : SDNode<"AMDGPUISD::STORE_MSKOR",
                        SDTypeProfile<0, 2, []>,
                        [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

def AMDGPUround : SDNode<"ISD::FROUND",
                         SDTypeProfile<1, 1, [SDTCisFP<0>, SDTCisSameAs<0,1>]>>;
