//===---- AMDCallingConv.td - Calling Conventions for Radeon GPUs ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the calling conventions for the AMD Radeon GPUs.
//
//===----------------------------------------------------------------------===//

// Inversion of CCIfInReg
class CCIfNotInReg<CCAction A> : CCIf<"!ArgFlags.isInReg()", A> {}
class CCIfExtend<CCAction A>
  : CCIf<"ArgFlags.isSExt() || ArgFlags.isZExt()", A>;

// Calling convention for SI
def CC_SI_Gfx : CallingConv<[
  // 0-3 are reserved for the stack buffer descriptor
  // 30-31 are reserved for the return address
  // 32 is reserved for the stack pointer
  // 33 is reserved for the frame pointer
  // 34 is reserved for the base pointer
  CCIfInReg<CCIfType<[f32, i32, f16, i16, v2i16, v2f16, bf16, v2bf16] , CCAssignToReg<
    !foreach(i, !range(4, 30), !cast<Register>("SGPR"#i))  // SGPR4-29
  >>>,

  CCIfNotInReg<CCIfType<[f32, i32, f16, i16, v2i16, v2f16, bf16, v2bf16] , CCAssignToReg<
    !foreach(i, !range(0, 32), !cast<Register>("VGPR"#i))  // VGPR0-31
  >>>,

  CCIfType<[i32, f32, v2i16, v2f16, i16, f16, i1, bf16, v2bf16], CCAssignToStack<4, 4>>
]>;

def RetCC_SI_Gfx : CallingConv<[
  CCIfType<[i1], CCPromoteToType<i32>>,
  CCIfType<[i1, i16], CCIfExtend<CCPromoteToType<i32>>>,

  CCIfNotInReg<CCIfType<[f32, i32, f16, i16, v2i16, v2f16, bf16, v2bf16] , CCAssignToReg<
    !foreach(i, !range(0, 136), !cast<Register>("VGPR"#i))  // VGPR0-135
  >>>,
]>;

def CC_SI_SHADER : CallingConv<[

  CCIfType<[i1], CCPromoteToType<i32>>,

  CCIfInReg<CCIfType<[f32, i32, f16, i16, v2i16, v2f16, bf16, v2bf16] , CCAssignToReg<
    !foreach(i, !range(0, 44), !cast<Register>("SGPR"#i))  // SGPR0-43
  >>>,

  // 32*4 + 4 is the minimum for a fetch shader consumer with 32 inputs.
  CCIfNotInReg<CCIfType<[f32, i32, f16, i16, v2i16, v2f16, bf16, v2bf16] , CCAssignToReg<
    !foreach(i, !range(0, 136), !cast<Register>("VGPR"#i))  // VGPR0-135
  >>>
]>;

def RetCC_SI_Shader : CallingConv<[
  CCIfType<[i1, i16], CCIfExtend<CCPromoteToType<i32>>>,
  CCIfType<[i32, i16, v2i16] , CCAssignToReg<
    !foreach(i, !range(0, 44), !cast<Register>("SGPR"#i))  // SGPR0-43
  >>,

  // 32*4 + 4 is the minimum for a fetch shader with 32 outputs.
  CCIfType<[f32, f16, v2f16, bf16, v2bf16] , CCAssignToReg<
    !foreach(i, !range(0, 136), !cast<Register>("VGPR"#i))  // VGPR0-135
  >>
]>;

def CSR_AMDGPU_VGPRs : CalleeSavedRegs<
  // The CSRs & scratch-registers are interleaved at a split boundary of 8.
  (add (sequence "VGPR%u", 40, 47),
    (sequence "VGPR%u", 56, 63),
    (sequence "VGPR%u", 72, 79),
    (sequence "VGPR%u", 88, 95),
    (sequence "VGPR%u", 104, 111),
    (sequence "VGPR%u", 120, 127),
    (sequence "VGPR%u", 136, 143),
    (sequence "VGPR%u", 152, 159),
    (sequence "VGPR%u", 168, 175),
    (sequence "VGPR%u", 184, 191),
    (sequence "VGPR%u", 200, 207),
    (sequence "VGPR%u", 216, 223),
    (sequence "VGPR%u", 232, 239),
    (sequence "VGPR%u", 248, 255))
>;

def CSR_AMDGPU_AGPRs : CalleeSavedRegs<
  (sequence "AGPR%u", 32, 255)
>;

def CSR_AMDGPU_SGPRs : CalleeSavedRegs<
  (sequence "SGPR%u", 30, 105)
>;

def CSR_AMDGPU_SI_Gfx_SGPRs : CalleeSavedRegs<
  (add (sequence "SGPR%u", 4, 31), (sequence "SGPR%u", 64, 105))
>;

def CSR_AMDGPU : CalleeSavedRegs<
  (add CSR_AMDGPU_VGPRs, CSR_AMDGPU_SGPRs)
>;

def CSR_AMDGPU_GFX90AInsts : CalleeSavedRegs<
  (add CSR_AMDGPU, CSR_AMDGPU_AGPRs)
>;

def CSR_AMDGPU_SI_Gfx : CalleeSavedRegs<
  (add CSR_AMDGPU_VGPRs, CSR_AMDGPU_SI_Gfx_SGPRs)
>;

def CSR_AMDGPU_SI_Gfx_GFX90AInsts : CalleeSavedRegs<
  (add CSR_AMDGPU_SI_Gfx, CSR_AMDGPU_AGPRs)
>;

def CSR_AMDGPU_CS_ChainPreserve : CalleeSavedRegs<
  (sequence "VGPR%u", 8, 255)
>;

def CSR_AMDGPU_NoRegs : CalleeSavedRegs<(add)>;

// Calling convention for leaf functions
def CC_AMDGPU_Func : CallingConv<[
  CCIfByVal<CCPassByVal<4, 4>>,
  CCIfType<[i1], CCPromoteToType<i32>>,
  CCIfType<[i8, i16], CCIfExtend<CCPromoteToType<i32>>>,

  CCIfInReg<CCIfType<[f32, i32, f16, i16, v2i16, v2f16, bf16, v2bf16] , CCAssignToReg<
    !foreach(i, !range(0, 30), !cast<Register>("SGPR"#i))  // SGPR0-29
  >>>,

  CCIfType<[i32, f32, i16, f16, v2i16, v2f16, i1, bf16, v2bf16], CCAssignToReg<
    !foreach(i, !range(0, 32), !cast<Register>("VGPR"#i))  // VGPR0-31
  >>,
  CCIfType<[i32, f32, v2i16, v2f16, i16, f16, i1, bf16, v2bf16], CCAssignToStack<4, 4>>
]>;

// Calling convention for leaf functions
def RetCC_AMDGPU_Func : CallingConv<[
  CCIfType<[i1], CCPromoteToType<i32>>,
  CCIfType<[i1, i16], CCIfExtend<CCPromoteToType<i32>>>,
  CCIfType<[i32, f32, i16, f16, v2i16, v2f16, bf16, v2bf16], CCAssignToReg<
    !foreach(i, !range(0, 32), !cast<Register>("VGPR"#i))  // VGPR0-31
  >>,
]>;

def CC_AMDGPU : CallingConv<[
   CCIf<"State.getMachineFunction().getSubtarget<GCNSubtarget>().getGeneration() >= "
          "AMDGPUSubtarget::SOUTHERN_ISLANDS",
        CCDelegateTo<CC_SI_SHADER>>,
   CCIf<"State.getMachineFunction().getSubtarget<GCNSubtarget>().getGeneration() >= "
          "AMDGPUSubtarget::SOUTHERN_ISLANDS && State.getCallingConv() == CallingConv::C",
        CCDelegateTo<CC_AMDGPU_Func>>
]>;

def CC_AMDGPU_CS_CHAIN : CallingConv<[
  CCIfInReg<CCIfType<[f32, i32, f16, i16, v2i16, v2f16, bf16, v2bf16] , CCAssignToReg<
    !foreach(i, !range(105), !cast<Register>("SGPR"#i))
  >>>,

  CCIfNotInReg<CCIfType<[f32, i32, f16, i16, v2i16, v2f16, bf16, v2bf16] , CCAssignToReg<
    !foreach(i, !range(8, 255), !cast<Register>("VGPR"#i))
  >>>
]>;

// Trivial class to denote when a def is used only to get a RegMask, i.e.
// SaveList is ignored and the def is not used as part of any calling
// convention.
class RegMask<dag mask> : CalleeSavedRegs<mask>;

def AMDGPU_AllVGPRs : RegMask<
  (sequence "VGPR%u", 0, 255)
>;

def AMDGPU_AllAGPRs : RegMask<
  (sequence "AGPR%u", 0, 255)
>;

def AMDGPU_AllVectorRegs : RegMask<
  (add AMDGPU_AllVGPRs, AMDGPU_AllAGPRs)
>;

def AMDGPU_AllAllocatableSRegs : RegMask<
  (add (sequence "SGPR%u", 0, 105), VCC_LO, VCC_HI)
>;
