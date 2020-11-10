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
  CCIfInReg<CCIfType<[f32, i32, f16, i16, v2i16, v2f16] , CCAssignToReg<[
    SGPR4, SGPR5, SGPR6, SGPR7,
    SGPR8, SGPR9, SGPR10, SGPR11, SGPR12, SGPR13, SGPR14, SGPR15,
    SGPR16, SGPR17, SGPR18, SGPR19, SGPR20, SGPR21, SGPR22, SGPR23,
    SGPR24, SGPR25, SGPR26, SGPR27, SGPR28, SGPR29,
  ]>>>,

  CCIfNotInReg<CCIfType<[f32, i32, f16, i16, v2i16, v2f16] , CCAssignToReg<[
    VGPR0, VGPR1, VGPR2, VGPR3, VGPR4, VGPR5, VGPR6, VGPR7,
    VGPR8, VGPR9, VGPR10, VGPR11, VGPR12, VGPR13, VGPR14, VGPR15,
    VGPR16, VGPR17, VGPR18, VGPR19, VGPR20, VGPR21, VGPR22, VGPR23,
    VGPR24, VGPR25, VGPR26, VGPR27, VGPR28, VGPR29, VGPR30, VGPR31
  ]>>>,

  CCIfType<[i32, f32, v2i16, v2f16, i16, f16, i1], CCAssignToStack<4, 4>>,
  CCIfType<[i64, f64, v2i32, v2f32], CCAssignToStack<8, 4>>,
  CCIfType<[v3i32, v3f32], CCAssignToStack<12, 4>>,
  CCIfType<[v4i32, v4f32, v2i64, v2f64], CCAssignToStack<16, 4>>,
  CCIfType<[v5i32, v5f32], CCAssignToStack<20, 4>>,
  CCIfType<[v8i32, v8f32], CCAssignToStack<32, 4>>,
  CCIfType<[v16i32, v16f32], CCAssignToStack<64, 4>>
]>;

def RetCC_SI_Gfx : CallingConv<[
  // 0-3 are reserved for the stack buffer descriptor
  // 32 is reserved for the stack pointer
  CCIfInReg<CCIfType<[f32, i32, f16, i16, v2i16, v2f16] , CCAssignToReg<[
    SGPR4, SGPR5, SGPR6, SGPR7,
    SGPR8, SGPR9, SGPR10, SGPR11, SGPR12, SGPR13, SGPR14, SGPR15,
    SGPR16, SGPR17, SGPR18, SGPR19, SGPR20, SGPR21, SGPR22, SGPR23,
    SGPR24, SGPR25, SGPR26, SGPR27, SGPR28, SGPR29, SGPR30, SGPR31,
    SGPR33, SGPR34, SGPR35, SGPR36, SGPR37, SGPR38, SGPR39,
    SGPR40, SGPR41, SGPR42, SGPR43
  ]>>>,

  CCIfNotInReg<CCIfType<[f32, i32, f16, i16, v2i16, v2f16] , CCAssignToReg<[
    VGPR0, VGPR1, VGPR2, VGPR3, VGPR4, VGPR5, VGPR6, VGPR7,
    VGPR8, VGPR9, VGPR10, VGPR11, VGPR12, VGPR13, VGPR14, VGPR15,
    VGPR16, VGPR17, VGPR18, VGPR19, VGPR20, VGPR21, VGPR22, VGPR23,
    VGPR24, VGPR25, VGPR26, VGPR27, VGPR28, VGPR29, VGPR30, VGPR31,
    VGPR32, VGPR33, VGPR34, VGPR35, VGPR36, VGPR37, VGPR38, VGPR39,
    VGPR40, VGPR41, VGPR42, VGPR43, VGPR44, VGPR45, VGPR46, VGPR47,
    VGPR48, VGPR49, VGPR50, VGPR51, VGPR52, VGPR53, VGPR54, VGPR55,
    VGPR56, VGPR57, VGPR58, VGPR59, VGPR60, VGPR61, VGPR62, VGPR63,
    VGPR64, VGPR65, VGPR66, VGPR67, VGPR68, VGPR69, VGPR70, VGPR71,
    VGPR72, VGPR73, VGPR74, VGPR75, VGPR76, VGPR77, VGPR78, VGPR79,
    VGPR80, VGPR81, VGPR82, VGPR83, VGPR84, VGPR85, VGPR86, VGPR87,
    VGPR88, VGPR89, VGPR90, VGPR91, VGPR92, VGPR93, VGPR94, VGPR95,
    VGPR96, VGPR97, VGPR98, VGPR99, VGPR100, VGPR101, VGPR102, VGPR103,
    VGPR104, VGPR105, VGPR106, VGPR107, VGPR108, VGPR109, VGPR110, VGPR111,
    VGPR112, VGPR113, VGPR114, VGPR115, VGPR116, VGPR117, VGPR118, VGPR119,
    VGPR120, VGPR121, VGPR122, VGPR123, VGPR124, VGPR125, VGPR126, VGPR127,
    VGPR128, VGPR129, VGPR130, VGPR131, VGPR132, VGPR133, VGPR134, VGPR135
  ]>>>,

  CCIfType<[i32, f32, v2i16, v2f16, i16, f16, i1], CCAssignToStack<4, 4>>,
  CCIfType<[i64, f64, v2i32, v2f32], CCAssignToStack<8, 4>>,
  CCIfType<[v3i32, v3f32], CCAssignToStack<12, 4>>,
  CCIfType<[v4i32, v4f32, v2i64, v2f64], CCAssignToStack<16, 4>>,
  CCIfType<[v5i32, v5f32], CCAssignToStack<20, 4>>,
  CCIfType<[v8i32, v8f32], CCAssignToStack<32, 4>>,
  CCIfType<[v16i32, v16f32], CCAssignToStack<64, 4>>
]>;

def CC_SI_SHADER : CallingConv<[

  CCIfInReg<CCIfType<[f32, i32, f16, i16, v2i16, v2f16] , CCAssignToReg<[
    SGPR0, SGPR1, SGPR2, SGPR3, SGPR4, SGPR5, SGPR6, SGPR7,
    SGPR8, SGPR9, SGPR10, SGPR11, SGPR12, SGPR13, SGPR14, SGPR15,
    SGPR16, SGPR17, SGPR18, SGPR19, SGPR20, SGPR21, SGPR22, SGPR23,
    SGPR24, SGPR25, SGPR26, SGPR27, SGPR28, SGPR29, SGPR30, SGPR31,
    SGPR32, SGPR33, SGPR34, SGPR35, SGPR36, SGPR37, SGPR38, SGPR39,
    SGPR40, SGPR41, SGPR42, SGPR43
  ]>>>,

  // 32*4 + 4 is the minimum for a fetch shader consumer with 32 inputs.
  CCIfNotInReg<CCIfType<[f32, i32, f16, i16, v2i16, v2f16] , CCAssignToReg<[
    VGPR0, VGPR1, VGPR2, VGPR3, VGPR4, VGPR5, VGPR6, VGPR7,
    VGPR8, VGPR9, VGPR10, VGPR11, VGPR12, VGPR13, VGPR14, VGPR15,
    VGPR16, VGPR17, VGPR18, VGPR19, VGPR20, VGPR21, VGPR22, VGPR23,
    VGPR24, VGPR25, VGPR26, VGPR27, VGPR28, VGPR29, VGPR30, VGPR31,
    VGPR32, VGPR33, VGPR34, VGPR35, VGPR36, VGPR37, VGPR38, VGPR39,
    VGPR40, VGPR41, VGPR42, VGPR43, VGPR44, VGPR45, VGPR46, VGPR47,
    VGPR48, VGPR49, VGPR50, VGPR51, VGPR52, VGPR53, VGPR54, VGPR55,
    VGPR56, VGPR57, VGPR58, VGPR59, VGPR60, VGPR61, VGPR62, VGPR63,
    VGPR64, VGPR65, VGPR66, VGPR67, VGPR68, VGPR69, VGPR70, VGPR71,
    VGPR72, VGPR73, VGPR74, VGPR75, VGPR76, VGPR77, VGPR78, VGPR79,
    VGPR80, VGPR81, VGPR82, VGPR83, VGPR84, VGPR85, VGPR86, VGPR87,
    VGPR88, VGPR89, VGPR90, VGPR91, VGPR92, VGPR93, VGPR94, VGPR95,
    VGPR96, VGPR97, VGPR98, VGPR99, VGPR100, VGPR101, VGPR102, VGPR103,
    VGPR104, VGPR105, VGPR106, VGPR107, VGPR108, VGPR109, VGPR110, VGPR111,
    VGPR112, VGPR113, VGPR114, VGPR115, VGPR116, VGPR117, VGPR118, VGPR119,
    VGPR120, VGPR121, VGPR122, VGPR123, VGPR124, VGPR125, VGPR126, VGPR127,
    VGPR128, VGPR129, VGPR130, VGPR131, VGPR132, VGPR133, VGPR134, VGPR135
  ]>>>
]>;

def RetCC_SI_Shader : CallingConv<[
  CCIfType<[i32, i16] , CCAssignToReg<[
    SGPR0, SGPR1, SGPR2, SGPR3, SGPR4, SGPR5, SGPR6, SGPR7,
    SGPR8, SGPR9, SGPR10, SGPR11, SGPR12, SGPR13, SGPR14, SGPR15,
    SGPR16, SGPR17, SGPR18, SGPR19, SGPR20, SGPR21, SGPR22, SGPR23,
    SGPR24, SGPR25, SGPR26, SGPR27, SGPR28, SGPR29, SGPR30, SGPR31,
    SGPR32, SGPR33, SGPR34, SGPR35, SGPR36, SGPR37, SGPR38, SGPR39,
    SGPR40, SGPR41, SGPR42, SGPR43
  ]>>,

  // 32*4 + 4 is the minimum for a fetch shader with 32 outputs.
  CCIfType<[f32, f16, v2f16] , CCAssignToReg<[
    VGPR0, VGPR1, VGPR2, VGPR3, VGPR4, VGPR5, VGPR6, VGPR7,
    VGPR8, VGPR9, VGPR10, VGPR11, VGPR12, VGPR13, VGPR14, VGPR15,
    VGPR16, VGPR17, VGPR18, VGPR19, VGPR20, VGPR21, VGPR22, VGPR23,
    VGPR24, VGPR25, VGPR26, VGPR27, VGPR28, VGPR29, VGPR30, VGPR31,
    VGPR32, VGPR33, VGPR34, VGPR35, VGPR36, VGPR37, VGPR38, VGPR39,
    VGPR40, VGPR41, VGPR42, VGPR43, VGPR44, VGPR45, VGPR46, VGPR47,
    VGPR48, VGPR49, VGPR50, VGPR51, VGPR52, VGPR53, VGPR54, VGPR55,
    VGPR56, VGPR57, VGPR58, VGPR59, VGPR60, VGPR61, VGPR62, VGPR63,
    VGPR64, VGPR65, VGPR66, VGPR67, VGPR68, VGPR69, VGPR70, VGPR71,
    VGPR72, VGPR73, VGPR74, VGPR75, VGPR76, VGPR77, VGPR78, VGPR79,
    VGPR80, VGPR81, VGPR82, VGPR83, VGPR84, VGPR85, VGPR86, VGPR87,
    VGPR88, VGPR89, VGPR90, VGPR91, VGPR92, VGPR93, VGPR94, VGPR95,
    VGPR96, VGPR97, VGPR98, VGPR99, VGPR100, VGPR101, VGPR102, VGPR103,
    VGPR104, VGPR105, VGPR106, VGPR107, VGPR108, VGPR109, VGPR110, VGPR111,
    VGPR112, VGPR113, VGPR114, VGPR115, VGPR116, VGPR117, VGPR118, VGPR119,
    VGPR120, VGPR121, VGPR122, VGPR123, VGPR124, VGPR125, VGPR126, VGPR127,
    VGPR128, VGPR129, VGPR130, VGPR131, VGPR132, VGPR133, VGPR134, VGPR135
  ]>>
]>;

def CSR_AMDGPU_VGPRs_24_255 : CalleeSavedRegs<
  (sequence "VGPR%u", 24, 255)
>;

def CSR_AMDGPU_VGPRs_32_255 : CalleeSavedRegs<
  (sequence "VGPR%u", 32, 255)
>;

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

def CSR_AMDGPU_SGPRs_32_105 : CalleeSavedRegs<
  (sequence "SGPR%u", 32, 105)
>;

// Just to get the regmask, not for calling convention purposes.
def CSR_AMDGPU_AllVGPRs : CalleeSavedRegs<
  (sequence "VGPR%u", 0, 255)
>;

// Just to get the regmask, not for calling convention purposes.
def CSR_AMDGPU_AllAllocatableSRegs : CalleeSavedRegs<
  (add (sequence "SGPR%u", 0, 105), VCC_LO, VCC_HI)
>;

def CSR_AMDGPU_HighRegs : CalleeSavedRegs<
  (add CSR_AMDGPU_VGPRs, CSR_AMDGPU_SGPRs_32_105)
>;

def CSR_AMDGPU_NoRegs : CalleeSavedRegs<(add)>;

// Calling convention for leaf functions
def CC_AMDGPU_Func : CallingConv<[
  CCIfByVal<CCPassByVal<4, 4>>,
  CCIfType<[i1], CCPromoteToType<i32>>,
  CCIfType<[i1, i8, i16], CCIfExtend<CCPromoteToType<i32>>>,
  CCIfType<[i32, f32, i16, f16, v2i16, v2f16, i1], CCAssignToReg<[
    VGPR0, VGPR1, VGPR2, VGPR3, VGPR4, VGPR5, VGPR6, VGPR7,
    VGPR8, VGPR9, VGPR10, VGPR11, VGPR12, VGPR13, VGPR14, VGPR15,
    VGPR16, VGPR17, VGPR18, VGPR19, VGPR20, VGPR21, VGPR22, VGPR23,
    VGPR24, VGPR25, VGPR26, VGPR27, VGPR28, VGPR29, VGPR30, VGPR31]>>,
  CCIfType<[i32, f32, v2i16, v2f16, i16, f16, i1], CCAssignToStack<4, 4>>,
  CCIfType<[i64, f64, v2i32, v2f32], CCAssignToStack<8, 4>>,
  CCIfType<[v3i32, v3f32], CCAssignToStack<12, 4>>,
  CCIfType<[v4i32, v4f32, v2i64, v2f64], CCAssignToStack<16, 4>>,
  CCIfType<[v5i32, v5f32], CCAssignToStack<20, 4>>,
  CCIfType<[v8i32, v8f32], CCAssignToStack<32, 4>>,
  CCIfType<[v16i32, v16f32], CCAssignToStack<64, 4>>
]>;

// Calling convention for leaf functions
def RetCC_AMDGPU_Func : CallingConv<[
  CCIfType<[i1], CCPromoteToType<i32>>,
  CCIfType<[i1, i16], CCIfExtend<CCPromoteToType<i32>>>,
  CCIfType<[i32, f32, i16, f16, v2i16, v2f16], CCAssignToReg<[
    VGPR0, VGPR1, VGPR2, VGPR3, VGPR4, VGPR5, VGPR6, VGPR7,
    VGPR8, VGPR9, VGPR10, VGPR11, VGPR12, VGPR13, VGPR14, VGPR15,
    VGPR16, VGPR17, VGPR18, VGPR19, VGPR20, VGPR21, VGPR22, VGPR23,
    VGPR24, VGPR25, VGPR26, VGPR27, VGPR28, VGPR29, VGPR30, VGPR31]>>,
]>;

def CC_AMDGPU : CallingConv<[
   CCIf<"static_cast<const GCNSubtarget&>"
         "(State.getMachineFunction().getSubtarget()).getGeneration() >= "
           "AMDGPUSubtarget::SOUTHERN_ISLANDS",
        CCDelegateTo<CC_SI_SHADER>>,
   CCIf<"static_cast<const GCNSubtarget&>"
         "(State.getMachineFunction().getSubtarget()).getGeneration() >= "
           "AMDGPUSubtarget::SOUTHERN_ISLANDS && State.getCallingConv() == CallingConv::C",
        CCDelegateTo<CC_AMDGPU_Func>>
]>;
