//===---- AMDCallingConv.td - Calling Conventions for Radeon GPUs ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This describes the calling conventions for the AMD Radeon GPUs.
//
//===----------------------------------------------------------------------===//

// Inversion of CCIfInReg
class CCIfNotInReg<CCAction A> : CCIf<"!ArgFlags.isInReg()", A> {}

// Calling convention for SI
def CC_SI : CallingConv<[

  CCIfInReg<CCIfType<[f32, i32] , CCAssignToReg<[
    SGPR0, SGPR1, SGPR2, SGPR3, SGPR4, SGPR5, SGPR6, SGPR7,
    SGPR8, SGPR9, SGPR10, SGPR11, SGPR12, SGPR13, SGPR14, SGPR15,
    SGPR16, SGPR17, SGPR18, SGPR19, SGPR20, SGPR21
  ]>>>,

  CCIfInReg<CCIfType<[i64] , CCAssignToRegWithShadow<
    [ SGPR0, SGPR2, SGPR4, SGPR6, SGPR8, SGPR10, SGPR12, SGPR14 ],
    [ SGPR1, SGPR3, SGPR5, SGPR7, SGPR9, SGPR11, SGPR13, SGPR15 ]
  >>>,

  CCIfNotInReg<CCIfType<[f32, i32] , CCAssignToReg<[
    VGPR0, VGPR1, VGPR2, VGPR3, VGPR4, VGPR5, VGPR6, VGPR7,
    VGPR8, VGPR9, VGPR10, VGPR11, VGPR12, VGPR13, VGPR14, VGPR15,
    VGPR16, VGPR17, VGPR18, VGPR19, VGPR20, VGPR21, VGPR22, VGPR23,
    VGPR24, VGPR25, VGPR26, VGPR27, VGPR28, VGPR29, VGPR30, VGPR31
  ]>>>,

  CCIfByVal<CCIfType<[i64] , CCAssignToRegWithShadow<
    [ SGPR0, SGPR2, SGPR4, SGPR6, SGPR8, SGPR10, SGPR12, SGPR14 ],
    [ SGPR1, SGPR3, SGPR5, SGPR7, SGPR9, SGPR11, SGPR13, SGPR15 ]
  >>>

]>;

// Calling convention for R600
def CC_R600 : CallingConv<[
  CCIfInReg<CCIfType<[v4f32, v4i32] , CCAssignToReg<[
    T0_XYZW, T1_XYZW, T2_XYZW, T3_XYZW, T4_XYZW, T5_XYZW, T6_XYZW, T7_XYZW,
    T8_XYZW, T9_XYZW, T10_XYZW, T11_XYZW, T12_XYZW, T13_XYZW, T14_XYZW, T15_XYZW,
    T16_XYZW, T17_XYZW, T18_XYZW, T19_XYZW, T20_XYZW, T21_XYZW, T22_XYZW,
    T23_XYZW, T24_XYZW, T25_XYZW, T26_XYZW, T27_XYZW, T28_XYZW, T29_XYZW,
    T30_XYZW, T31_XYZW, T32_XYZW
  ]>>>
]>;

// Calling convention for compute kernels
def CC_AMDGPU_Kernel : CallingConv<[
  CCCustom<"allocateStack">
]>;

def CC_AMDGPU : CallingConv<[
  CCIf<"State.getTarget().getSubtarget<AMDGPUSubtarget>().getGeneration() >= "
       "AMDGPUSubtarget::SOUTHERN_ISLANDS && "
       "State.getMachineFunction().getInfo<SIMachineFunctionInfo>()->"#
       "ShaderType == ShaderType::COMPUTE", CCDelegateTo<CC_AMDGPU_Kernel>>,
  CCIf<"State.getTarget().getSubtarget<AMDGPUSubtarget>().getGeneration() < "
       "AMDGPUSubtarget::SOUTHERN_ISLANDS && "
       "State.getMachineFunction().getInfo<R600MachineFunctionInfo>()->"
       "ShaderType == ShaderType::COMPUTE", CCDelegateTo<CC_AMDGPU_Kernel>>,
  CCIf<"State.getTarget().getSubtarget<AMDGPUSubtarget>()"#
       ".getGeneration() >= AMDGPUSubtarget::SOUTHERN_ISLANDS", CCDelegateTo<CC_SI>>,
  CCIf<"State.getTarget().getSubtarget<AMDGPUSubtarget>()"#
       ".getGeneration() < AMDGPUSubtarget::SOUTHERN_ISLANDS", CCDelegateTo<CC_R600>>
]>;
