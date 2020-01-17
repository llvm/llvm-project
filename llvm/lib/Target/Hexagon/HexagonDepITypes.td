//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Automatically generated file, do not edit!
//===----------------------------------------------------------------------===//

class IType<bits<7> t> { bits<7> Value = t; }
def TypeALU32_2op : IType<0>;
def TypeALU32_3op : IType<1>;
def TypeALU32_ADDI : IType<2>;
def TypeALU64 : IType<3>;
def TypeCJ : IType<4>;
def TypeCR : IType<7>;
def TypeCVI_4SLOT_MPY : IType<8>;
def TypeCVI_GATHER : IType<9>;
def TypeCVI_GATHER_DV : IType<10>;
def TypeCVI_GATHER_RST : IType<11>;
def TypeCVI_HIST : IType<12>;
def TypeCVI_SCATTER : IType<13>;
def TypeCVI_SCATTER_DV : IType<14>;
def TypeCVI_SCATTER_NEW_RST : IType<15>;
def TypeCVI_SCATTER_NEW_ST : IType<16>;
def TypeCVI_SCATTER_RST : IType<17>;
def TypeCVI_VA : IType<18>;
def TypeCVI_VA_DV : IType<19>;
def TypeCVI_VM_LD : IType<20>;
def TypeCVI_VM_NEW_ST : IType<21>;
def TypeCVI_VM_ST : IType<22>;
def TypeCVI_VM_STU : IType<23>;
def TypeCVI_VM_TMP_LD : IType<24>;
def TypeCVI_VM_VP_LDU : IType<25>;
def TypeCVI_VP : IType<26>;
def TypeCVI_VP_VS : IType<27>;
def TypeCVI_VS : IType<28>;
def TypeCVI_VS_VX : IType<29>;
def TypeCVI_VX : IType<30>;
def TypeCVI_VX_DV : IType<31>;
def TypeCVI_VX_LATE : IType<32>;
def TypeCVI_ZW : IType<33>;
def TypeDUPLEX : IType<34>;
def TypeENDLOOP : IType<35>;
def TypeEXTENDER : IType<36>;
def TypeJ : IType<37>;
def TypeLD : IType<38>;
def TypeM : IType<39>;
def TypeMAPPING : IType<40>;
def TypeNCJ : IType<41>;
def TypePSEUDO : IType<42>;
def TypeST : IType<43>;
def TypeSUBINSN : IType<44>;
def TypeS_2op : IType<45>;
def TypeS_3op : IType<46>;
def TypeV2LDST : IType<49>;
def TypeV4LDST : IType<50>;
