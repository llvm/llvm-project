//===--- HexagonDepITypes.td ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

class IType<bits<6> t> { bits<6> Value = t; }
def TypeALU32_2op : IType<0>;
def TypeALU32_3op : IType<1>;
def TypeALU32_ADDI : IType<2>;
def TypeALU64 : IType<3>;
def TypeCJ : IType<4>;
def TypeCOPROC_VMEM : IType<5>;
def TypeCR : IType<7>;
def TypeCVI_HIST : IType<10>;
def TypeCVI_VA : IType<16>;
def TypeCVI_VA_DV : IType<17>;
def TypeCVI_VINLANESAT : IType<18>;
def TypeCVI_VM_CUR_LD : IType<19>;
def TypeCVI_VM_LD : IType<20>;
def TypeCVI_VM_NEW_ST : IType<21>;
def TypeCVI_VM_ST : IType<22>;
def TypeCVI_VM_STU : IType<23>;
def TypeCVI_VM_TMP_LD : IType<24>;
def TypeCVI_VM_VP_LDU : IType<25>;
def TypeCVI_VP : IType<26>;
def TypeCVI_VP_VS : IType<27>;
def TypeCVI_VS : IType<28>;
def TypeCVI_VX : IType<30>;
def TypeCVI_VX_DV : IType<31>;
def TypeDUPLEX : IType<32>;
def TypeENDLOOP : IType<33>;
def TypeEXTENDER : IType<34>;
def TypeJ : IType<35>;
def TypeLD : IType<36>;
def TypeM : IType<37>;
def TypeMAPPING : IType<38>;
def TypeNCJ : IType<39>;
def TypePSEUDO : IType<40>;
def TypeST : IType<41>;
def TypeSUBINSN : IType<42>;
def TypeS_2op : IType<43>;
def TypeS_3op : IType<44>;
def TypeV2LDST : IType<47>;
def TypeV4LDST : IType<48>;
