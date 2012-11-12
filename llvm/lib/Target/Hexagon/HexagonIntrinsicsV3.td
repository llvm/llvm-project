//=- HexagonIntrinsicsV3.td - Target Description for Hexagon -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Hexagon V3 Compiler Intrinsics in TableGen format.
//
//===----------------------------------------------------------------------===//




// MTYPE / COMPLEX / Vector reduce complex multiply real or imaginary.
def Hexagon_M2_vrcmpys_s1:
  di_MInst_disi_s1_sat            <"vrcmpys",  int_hexagon_M2_vrcmpys_s1>;
def Hexagon_M2_vrcmpys_acc_s1:
  di_MInst_didisi_acc_s1_sat      <"vrcmpys",  int_hexagon_M2_vrcmpys_acc_s1>;
def Hexagon_M2_vrcmpys_s1rp:
  si_MInst_disi_s1_rnd_sat        <"vrcmpys",  int_hexagon_M2_vrcmpys_s1rp>;




/********************************************************************
*            MTYPE/VB                                               *
*********************************************************************/

// MTYPE / VB / Vector reduce add unsigned bytes.
def Hexagon_M2_vradduh:
  si_MInst_didi                   <"vradduh",  int_hexagon_M2_vradduh>;


/********************************************************************
*            ALU64/ALU                                              *
*********************************************************************/

// ALU64 / ALU / Add.
def Hexagon_A2_addsp:
  di_ALU64_sidi                   <"add",      int_hexagon_A2_addsp>;
def Hexagon_A2_addpsat:
  di_ALU64_didi                   <"add",      int_hexagon_A2_addpsat>;

def Hexagon_A2_maxp:
  di_ALU64_didi                   <"max",      int_hexagon_A2_maxp>;
def Hexagon_A2_maxup:
  di_ALU64_didi                   <"maxu",     int_hexagon_A2_maxup>;
