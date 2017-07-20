//===-- ARMSystemRegister.td - ARM Register defs -------------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

include "llvm/TableGen/SearchableTable.td"

//===----------------------------------------------------------------------===//
//  Declarations that describe the ARM system-registers
//===----------------------------------------------------------------------===//

// M-Class System Registers.
// 'Mask' bits create unique keys for searches.
//
class MClassSysReg<bits<1> UniqMask1,
                   bits<1> UniqMask2,
                   bits<1> UniqMask3,
                   bits<12> Enc12,
                   string name> : SearchableTable {
  let SearchableFields = ["Name", "M1Encoding12", "M2M3Encoding8", "Encoding"];
  string Name;
  bits<13> M1Encoding12;
  bits<10> M2M3Encoding8;
  bits<12> Encoding;

  let Name = name;
  let EnumValueField = "M1Encoding12";
  let EnumValueField = "M2M3Encoding8";
  let EnumValueField = "Encoding";

  let M1Encoding12{12}    = UniqMask1;
  let M1Encoding12{11-00} = Enc12;
  let Encoding            = Enc12;

  let M2M3Encoding8{9}    = UniqMask2;
  let M2M3Encoding8{8}    = UniqMask3;
  let M2M3Encoding8{7-0}  = Enc12{7-0};
  code Requires           = [{ {} }];
}

// [|i|e|x]apsr_nzcvq has alias [|i|e|x]apsr.
//                 Mask1 Mask2 Mask3 Enc12, Name
let Requires = [{ {ARM::FeatureDSP} }] in {
def : MClassSysReg<0,    0,    0,    0x400, "apsr_g">;
def : MClassSysReg<0,    1,    1,    0xc00, "apsr_nzcvqg">;
def : MClassSysReg<0,    0,    0,    0x401, "iapsr_g">;
def : MClassSysReg<0,    1,    1,    0xc01, "iapsr_nzcvqg">;
def : MClassSysReg<0,    0,    0,    0x402, "eapsr_g">;
def : MClassSysReg<0,    1,    1,    0xc02, "eapsr_nzcvqg">;
def : MClassSysReg<0,    0,    0,    0x403, "xpsr_g">;
def : MClassSysReg<0,    1,    1,    0xc03, "xpsr_nzcvqg">;
}

def : MClassSysReg<0,    0,    1,    0x800, "apsr">;
def : MClassSysReg<1,    1,    0,    0x800, "apsr_nzcvq">;
def : MClassSysReg<0,    0,    1,    0x801, "iapsr">;
def : MClassSysReg<1,    1,    0,    0x801, "iapsr_nzcvq">;
def : MClassSysReg<0,    0,    1,    0x802, "eapsr">;
def : MClassSysReg<1,    1,    0,    0x802, "eapsr_nzcvq">;
def : MClassSysReg<0,    0,    1,    0x803, "xpsr">;
def : MClassSysReg<1,    1,    0,    0x803, "xpsr_nzcvq">;

def : MClassSysReg<0,    0,    1,    0x805, "ipsr">;
def : MClassSysReg<0,    0,    1,    0x806, "epsr">;
def : MClassSysReg<0,    0,    1,    0x807, "iepsr">;
def : MClassSysReg<0,    0,    1,    0x808, "msp">;
def : MClassSysReg<0,    0,    1,    0x809, "psp">;

let Requires = [{ {ARM::HasV8MBaselineOps} }] in {
def : MClassSysReg<0,    0,    1,    0x80a, "msplim">;
def : MClassSysReg<0,    0,    1,    0x80b, "psplim">;
}

def : MClassSysReg<0,    0,    1,    0x810, "primask">;

let Requires = [{ {ARM::HasV7Ops} }] in {
def : MClassSysReg<0,    0,    1,    0x811, "basepri">;
def : MClassSysReg<0,    0,    1,    0x812, "basepri_max">;
def : MClassSysReg<0,    0,    1,    0x813, "faultmask">;
}

def : MClassSysReg<0,    0,    1,    0x814, "control">;

let Requires = [{ {ARM::Feature8MSecExt} }] in {
def : MClassSysReg<0,    0,    1,    0x888, "msp_ns">;
def : MClassSysReg<0,    0,    1,    0x889, "psp_ns">;
}

let Requires = [{ {ARM::Feature8MSecExt, ARM::HasV8MBaselineOps} }] in {
def : MClassSysReg<0,    0,    1,    0x88a, "msplim_ns">;
def : MClassSysReg<0,    0,    1,    0x88b, "psplim_ns">;
}

def : MClassSysReg<0,    0,    1,    0x890, "primask_ns">;

let Requires = [{ {ARM::Feature8MSecExt, ARM::HasV7Ops} }] in {
def : MClassSysReg<0,    0,    1,    0x891, "basepri_ns">;
def : MClassSysReg<0,    0,    1,    0x893, "faultmask_ns">;
}

let Requires = [{ {ARM::Feature8MSecExt} }] in {
def : MClassSysReg<0,    0,    1,    0x894, "control_ns">;
def : MClassSysReg<0,    0,    1,    0x898, "sp_ns">;
}
