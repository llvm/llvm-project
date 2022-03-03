//===-- MIMGInstructions.td - MIMG Instruction Definitions ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// MIMG-specific encoding families to distinguish between semantically
// equivalent machine instructions with different encoding.
//
// - MIMGEncGfx6: encoding introduced with gfx6 (obsoleted for atomics in gfx8)
// - MIMGEncGfx8: encoding introduced with gfx8 for atomics
// - MIMGEncGfx90a: encoding for gfx90a for atomics
// - MIMGEncGfx10Default: gfx10 default (non-NSA) encoding
// - MIMGEncGfx10NSA: gfx10 NSA encoding
class MIMGEncoding;

def MIMGEncGfx6 : MIMGEncoding;
def MIMGEncGfx8 : MIMGEncoding;
def MIMGEncGfx90a : MIMGEncoding;
def MIMGEncGfx10Default : MIMGEncoding;
def MIMGEncGfx10NSA : MIMGEncoding;

def MIMGEncoding : GenericEnum {
  let FilterClass = "MIMGEncoding";
}

// Represent an ISA-level opcode, independent of the encoding and the
// vdata/vaddr size.
class MIMGBaseOpcode : PredicateControl {
  MIMGBaseOpcode BaseOpcode = !cast<MIMGBaseOpcode>(NAME);
  bit Store = 0;
  bit Atomic = 0;
  bit AtomicX2 = 0; // (f)cmpswap
  bit Sampler = 0;
  bit Gather4 = 0;
  bits<8> NumExtraArgs = 0;
  bit Gradients = 0;
  bit G16 = 0;
  bit Coordinates = 1;
  bit LodOrClampOrMip = 0;
  bit HasD16 = 0;
  bit IsAtomicRet = 0;
  bit MSAA = 0;
  bit BVH = 0;
}

def MIMGBaseOpcode : GenericEnum {
  let FilterClass = "MIMGBaseOpcode";
}

def MIMGBaseOpcodesTable : GenericTable {
  let FilterClass = "MIMGBaseOpcode";
  let CppTypeName = "MIMGBaseOpcodeInfo";
  let Fields = ["BaseOpcode", "Store", "Atomic", "AtomicX2", "Sampler",
                "Gather4", "NumExtraArgs", "Gradients", "G16", "Coordinates",
                "LodOrClampOrMip", "HasD16", "MSAA", "BVH"];
  string TypeOf_BaseOpcode = "MIMGBaseOpcode";

  let PrimaryKey = ["BaseOpcode"];
  let PrimaryKeyName = "getMIMGBaseOpcodeInfo";
}

def MIMGDim : GenericEnum {
  let FilterClass = "AMDGPUDimProps";
}

def MIMGDimInfoTable : GenericTable {
  let FilterClass = "AMDGPUDimProps";
  let CppTypeName = "MIMGDimInfo";
  let Fields = ["Dim", "NumCoords", "NumGradients", "MSAA", "DA", "Encoding", "AsmSuffix"];
  string TypeOf_Dim = "MIMGDim";

  let PrimaryKey = ["Dim"];
  let PrimaryKeyName = "getMIMGDimInfo";
}

def getMIMGDimInfoByEncoding : SearchIndex {
  let Table = MIMGDimInfoTable;
  let Key = ["Encoding"];
}

def getMIMGDimInfoByAsmSuffix : SearchIndex {
  let Table = MIMGDimInfoTable;
  let Key = ["AsmSuffix"];
}

def MIMG {
  int NOP = -1;
}

class mimgopc <int base, int vi = base, int si = base> {
  field bits<8> BASE = base; // Opcode for all but atomics
  field bits<8> VI = vi; // VI is only used for atomic instructions
  field bits<8> SI = si; // SI is only used for atomic instructions
  bit HAS_BASE = !ne(base, MIMG.NOP);
  bit HAS_VI = !ne(vi, MIMG.NOP);
  bit HAS_SI = !ne(si, MIMG.NOP);
}

class MIMGLZMapping<MIMGBaseOpcode l, MIMGBaseOpcode lz> {
  MIMGBaseOpcode L = l;
  MIMGBaseOpcode LZ = lz;
}

def MIMGLZMappingTable : GenericTable {
  let FilterClass = "MIMGLZMapping";
  let CppTypeName = "MIMGLZMappingInfo";
  let Fields = ["L", "LZ"];
  string TypeOf_L = "MIMGBaseOpcode";
  string TypeOf_LZ = "MIMGBaseOpcode";

  let PrimaryKey = ["L"];
  let PrimaryKeyName = "getMIMGLZMappingInfo";
}

class MIMGMIPMapping<MIMGBaseOpcode mip, MIMGBaseOpcode nonmip> {
  MIMGBaseOpcode MIP = mip;
  MIMGBaseOpcode NONMIP = nonmip;
}

def MIMGMIPMappingTable : GenericTable {
  let FilterClass = "MIMGMIPMapping";
  let CppTypeName = "MIMGMIPMappingInfo";
  let Fields = ["MIP", "NONMIP"];
  string TypeOf_MIP = "MIMGBaseOpcode";
  string TypeOf_NONMIP = "MIMGBaseOpcode";

  let PrimaryKey = ["MIP"];
  let PrimaryKeyName = "getMIMGMIPMappingInfo";
}

class MIMGBiasMapping<MIMGBaseOpcode bias, MIMGBaseOpcode nobias> {
  MIMGBaseOpcode Bias = bias;
  MIMGBaseOpcode NoBias = nobias;
}

def MIMGBiasMappingTable : GenericTable {
  let FilterClass = "MIMGBiasMapping";
  let CppTypeName = "MIMGBiasMappingInfo";
  let Fields = ["Bias", "NoBias"];
  string TypeOf_Bias = "MIMGBaseOpcode";
  string TypeOf_NoBias = "MIMGBaseOpcode";

  let PrimaryKey = ["Bias"];
  let PrimaryKeyName = "getMIMGBiasMappingInfo";
}

class MIMGOffsetMapping<MIMGBaseOpcode offset, MIMGBaseOpcode nooffset> {
  MIMGBaseOpcode Offset = offset;
  MIMGBaseOpcode NoOffset = nooffset;
}

def MIMGOffsetMappingTable : GenericTable {
  let FilterClass = "MIMGOffsetMapping";
  let CppTypeName = "MIMGOffsetMappingInfo";
  let Fields = ["Offset", "NoOffset"];
  string TypeOf_Offset = "MIMGBaseOpcode";
  string TypeOf_NoOffset = "MIMGBaseOpcode";

  let PrimaryKey = ["Offset"];
  let PrimaryKeyName = "getMIMGOffsetMappingInfo";
}

class MIMGG16Mapping<MIMGBaseOpcode g, MIMGBaseOpcode g16> {
  MIMGBaseOpcode G = g;
  MIMGBaseOpcode G16 = g16;
}

def MIMGG16MappingTable : GenericTable {
  let FilterClass = "MIMGG16Mapping";
  let CppTypeName = "MIMGG16MappingInfo";
  let Fields = ["G", "G16"];
  string TypeOf_G = "MIMGBaseOpcode";
  string TypeOf_G16 = "MIMGBaseOpcode";

  let PrimaryKey = ["G"];
  let PrimaryKeyName = "getMIMGG16MappingInfo";
}

class MIMG_Base <dag outs, string dns = "">
  : InstSI <outs, (ins), "", []> {

  let VM_CNT = 1;
  let EXP_CNT = 1;
  let MIMG = 1;
  let Uses = [EXEC];
  let mayLoad = 1;
  let mayStore = 0;
  let SchedRW = [WriteVMEM];
  let UseNamedOperandTable = 1;
  let hasSideEffects = 0; // XXX ????

  let DecoderNamespace = dns;
  let isAsmParserOnly = !eq(dns, "");
}

class MIMG <dag outs, string dns = "">
  : MIMG_Base <outs, dns> {

  let hasPostISelHook = 1;
  let AsmMatchConverter = "cvtMIMG";

  Instruction Opcode = !cast<Instruction>(NAME);
  MIMGBaseOpcode BaseOpcode;
  MIMGEncoding MIMGEncoding;
  bits<8> VDataDwords;
  bits<8> VAddrDwords;
}

def MIMGInfoTable : GenericTable {
  let FilterClass = "MIMG";
  let CppTypeName = "MIMGInfo";
  let Fields = ["Opcode", "BaseOpcode", "MIMGEncoding", "VDataDwords", "VAddrDwords"];
  string TypeOf_BaseOpcode = "MIMGBaseOpcode";
  string TypeOf_MIMGEncoding = "MIMGEncoding";

  let PrimaryKey = ["BaseOpcode", "MIMGEncoding", "VDataDwords", "VAddrDwords"];
  let PrimaryKeyName = "getMIMGOpcodeHelper";
}

def getMIMGInfo : SearchIndex {
  let Table = MIMGInfoTable;
  let Key = ["Opcode"];
}

// This class used to use !foldl to memoize the AddrAsmNames list.
// It turned out that that was much slower than using !filter.
class MIMGNSAHelper<int num_addrs> {
  list<string> AddrAsmNames =
    !foreach(i, !filter(i, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        !lt(i, num_addrs)), "vaddr" # i);
  dag AddrIns = !dag(ins, !foreach(arg, AddrAsmNames, VGPR_32), AddrAsmNames);
  string AddrAsm = "[$" # !interleave(AddrAsmNames, ", $") # "]";

  int NSA = !if(!le(num_addrs, 1), ?,
            !if(!le(num_addrs, 5), 1,
            !if(!le(num_addrs, 9), 2,
            !if(!le(num_addrs, 13), 3, ?))));
}

// Base class of all pre-gfx10 MIMG instructions.
class MIMG_gfx6789<bits<8> op, dag outs, string dns = "">
  : MIMG<outs, dns>, MIMGe_gfx6789<op> {
  let SubtargetPredicate = isGFX6GFX7GFX8GFX9NotGFX90A;
  let AssemblerPredicate = isGFX6GFX7GFX8GFX9NotGFX90A;

  let MIMGEncoding = MIMGEncGfx6;

  let d16 = !if(BaseOpcode.HasD16, ?, 0);
}

class MIMG_gfx90a<bits<8> op, dag outs, string dns = "">
  : MIMG<outs, dns>, MIMGe_gfx90a<op> {
  let SubtargetPredicate = isGFX90APlus;
  let AssemblerPredicate = isGFX90APlus;

  let MIMGEncoding = MIMGEncGfx90a;

  let d16 = !if(BaseOpcode.HasD16, ?, 0);
}

// Base class of all non-NSA gfx10 MIMG instructions.
class MIMG_gfx10<int op, dag outs, string dns = "">
  : MIMG<outs, dns>, MIMGe_gfx10<op> {
  let SubtargetPredicate = isGFX10Plus;
  let AssemblerPredicate = isGFX10Plus;

  let MIMGEncoding = MIMGEncGfx10Default;

  let d16 = !if(BaseOpcode.HasD16, ?, 0);
  let nsa = 0;
}

// Base class for all NSA MIMG instructions.
// Note that 1-dword addresses always use non-NSA variants.
class MIMG_nsa_gfx10<int op, dag outs, int num_addrs, string dns="">
  : MIMG<outs, dns>, MIMGe_gfx10<op> {
  let SubtargetPredicate = isGFX10Plus;
  let AssemblerPredicate = isGFX10Plus;

  let MIMGEncoding = MIMGEncGfx10NSA;

  MIMGNSAHelper nsah = MIMGNSAHelper<num_addrs>;
  dag AddrIns = nsah.AddrIns;
  string AddrAsm = nsah.AddrAsm;

  let d16 = !if(BaseOpcode.HasD16, ?, 0);
  let nsa = nsah.NSA;
}

class MIMG_NoSampler_Helper <mimgopc op, string asm,
                             RegisterClass dst_rc,
                             RegisterClass addr_rc,
                             string dns="">
  : MIMG_gfx6789 <op.BASE, (outs dst_rc:$vdata), dns> {
  let InOperandList = !con((ins addr_rc:$vaddr, SReg_256:$srsrc,
                                DMask:$dmask, UNorm:$unorm, CPol:$cpol,
                                R128A16:$r128, TFE:$tfe, LWE:$lwe, DA:$da),
                           !if(BaseOpcode.HasD16, (ins D16:$d16), (ins)));
  let AsmString = asm#" $vdata, $vaddr, $srsrc$dmask$unorm$cpol$r128$tfe$lwe$da"
                      #!if(BaseOpcode.HasD16, "$d16", "");
}

class MIMG_NoSampler_Helper_gfx90a <mimgopc op, string asm,
                                    RegisterClass dst_rc,
                                    RegisterClass addr_rc,
                                    string dns="">
  : MIMG_gfx90a <op.BASE, (outs getLdStRegisterOperand<dst_rc>.ret:$vdata), dns> {
  let InOperandList = !con((ins addr_rc:$vaddr, SReg_256:$srsrc,
                                DMask:$dmask, UNorm:$unorm, CPol:$cpol,
                                R128A16:$r128, LWE:$lwe, DA:$da),
                           !if(BaseOpcode.HasD16, (ins D16:$d16), (ins)));
  let AsmString = asm#" $vdata, $vaddr, $srsrc$dmask$unorm$cpol$r128$lwe$da"
                      #!if(BaseOpcode.HasD16, "$d16", "");
}

class MIMG_NoSampler_gfx10<mimgopc op, string opcode,
                           RegisterClass DataRC, RegisterClass AddrRC,
                           string dns="">
  : MIMG_gfx10<op.BASE, (outs DataRC:$vdata), dns> {
  let InOperandList = !con((ins AddrRC:$vaddr0, SReg_256:$srsrc, DMask:$dmask,
                                Dim:$dim, UNorm:$unorm, CPol:$cpol,
                                R128A16:$r128, GFX10A16:$a16, TFE:$tfe, LWE:$lwe),
                           !if(BaseOpcode.HasD16, (ins D16:$d16), (ins)));
  let AsmString = opcode#" $vdata, $vaddr0, $srsrc$dmask$dim$unorm$cpol$r128$a16$tfe$lwe"
                    #!if(BaseOpcode.HasD16, "$d16", "");
}

class MIMG_NoSampler_nsa_gfx10<mimgopc op, string opcode,
                               RegisterClass DataRC, int num_addrs,
                               string dns="">
  : MIMG_nsa_gfx10<op.BASE, (outs DataRC:$vdata), num_addrs, dns> {
  let InOperandList = !con(AddrIns,
                           (ins SReg_256:$srsrc, DMask:$dmask,
                                Dim:$dim, UNorm:$unorm, CPol:$cpol,
                                R128A16:$r128, GFX10A16:$a16, TFE:$tfe, LWE:$lwe),
                           !if(BaseOpcode.HasD16, (ins D16:$d16), (ins)));
  let AsmString = opcode#" $vdata, "#AddrAsm#", $srsrc$dmask$dim$unorm$cpol$r128$a16$tfe$lwe"
                    #!if(BaseOpcode.HasD16, "$d16", "");
}

multiclass MIMG_NoSampler_Src_Helper <mimgopc op, string asm,
                                             RegisterClass dst_rc,
                                             bit enableDisasm,
                                             bit ExtendedImageInst = 1> {
  let ssamp = 0 in {
    let VAddrDwords = 1 in {
      if op.HAS_BASE then {
        def _V1 : MIMG_NoSampler_Helper <op, asm, dst_rc, VGPR_32,
                                         !if(enableDisasm, "AMDGPU", "")>;
        if !not(ExtendedImageInst) then
        def _V1_gfx90a : MIMG_NoSampler_Helper_gfx90a <op, asm, dst_rc, VGPR_32,
                                       !if(enableDisasm, "GFX90A", "")>;
        def _V1_gfx10 : MIMG_NoSampler_gfx10<op, asm, dst_rc, VGPR_32,
                                             !if(enableDisasm, "AMDGPU", "")>;
      }
    }

    let VAddrDwords = 2 in {
      if op.HAS_BASE then {
        def _V2 : MIMG_NoSampler_Helper <op, asm, dst_rc, VReg_64>;
        if !not(ExtendedImageInst) then
        def _V2_gfx90a : MIMG_NoSampler_Helper_gfx90a <op, asm, dst_rc, VReg_64>;
        def _V2_gfx10 : MIMG_NoSampler_gfx10<op, asm, dst_rc, VReg_64>;
        def _V2_nsa_gfx10 : MIMG_NoSampler_nsa_gfx10<op, asm, dst_rc, 2>;
      }
    }

    let VAddrDwords = 3 in {
      if op.HAS_BASE then {
        def _V3 : MIMG_NoSampler_Helper <op, asm, dst_rc, VReg_96>;
        if !not(ExtendedImageInst) then
        def _V3_gfx90a : MIMG_NoSampler_Helper_gfx90a <op, asm, dst_rc, VReg_96>;
        def _V3_gfx10 : MIMG_NoSampler_gfx10<op, asm, dst_rc, VReg_96>;
        def _V3_nsa_gfx10 : MIMG_NoSampler_nsa_gfx10<op, asm, dst_rc, 3>;
      }
    }

    let VAddrDwords = 4 in {
      if op.HAS_BASE then {
        def _V4 : MIMG_NoSampler_Helper <op, asm, dst_rc, VReg_128>;
        if !not(ExtendedImageInst) then
        def _V4_gfx90a : MIMG_NoSampler_Helper_gfx90a <op, asm, dst_rc, VReg_128>;
        def _V4_gfx10 : MIMG_NoSampler_gfx10<op, asm, dst_rc, VReg_128>;
        def _V4_nsa_gfx10 : MIMG_NoSampler_nsa_gfx10<op, asm, dst_rc, 4,
                                                     !if(enableDisasm, "AMDGPU", "")>;
      }
    }
  }
}

multiclass MIMG_NoSampler <mimgopc op, string asm, bit has_d16, bit mip = 0,
                           bit isResInfo = 0,
                           bit msaa = 0> {
  def "" : MIMGBaseOpcode {
    let Coordinates = !not(isResInfo);
    let LodOrClampOrMip = mip;
    let HasD16 = has_d16;
    let MSAA = msaa;
  }

  let BaseOpcode = !cast<MIMGBaseOpcode>(NAME),
      mayLoad = !not(isResInfo) in {
    let VDataDwords = 1 in
    defm _V1 : MIMG_NoSampler_Src_Helper <op, asm, VGPR_32, 1, msaa>;
    let VDataDwords = 2 in
    defm _V2 : MIMG_NoSampler_Src_Helper <op, asm, VReg_64, 0, msaa>;
    let VDataDwords = 3 in
    defm _V3 : MIMG_NoSampler_Src_Helper <op, asm, VReg_96, 0, msaa>;
    let VDataDwords = 4 in
    defm _V4 : MIMG_NoSampler_Src_Helper <op, asm, VReg_128, 0, msaa>;
    let VDataDwords = 5 in
    defm _V5 : MIMG_NoSampler_Src_Helper <op, asm, VReg_160, 0, msaa>;
  }
}

class MIMG_Store_Helper <mimgopc op, string asm,
                         RegisterClass data_rc,
                         RegisterClass addr_rc,
                         string dns = "">
  : MIMG_gfx6789<op.BASE, (outs), dns> {
  let InOperandList = !con((ins data_rc:$vdata, addr_rc:$vaddr, SReg_256:$srsrc,
                                DMask:$dmask, UNorm:$unorm, CPol:$cpol,
                                R128A16:$r128, TFE:$tfe, LWE:$lwe, DA:$da),
                           !if(BaseOpcode.HasD16, (ins D16:$d16), (ins)));
  let AsmString = asm#" $vdata, $vaddr, $srsrc$dmask$unorm$cpol$r128$tfe$lwe$da"
                      #!if(BaseOpcode.HasD16, "$d16", "");
}

class MIMG_Store_Helper_gfx90a <mimgopc op, string asm,
                                RegisterClass data_rc,
                                RegisterClass addr_rc,
                                string dns = "">
  : MIMG_gfx90a<op.BASE, (outs), dns> {
  let InOperandList = !con((ins getLdStRegisterOperand<data_rc>.ret:$vdata,
                                addr_rc:$vaddr, SReg_256:$srsrc,
                                DMask:$dmask, UNorm:$unorm, CPol:$cpol,
                                R128A16:$r128, LWE:$lwe, DA:$da),
                           !if(BaseOpcode.HasD16, (ins D16:$d16), (ins)));
  let AsmString = asm#" $vdata, $vaddr, $srsrc$dmask$unorm$cpol$r128$lwe$da"
                      #!if(BaseOpcode.HasD16, "$d16", "");
}

class MIMG_Store_gfx10<mimgopc op, string opcode,
                       RegisterClass DataRC, RegisterClass AddrRC,
                       string dns="">
  : MIMG_gfx10<op.BASE, (outs), dns> {
  let InOperandList = !con((ins DataRC:$vdata, AddrRC:$vaddr0, SReg_256:$srsrc,
                                DMask:$dmask, Dim:$dim, UNorm:$unorm, CPol:$cpol,
                                R128A16:$r128, GFX10A16:$a16, TFE:$tfe, LWE:$lwe),
                           !if(BaseOpcode.HasD16, (ins D16:$d16), (ins)));
  let AsmString = opcode#" $vdata, $vaddr0, $srsrc$dmask$dim$unorm$cpol$r128$a16$tfe$lwe"
                    #!if(BaseOpcode.HasD16, "$d16", "");
}

class MIMG_Store_nsa_gfx10<mimgopc op, string opcode,
                           RegisterClass DataRC, int num_addrs,
                           string dns="">
  : MIMG_nsa_gfx10<op.BASE, (outs), num_addrs, dns> {
  let InOperandList = !con((ins DataRC:$vdata),
                           AddrIns,
                           (ins SReg_256:$srsrc, DMask:$dmask,
                                Dim:$dim, UNorm:$unorm, CPol:$cpol,
                                R128A16:$r128, GFX10A16:$a16, TFE:$tfe, LWE:$lwe),
                           !if(BaseOpcode.HasD16, (ins D16:$d16), (ins)));
  let AsmString = opcode#" $vdata, "#AddrAsm#", $srsrc$dmask$dim$unorm$cpol$r128$a16$tfe$lwe"
                    #!if(BaseOpcode.HasD16, "$d16", "");
}

multiclass MIMG_Store_Addr_Helper <mimgopc op, string asm,
                                  RegisterClass data_rc,
                                  bit enableDisasm> {
  let mayLoad = 0, mayStore = 1, hasSideEffects = 0, hasPostISelHook = 0,
      DisableWQM = 1, ssamp = 0 in {
    let VAddrDwords = 1 in {
      if op.HAS_BASE then {
        def _V1 : MIMG_Store_Helper <op, asm, data_rc, VGPR_32,
                                     !if(enableDisasm, "AMDGPU", "")>;
        def _V1_gfx90a : MIMG_Store_Helper_gfx90a <op, asm, data_rc, VGPR_32,
                                     !if(enableDisasm, "GFX90A", "")>;
        def _V1_gfx10 : MIMG_Store_gfx10 <op, asm, data_rc, VGPR_32,
                                          !if(enableDisasm, "AMDGPU", "")>;
      }
    }
    let VAddrDwords = 2 in {
      if op.HAS_BASE then {
        def _V2 : MIMG_Store_Helper <op, asm, data_rc, VReg_64>;
        def _V2_gfx90a : MIMG_Store_Helper_gfx90a <op, asm, data_rc, VReg_64>;
        def _V2_gfx10 : MIMG_Store_gfx10 <op, asm, data_rc, VReg_64>;
        def _V2_nsa_gfx10 : MIMG_Store_nsa_gfx10 <op, asm, data_rc, 2>;
      }
    }
    let VAddrDwords = 3 in {
      if op.HAS_BASE then {
        def _V3 : MIMG_Store_Helper <op, asm, data_rc, VReg_96>;
        def _V3_gfx90a : MIMG_Store_Helper_gfx90a <op, asm, data_rc, VReg_96>;
        def _V3_gfx10 : MIMG_Store_gfx10 <op, asm, data_rc, VReg_96>;
        def _V3_nsa_gfx10 : MIMG_Store_nsa_gfx10 <op, asm, data_rc, 3>;
      }
    }
    let VAddrDwords = 4 in {
      if op.HAS_BASE then {
        def _V4 : MIMG_Store_Helper <op, asm, data_rc, VReg_128>;
        def _V4_gfx90a : MIMG_Store_Helper_gfx90a <op, asm, data_rc, VReg_128>;
        def _V4_gfx10 : MIMG_Store_gfx10 <op, asm, data_rc, VReg_128>;
        def _V4_nsa_gfx10 : MIMG_Store_nsa_gfx10 <op, asm, data_rc, 4,
                                                         !if(enableDisasm, "AMDGPU", "")>;
      }
    }
  }
}

multiclass MIMG_Store <mimgopc op, string asm, bit has_d16, bit mip = 0> {
  def "" : MIMGBaseOpcode {
    let Store = 1;
    let LodOrClampOrMip = mip;
    let HasD16 = has_d16;
  }

  let BaseOpcode = !cast<MIMGBaseOpcode>(NAME) in {
    let VDataDwords = 1 in
    defm _V1 : MIMG_Store_Addr_Helper <op, asm, VGPR_32, 1>;
    let VDataDwords = 2 in
    defm _V2 : MIMG_Store_Addr_Helper <op, asm, VReg_64, 0>;
    let VDataDwords = 3 in
    defm _V3 : MIMG_Store_Addr_Helper <op, asm, VReg_96, 0>;
    let VDataDwords = 4 in
    defm _V4 : MIMG_Store_Addr_Helper <op, asm, VReg_128, 0>;
    let VDataDwords = 5 in
    defm _V5 : MIMG_Store_Addr_Helper <op, asm, VReg_160, 0>;
  }
}

class MIMG_Atomic_gfx6789_base <bits<8> op, string asm, RegisterClass data_rc,
                                RegisterClass addr_rc, string dns="">
  : MIMG_gfx6789 <op, (outs data_rc:$vdst), dns> {
  let Constraints = "$vdst = $vdata";
  let AsmMatchConverter = "cvtMIMGAtomic";

  let InOperandList = (ins data_rc:$vdata, addr_rc:$vaddr, SReg_256:$srsrc,
                           DMask:$dmask, UNorm:$unorm, CPol:$cpol,
                           R128A16:$r128, TFE:$tfe, LWE:$lwe, DA:$da);
  let AsmString = asm#" $vdst, $vaddr, $srsrc$dmask$unorm$cpol$r128$tfe$lwe$da";
}

class MIMG_Atomic_gfx90a_base <bits<8> op, string asm, RegisterClass data_rc,
                               RegisterClass addr_rc, string dns="">
  : MIMG_gfx90a <op, (outs getLdStRegisterOperand<data_rc>.ret:$vdst), dns> {
  let Constraints = "$vdst = $vdata";
  let AsmMatchConverter = "cvtMIMGAtomic";

  let InOperandList = (ins getLdStRegisterOperand<data_rc>.ret:$vdata,
                           addr_rc:$vaddr, SReg_256:$srsrc,
                           DMask:$dmask, UNorm:$unorm, CPol:$cpol,
                           R128A16:$r128, LWE:$lwe, DA:$da);
  let AsmString = asm#" $vdst, $vaddr, $srsrc$dmask$unorm$cpol$r128$lwe$da";
}

class MIMG_Atomic_si<mimgopc op, string asm, RegisterClass data_rc,
                     RegisterClass addr_rc, bit enableDasm = 0>
  : MIMG_Atomic_gfx6789_base<op.SI, asm, data_rc, addr_rc,
                             !if(enableDasm, "GFX6GFX7", "")> {
  let AssemblerPredicate = isGFX6GFX7;
}

class MIMG_Atomic_vi<mimgopc op, string asm, RegisterClass data_rc,
                     RegisterClass addr_rc, bit enableDasm = 0>
  : MIMG_Atomic_gfx6789_base<op.VI, asm, data_rc, addr_rc, !if(enableDasm, "GFX8", "")> {
  let AssemblerPredicate = isGFX8GFX9NotGFX90A;
  let MIMGEncoding = MIMGEncGfx8;
}

class MIMG_Atomic_gfx90a<mimgopc op, string asm, RegisterClass data_rc,
                         RegisterClass addr_rc, bit enableDasm = 0>
  : MIMG_Atomic_gfx90a_base<op.VI, asm, data_rc, addr_rc, !if(enableDasm, "GFX90A", "")> {
  let AssemblerPredicate = isGFX90APlus;
  let MIMGEncoding = MIMGEncGfx90a;
}

class MIMG_Atomic_gfx10<mimgopc op, string opcode,
                        RegisterClass DataRC, RegisterClass AddrRC,
                        bit enableDisasm = 0>
  : MIMG_gfx10<!cast<int>(op.BASE), (outs DataRC:$vdst),
               !if(enableDisasm, "AMDGPU", "")> {
  let Constraints = "$vdst = $vdata";
  let AsmMatchConverter = "cvtMIMGAtomic";

  let InOperandList = (ins DataRC:$vdata, AddrRC:$vaddr0, SReg_256:$srsrc,
                           DMask:$dmask, Dim:$dim, UNorm:$unorm, CPol:$cpol,
                           R128A16:$r128, GFX10A16:$a16, TFE:$tfe, LWE:$lwe);
  let AsmString = opcode#" $vdst, $vaddr0, $srsrc$dmask$dim$unorm$cpol$r128$a16$tfe$lwe";
}

class MIMG_Atomic_nsa_gfx10<mimgopc op, string opcode,
                            RegisterClass DataRC, int num_addrs,
                            bit enableDisasm = 0>
  : MIMG_nsa_gfx10<!cast<int>(op.BASE), (outs DataRC:$vdst), num_addrs,
                   !if(enableDisasm, "AMDGPU", "")> {
  let Constraints = "$vdst = $vdata";
  let AsmMatchConverter = "cvtMIMGAtomic";

  let InOperandList = !con((ins DataRC:$vdata),
                           AddrIns,
                           (ins SReg_256:$srsrc, DMask:$dmask,
                                Dim:$dim, UNorm:$unorm, CPol:$cpol,
                                R128A16:$r128, GFX10A16:$a16, TFE:$tfe, LWE:$lwe));
  let AsmString = opcode#" $vdata, "#AddrAsm#", $srsrc$dmask$dim$unorm$cpol$r128$a16$tfe$lwe";
}

multiclass MIMG_Atomic_Addr_Helper_m <mimgopc op, string asm,
                                      RegisterClass data_rc,
                                      bit enableDasm = 0,
                                      bit isFP = 0> {
  let hasSideEffects = 1, // FIXME: remove this
      mayLoad = 1, mayStore = 1, hasPostISelHook = 0, DisableWQM = 1,
      ssamp = 0, FPAtomic = isFP in {
    let VAddrDwords = 1 in {
      if op.HAS_SI then {
        def _V1_si : MIMG_Atomic_si <op, asm, data_rc, VGPR_32, enableDasm>;
      }
      if op.HAS_VI then {
        def _V1_vi : MIMG_Atomic_vi <op, asm, data_rc, VGPR_32, enableDasm>;
        def _V1_gfx90a : MIMG_Atomic_gfx90a <op, asm, data_rc, VGPR_32, enableDasm>;
      }
      if op.HAS_BASE then {
        def _V1_gfx10 : MIMG_Atomic_gfx10 <op, asm, data_rc, VGPR_32, enableDasm>;
      }
    }
    let VAddrDwords = 2 in {
      if op.HAS_SI then {
        def _V2_si : MIMG_Atomic_si <op, asm, data_rc, VReg_64, 0>;
      }
      if op.HAS_VI then {
        def _V2_vi : MIMG_Atomic_vi <op, asm, data_rc, VReg_64, 0>;
        def _V2_gfx90a : MIMG_Atomic_gfx90a <op, asm, data_rc, VReg_64, 0>;
      }
      if op.HAS_BASE then {
        def _V2_gfx10 : MIMG_Atomic_gfx10 <op, asm, data_rc, VReg_64, 0>;
        def _V2_nsa_gfx10 : MIMG_Atomic_nsa_gfx10 <op, asm, data_rc, 2, 0>;
      }
    }
    let VAddrDwords = 3 in {
      if op.HAS_SI then {
        def _V3_si : MIMG_Atomic_si <op, asm, data_rc, VReg_96, 0>;
      }
      if op.HAS_VI then {
        def _V3_vi : MIMG_Atomic_vi <op, asm, data_rc, VReg_96, 0>;
        def _V3_gfx90a : MIMG_Atomic_gfx90a <op, asm, data_rc, VReg_96, 0>;
      }
      if op.HAS_BASE then {
        def _V3_gfx10 : MIMG_Atomic_gfx10 <op, asm, data_rc, VReg_96, 0>;
        def _V3_nsa_gfx10 : MIMG_Atomic_nsa_gfx10 <op, asm, data_rc, 3, 0>;
      }
    }
    let VAddrDwords = 4 in {
      if op.HAS_SI then {
        def _V4_si : MIMG_Atomic_si <op, asm, data_rc, VReg_128, 0>;
      }
      if op.HAS_VI then {
        def _V4_vi : MIMG_Atomic_vi <op, asm, data_rc, VReg_128, 0>;
        def _V4_gfx90a : MIMG_Atomic_gfx90a <op, asm, data_rc, VReg_128, 0>;
      }
      if op.HAS_BASE then {
        def _V4_gfx10 : MIMG_Atomic_gfx10 <op, asm, data_rc, VReg_128, 0>;
        def _V4_nsa_gfx10 : MIMG_Atomic_nsa_gfx10 <op, asm, data_rc, 4, enableDasm>;
      }
    }
  }
}

multiclass MIMG_Atomic <mimgopc op, string asm, bit isCmpSwap = 0, bit isFP = 0> { // 64-bit atomics
  let IsAtomicRet = 1 in {
    def "" : MIMGBaseOpcode {
      let Atomic = 1;
      let AtomicX2 = isCmpSwap;
    }

    let BaseOpcode = !cast<MIMGBaseOpcode>(NAME) in {
      // _V* variants have different dst size, but the size is encoded implicitly,
      // using dmask and tfe. Only 32-bit variant is registered with disassembler.
      // Other variants are reconstructed by disassembler using dmask and tfe.
      let VDataDwords = !if(isCmpSwap, 2, 1) in
      defm _V1 : MIMG_Atomic_Addr_Helper_m <op, asm, !if(isCmpSwap, VReg_64, VGPR_32), 1, isFP>;
      let VDataDwords = !if(isCmpSwap, 4, 2) in
      defm _V2 : MIMG_Atomic_Addr_Helper_m <op, asm, !if(isCmpSwap, VReg_128, VReg_64), 0, isFP>;
    }
  } // End IsAtomicRet = 1
}

class MIMG_Sampler_Helper <mimgopc op, string asm, RegisterClass dst_rc,
                           RegisterClass src_rc, string dns="">
  : MIMG_gfx6789 <op.BASE, (outs dst_rc:$vdata), dns> {
  let InOperandList = !con((ins src_rc:$vaddr, SReg_256:$srsrc, SReg_128:$ssamp,
                                DMask:$dmask, UNorm:$unorm, CPol:$cpol,
                                R128A16:$r128, TFE:$tfe, LWE:$lwe, DA:$da),
                           !if(BaseOpcode.HasD16, (ins D16:$d16), (ins)));
  let AsmString = asm#" $vdata, $vaddr, $srsrc, $ssamp$dmask$unorm$cpol$r128$tfe$lwe$da"
                      #!if(BaseOpcode.HasD16, "$d16", "");
}

class MIMG_Sampler_gfx90a<mimgopc op, string asm, RegisterClass dst_rc,
                          RegisterClass src_rc, string dns="">
  : MIMG_gfx90a<op.BASE, (outs getLdStRegisterOperand<dst_rc>.ret:$vdata), dns> {
  let InOperandList = !con((ins src_rc:$vaddr, SReg_256:$srsrc, SReg_128:$ssamp,
                                DMask:$dmask, UNorm:$unorm, CPol:$cpol,
                                R128A16:$r128, LWE:$lwe, DA:$da),
                           !if(BaseOpcode.HasD16, (ins D16:$d16), (ins)));
  let AsmString = asm#" $vdata, $vaddr, $srsrc, $ssamp$dmask$unorm$cpol$r128$lwe$da"
                      #!if(BaseOpcode.HasD16, "$d16", "");
}

class MIMG_Sampler_gfx10<mimgopc op, string opcode,
                         RegisterClass DataRC, RegisterClass AddrRC,
                         string dns="">
  : MIMG_gfx10<op.BASE, (outs DataRC:$vdata), dns> {
  let InOperandList = !con((ins AddrRC:$vaddr0, SReg_256:$srsrc, SReg_128:$ssamp,
                                DMask:$dmask, Dim:$dim, UNorm:$unorm, CPol:$cpol,
                                R128A16:$r128, GFX10A16:$a16, TFE:$tfe, LWE:$lwe),
                           !if(BaseOpcode.HasD16, (ins D16:$d16), (ins)));
  let AsmString = opcode#" $vdata, $vaddr0, $srsrc, $ssamp$dmask$dim$unorm"
                    #"$cpol$r128$a16$tfe$lwe"
                    #!if(BaseOpcode.HasD16, "$d16", "");
}

class MIMG_Sampler_nsa_gfx10<mimgopc op, string opcode,
                             RegisterClass DataRC, int num_addrs,
                             string dns="">
  : MIMG_nsa_gfx10<op.BASE, (outs DataRC:$vdata), num_addrs, dns> {
  let InOperandList = !con(AddrIns,
                           (ins SReg_256:$srsrc, SReg_128:$ssamp, DMask:$dmask,
                                Dim:$dim, UNorm:$unorm, CPol:$cpol,
                                R128A16:$r128, GFX10A16:$a16, TFE:$tfe, LWE:$lwe),
                           !if(BaseOpcode.HasD16, (ins D16:$d16), (ins)));
  let AsmString = opcode#" $vdata, "#AddrAsm#", $srsrc, $ssamp$dmask$dim$unorm"
                    #"$cpol$r128$a16$tfe$lwe"
                    #!if(BaseOpcode.HasD16, "$d16", "");
}

class MIMGAddrSize<int dw, bit enable_disasm> {
  int NumWords = dw;

  RegisterClass RegClass = !if(!le(NumWords, 0), ?,
                           !if(!eq(NumWords, 1), VGPR_32,
                           !if(!eq(NumWords, 2), VReg_64,
                           !if(!eq(NumWords, 3), VReg_96,
                           !if(!eq(NumWords, 4), VReg_128,
                           !if(!eq(NumWords, 5), VReg_160,
                           !if(!eq(NumWords, 6), VReg_192,
                           !if(!eq(NumWords, 7), VReg_224,
                           !if(!le(NumWords, 8), VReg_256,
                           !if(!le(NumWords, 16), VReg_512, ?))))))))));

  // Whether the instruction variant with this vaddr size should be enabled for
  // the auto-generated disassembler.
  bit Disassemble = enable_disasm;
}

// Return whether x is in lst.
class isIntInList<int x, list<int> lst> {
  bit ret = !foldl(0, lst, lhs, y, !or(lhs, !eq(x, y)));
}

// Return whether a value inside the range [min, max] (endpoints inclusive)
// is in the given list.
class isRangeInList<int min, int max, list<int> lst> {
  bit ret = !foldl(0, lst, lhs, y, !or(lhs, !and(!le(min, y), !le(y, max))));
}

class MIMGAddrSizes_dw_range<list<int> range> {
  int Min = !head(range);
  int Max = !if(!empty(!tail(range)), Min, !head(!tail(range)));
}

class MIMG_Sampler_AddrSizes<AMDGPUSampleVariant sample> {
  // List of all possible numbers of address words, taking all combinations of
  // A16 and image dimension into account (note: no MSAA, since this is for
  // sample/gather ops).
  list<int> AllNumAddrWords =
    !foreach(dw, !if(sample.Gradients,
                     !if(!eq(sample.LodOrClamp, ""),
                         [2, 3, 4, 5, 6, 7, 8, 9],
                         [2, 3, 4, 5, 6, 7, 8, 9, 10]),
                     !if(!eq(sample.LodOrClamp, ""),
                         [1, 2, 3],
                         [1, 2, 3, 4])),
             !add(dw, !size(sample.ExtraAddrArgs)));

  // Generate machine instructions based on possible register classes for the
  // required numbers of address words. The disassembler defaults to the
  // smallest register class.
  list<MIMGAddrSize> MachineInstrs =
    !foldl([]<MIMGAddrSize>,
           !foreach(range,
                    // V4 is generated for V3 and V4
                    // V8 is generated for V5 through V8
                    // V16 is generated for V9 through V16
                    [[1],[2],[3],[3,4],[5],[6],[7],[5,8],[9,16]],
                    MIMGAddrSizes_dw_range<range>),
           lhs, dw,
           !if(isRangeInList<dw.Min, dw.Max, AllNumAddrWords>.ret,
               !listconcat(lhs, [MIMGAddrSize<dw.Max, !empty(lhs)>]),
               lhs));

  // For NSA, generate machine instructions for all possible numbers of words
  // except 1 (which is already covered by the non-NSA case).
  // The disassembler defaults to the largest number of arguments among the
  // variants with the same number of NSA words, and custom code then derives
  // the exact variant based on the sample variant and the image dimension.
  list<MIMGAddrSize> NSAInstrs =
    !foldl([]<MIMGAddrSize>, [[12, 11, 10], [9, 8, 7, 6], [5, 4, 3, 2]], prev, nsa_group,
           !listconcat(prev,
                       !foldl([]<MIMGAddrSize>, nsa_group, lhs, dw,
                              !if(isIntInList<dw, AllNumAddrWords>.ret,
                                  !listconcat(lhs, [MIMGAddrSize<dw, !empty(lhs)>]),
                                  lhs))));
}

multiclass MIMG_Sampler_Src_Helper <mimgopc op, string asm,
                                    AMDGPUSampleVariant sample, RegisterClass dst_rc,
                                    bit enableDisasm = 0,
                                    bit ExtendedImageInst = 1> {
  foreach addr = MIMG_Sampler_AddrSizes<sample>.MachineInstrs in {
    let VAddrDwords = addr.NumWords in {
      if op.HAS_BASE then {
        def _V # addr.NumWords
          : MIMG_Sampler_Helper <op, asm, dst_rc, addr.RegClass,
                                 !if(!and(enableDisasm, addr.Disassemble), "AMDGPU", "")>;
        if !not(ExtendedImageInst) then
        def _V # addr.NumWords # _gfx90a
          : MIMG_Sampler_gfx90a <op, asm, dst_rc, addr.RegClass,
                                 !if(!and(enableDisasm, addr.Disassemble), "GFX90A", "")>;
        def _V # addr.NumWords # _gfx10
          : MIMG_Sampler_gfx10 <op, asm, dst_rc, addr.RegClass,
                                 !if(!and(enableDisasm, addr.Disassemble), "AMDGPU", "")>;
      }
    }
  }

  foreach addr = MIMG_Sampler_AddrSizes<sample>.NSAInstrs in {
    let VAddrDwords = addr.NumWords in {
      if op.HAS_BASE then {
        def _V # addr.NumWords # _nsa_gfx10
          : MIMG_Sampler_nsa_gfx10<op, asm, dst_rc, addr.NumWords,
                                   !if(!and(enableDisasm, addr.Disassemble), "AMDGPU", "")>;
      }
    }
  }
}

class MIMG_Sampler_BaseOpcode<AMDGPUSampleVariant sample>
  : MIMGBaseOpcode {
  let Sampler = 1;
  let NumExtraArgs = !size(sample.ExtraAddrArgs);
  let Gradients = sample.Gradients;
  let LodOrClampOrMip = !ne(sample.LodOrClamp, "");
}

multiclass MIMG_Sampler <mimgopc op, AMDGPUSampleVariant sample, bit wqm = 0,
                         bit isG16 = 0, bit isGetLod = 0,
                         string asm = "image_sample"#sample.LowerCaseMod#!if(isG16, "_g16", ""),
                         bit ExtendedImageInst = !ne(sample.LowerCaseMod, "")> {
  def "" : MIMG_Sampler_BaseOpcode<sample> {
    let HasD16 = !not(isGetLod);
    let G16 = isG16;
  }

  let BaseOpcode = !cast<MIMGBaseOpcode>(NAME), WQM = wqm,
      mayLoad = !not(isGetLod) in {
    let VDataDwords = 1 in
    defm _V1 : MIMG_Sampler_Src_Helper<op, asm, sample, VGPR_32, 1, ExtendedImageInst>;
    let VDataDwords = 2 in
    defm _V2 : MIMG_Sampler_Src_Helper<op, asm, sample, VReg_64, 0, ExtendedImageInst>;
    let VDataDwords = 3 in
    defm _V3 : MIMG_Sampler_Src_Helper<op, asm, sample, VReg_96, 0, ExtendedImageInst>;
    let VDataDwords = 4 in
    defm _V4 : MIMG_Sampler_Src_Helper<op, asm, sample, VReg_128, 0, ExtendedImageInst>;
    let VDataDwords = 5 in
    defm _V5 : MIMG_Sampler_Src_Helper<op, asm, sample, VReg_160, 0, ExtendedImageInst>;
  }
}

multiclass MIMG_Sampler_WQM <mimgopc op, AMDGPUSampleVariant sample>
    : MIMG_Sampler<op, sample, 1>;

multiclass MIMG_Gather <mimgopc op, AMDGPUSampleVariant sample, bit wqm = 0,
                        string asm = "image_gather4"#sample.LowerCaseMod> {
  def "" : MIMG_Sampler_BaseOpcode<sample> {
    let HasD16 = 1;
    let Gather4 = 1;
  }

  let BaseOpcode = !cast<MIMGBaseOpcode>(NAME), WQM = wqm,
      Gather4 = 1 in {
    let VDataDwords = 2 in
    defm _V2 : MIMG_Sampler_Src_Helper<op, asm, sample, VReg_64>; /* for packed D16 only */
    let VDataDwords = 4 in
    defm _V4 : MIMG_Sampler_Src_Helper<op, asm, sample, VReg_128, 1>;
    let VDataDwords = 5 in
    defm _V5 : MIMG_Sampler_Src_Helper<op, asm, sample, VReg_160>;
  }
}

multiclass MIMG_Gather_WQM <mimgopc op, AMDGPUSampleVariant sample>
    : MIMG_Gather<op, sample, 1>;

class MIMG_IntersectRay_Helper<bit Is64, bit A16> {
  int num_addrs = !if(Is64, !if(A16, 9, 12), !if(A16, 8, 11));
  // TODO: MIMGAddrSize will choose VReg_512 which is a 16 register tuple,
  // when we only need 9, 11 or 12 depending on A16 field and ptr size.
  RegisterClass RegClass = MIMGAddrSize<num_addrs, 0>.RegClass;
  int VAddrDwords = !srl(RegClass.Size, 5);
}

class MIMG_IntersectRay_gfx10<mimgopc op, string opcode, RegisterClass AddrRC, bit A16>
    : MIMG_gfx10<op.BASE, (outs VReg_128:$vdata), "AMDGPU"> {

  let InOperandList = !con((ins AddrRC:$vaddr0, SReg_128:$srsrc),
                           !if(A16, (ins GFX10A16:$a16), (ins)));
  let AsmString = opcode#" $vdata, $vaddr0, $srsrc"#!if(A16, "$a16", "");

  let nsa = 0;
}

class MIMG_IntersectRay_nsa_gfx10<mimgopc op, string opcode, int num_addrs, bit A16>
    : MIMG_nsa_gfx10<op.BASE, (outs VReg_128:$vdata), num_addrs, "AMDGPU"> {
  let InOperandList = !con(nsah.AddrIns,
                           (ins SReg_128:$srsrc),
                           !if(A16, (ins GFX10A16:$a16), (ins)));
  let AsmString = opcode#" $vdata, "#nsah.AddrAsm#", $srsrc"#!if(A16, "$a16", "");
}

multiclass MIMG_IntersectRay<mimgopc op, string opcode, bit Is64, bit A16> {
  defvar info = MIMG_IntersectRay_Helper<Is64, A16>;
  def "" : MIMGBaseOpcode {
    let BVH = 1;
  }
  let AsmMatchConverter = !if(A16, "cvtIntersectRay", ""),
      dmask = 0xf,
      unorm = 1,
      d16 = 0,
      cpol = 0,
      tfe = 0,
      lwe = 0,
      r128 = 1,
      ssamp = 0,
      dim = {0, 0, 0},
      a16 = A16,
      d16 = 0,
      BaseOpcode = !cast<MIMGBaseOpcode>(NAME),
      VDataDwords = 4 in {
    def _sa_gfx10 : MIMG_IntersectRay_gfx10<op, opcode, info.RegClass, A16> {
      let VAddrDwords = info.VAddrDwords;
    }
    def _nsa_gfx10 : MIMG_IntersectRay_nsa_gfx10<op, opcode, info.num_addrs, A16> {
      let VAddrDwords = info.num_addrs;
    }
  }
}

//===----------------------------------------------------------------------===//
// MIMG Instructions
//===----------------------------------------------------------------------===//
let OtherPredicates = [HasImageInsts] in {

defm IMAGE_LOAD                 : MIMG_NoSampler <mimgopc<0x00>, "image_load", 1>;
defm IMAGE_LOAD_MIP             : MIMG_NoSampler <mimgopc<0x01>, "image_load_mip", 1, 1>;
defm IMAGE_LOAD_PCK             : MIMG_NoSampler <mimgopc<0x02>, "image_load_pck", 0>;
defm IMAGE_LOAD_PCK_SGN         : MIMG_NoSampler <mimgopc<0x03>, "image_load_pck_sgn", 0>;
defm IMAGE_LOAD_MIP_PCK         : MIMG_NoSampler <mimgopc<0x04>, "image_load_mip_pck", 0, 1>;
defm IMAGE_LOAD_MIP_PCK_SGN     : MIMG_NoSampler <mimgopc<0x05>, "image_load_mip_pck_sgn", 0, 1>;
defm IMAGE_STORE                : MIMG_Store <mimgopc<0x08>, "image_store", 1>;
defm IMAGE_STORE_MIP            : MIMG_Store <mimgopc<0x09>, "image_store_mip", 1, 1>;
defm IMAGE_STORE_PCK            : MIMG_Store <mimgopc<0x0a>, "image_store_pck", 0>;
defm IMAGE_STORE_MIP_PCK        : MIMG_Store <mimgopc<0x0b>, "image_store_mip_pck", 0, 1>;

defm IMAGE_GET_RESINFO          : MIMG_NoSampler <mimgopc<0x0e>, "image_get_resinfo", 0, 1, 1>;

defm IMAGE_ATOMIC_SWAP          : MIMG_Atomic <mimgopc<0x0f, 0x10, 0x0f>, "image_atomic_swap">;
defm IMAGE_ATOMIC_CMPSWAP       : MIMG_Atomic <mimgopc<0x10, 0x11, 0x10>, "image_atomic_cmpswap", 1>;
defm IMAGE_ATOMIC_ADD           : MIMG_Atomic <mimgopc<0x11, 0x12, 0x11>, "image_atomic_add">;
defm IMAGE_ATOMIC_SUB           : MIMG_Atomic <mimgopc<0x12, 0x13, 0x12>, "image_atomic_sub">;
defm IMAGE_ATOMIC_RSUB          : MIMG_Atomic <mimgopc<MIMG.NOP, MIMG.NOP, 0x13>, "image_atomic_rsub">;
defm IMAGE_ATOMIC_SMIN          : MIMG_Atomic <mimgopc<0x14>, "image_atomic_smin">;
defm IMAGE_ATOMIC_UMIN          : MIMG_Atomic <mimgopc<0x15>, "image_atomic_umin">;
defm IMAGE_ATOMIC_SMAX          : MIMG_Atomic <mimgopc<0x16>, "image_atomic_smax">;
defm IMAGE_ATOMIC_UMAX          : MIMG_Atomic <mimgopc<0x17>, "image_atomic_umax">;
defm IMAGE_ATOMIC_AND           : MIMG_Atomic <mimgopc<0x18>, "image_atomic_and">;
defm IMAGE_ATOMIC_OR            : MIMG_Atomic <mimgopc<0x19>, "image_atomic_or">;
defm IMAGE_ATOMIC_XOR           : MIMG_Atomic <mimgopc<0x1a>, "image_atomic_xor">;
defm IMAGE_ATOMIC_INC           : MIMG_Atomic <mimgopc<0x1b>, "image_atomic_inc">;
defm IMAGE_ATOMIC_DEC           : MIMG_Atomic <mimgopc<0x1c>, "image_atomic_dec">;
defm IMAGE_ATOMIC_FCMPSWAP      : MIMG_Atomic <mimgopc<0x1d, MIMG.NOP>, "image_atomic_fcmpswap", 1, 1>;
defm IMAGE_ATOMIC_FMIN          : MIMG_Atomic <mimgopc<0x1e, MIMG.NOP>, "image_atomic_fmin", 0, 1>;
defm IMAGE_ATOMIC_FMAX          : MIMG_Atomic <mimgopc<0x1f, MIMG.NOP>, "image_atomic_fmax", 0, 1>;

defm IMAGE_SAMPLE               : MIMG_Sampler_WQM <mimgopc<0x20>, AMDGPUSample>;
let OtherPredicates = [HasExtendedImageInsts] in {
defm IMAGE_SAMPLE_CL            : MIMG_Sampler_WQM <mimgopc<0x21>, AMDGPUSample_cl>;
defm IMAGE_SAMPLE_D             : MIMG_Sampler <mimgopc<0x22>, AMDGPUSample_d>;
defm IMAGE_SAMPLE_D_CL          : MIMG_Sampler <mimgopc<0x23>, AMDGPUSample_d_cl>;
defm IMAGE_SAMPLE_D_G16         : MIMG_Sampler <mimgopc<0xa2>, AMDGPUSample_d, 0, 1>;
defm IMAGE_SAMPLE_D_CL_G16      : MIMG_Sampler <mimgopc<0xa3>, AMDGPUSample_d_cl, 0, 1>;
defm IMAGE_SAMPLE_L             : MIMG_Sampler <mimgopc<0x24>, AMDGPUSample_l>;
defm IMAGE_SAMPLE_B             : MIMG_Sampler_WQM <mimgopc<0x25>, AMDGPUSample_b>;
defm IMAGE_SAMPLE_B_CL          : MIMG_Sampler_WQM <mimgopc<0x26>, AMDGPUSample_b_cl>;
defm IMAGE_SAMPLE_LZ            : MIMG_Sampler <mimgopc<0x27>, AMDGPUSample_lz>;
defm IMAGE_SAMPLE_C             : MIMG_Sampler_WQM <mimgopc<0x28>, AMDGPUSample_c>;
defm IMAGE_SAMPLE_C_CL          : MIMG_Sampler_WQM <mimgopc<0x29>, AMDGPUSample_c_cl>;
defm IMAGE_SAMPLE_C_D           : MIMG_Sampler <mimgopc<0x2a>, AMDGPUSample_c_d>;
defm IMAGE_SAMPLE_C_D_CL        : MIMG_Sampler <mimgopc<0x2b>, AMDGPUSample_c_d_cl>;
defm IMAGE_SAMPLE_C_D_G16       : MIMG_Sampler <mimgopc<0xaa>, AMDGPUSample_c_d, 0, 1>;
defm IMAGE_SAMPLE_C_D_CL_G16    : MIMG_Sampler <mimgopc<0xab>, AMDGPUSample_c_d_cl, 0, 1>;
defm IMAGE_SAMPLE_C_L           : MIMG_Sampler <mimgopc<0x2c>, AMDGPUSample_c_l>;
defm IMAGE_SAMPLE_C_B           : MIMG_Sampler_WQM <mimgopc<0x2d>, AMDGPUSample_c_b>;
defm IMAGE_SAMPLE_C_B_CL        : MIMG_Sampler_WQM <mimgopc<0x2e>, AMDGPUSample_c_b_cl>;
defm IMAGE_SAMPLE_C_LZ          : MIMG_Sampler <mimgopc<0x2f>, AMDGPUSample_c_lz>;
defm IMAGE_SAMPLE_O             : MIMG_Sampler_WQM <mimgopc<0x30>, AMDGPUSample_o>;
defm IMAGE_SAMPLE_CL_O          : MIMG_Sampler_WQM <mimgopc<0x31>, AMDGPUSample_cl_o>;
defm IMAGE_SAMPLE_D_O           : MIMG_Sampler <mimgopc<0x32>, AMDGPUSample_d_o>;
defm IMAGE_SAMPLE_D_CL_O        : MIMG_Sampler <mimgopc<0x33>, AMDGPUSample_d_cl_o>;
defm IMAGE_SAMPLE_D_O_G16       : MIMG_Sampler <mimgopc<0xb2>, AMDGPUSample_d_o, 0, 1>;
defm IMAGE_SAMPLE_D_CL_O_G16    : MIMG_Sampler <mimgopc<0xb3>, AMDGPUSample_d_cl_o, 0, 1>;
defm IMAGE_SAMPLE_L_O           : MIMG_Sampler <mimgopc<0x34>, AMDGPUSample_l_o>;
defm IMAGE_SAMPLE_B_O           : MIMG_Sampler_WQM <mimgopc<0x35>, AMDGPUSample_b_o>;
defm IMAGE_SAMPLE_B_CL_O        : MIMG_Sampler_WQM <mimgopc<0x36>, AMDGPUSample_b_cl_o>;
defm IMAGE_SAMPLE_LZ_O          : MIMG_Sampler <mimgopc<0x37>, AMDGPUSample_lz_o>;
defm IMAGE_SAMPLE_C_O           : MIMG_Sampler_WQM <mimgopc<0x38>, AMDGPUSample_c_o>;
defm IMAGE_SAMPLE_C_CL_O        : MIMG_Sampler_WQM <mimgopc<0x39>, AMDGPUSample_c_cl_o>;
defm IMAGE_SAMPLE_C_D_O         : MIMG_Sampler <mimgopc<0x3a>, AMDGPUSample_c_d_o>;
defm IMAGE_SAMPLE_C_D_CL_O      : MIMG_Sampler <mimgopc<0x3b>, AMDGPUSample_c_d_cl_o>;
defm IMAGE_SAMPLE_C_D_O_G16     : MIMG_Sampler <mimgopc<0xba>, AMDGPUSample_c_d_o, 0, 1>;
defm IMAGE_SAMPLE_C_D_CL_O_G16  : MIMG_Sampler <mimgopc<0xbb>, AMDGPUSample_c_d_cl_o, 0, 1>;
defm IMAGE_SAMPLE_C_L_O         : MIMG_Sampler <mimgopc<0x3c>, AMDGPUSample_c_l_o>;
defm IMAGE_SAMPLE_C_B_CL_O      : MIMG_Sampler_WQM <mimgopc<0x3e>, AMDGPUSample_c_b_cl_o>;
defm IMAGE_SAMPLE_C_B_O         : MIMG_Sampler_WQM <mimgopc<0x3d>, AMDGPUSample_c_b_o>;
defm IMAGE_SAMPLE_C_LZ_O        : MIMG_Sampler <mimgopc<0x3f>, AMDGPUSample_c_lz_o>;
defm IMAGE_GATHER4              : MIMG_Gather_WQM <mimgopc<0x40>, AMDGPUSample>;
defm IMAGE_GATHER4_CL           : MIMG_Gather_WQM <mimgopc<0x41>, AMDGPUSample_cl>;
defm IMAGE_GATHER4_L            : MIMG_Gather <mimgopc<0x44>, AMDGPUSample_l>;
defm IMAGE_GATHER4_B            : MIMG_Gather_WQM <mimgopc<0x45>, AMDGPUSample_b>;
defm IMAGE_GATHER4_B_CL         : MIMG_Gather_WQM <mimgopc<0x46>, AMDGPUSample_b_cl>;
defm IMAGE_GATHER4_LZ           : MIMG_Gather <mimgopc<0x47>, AMDGPUSample_lz>;
defm IMAGE_GATHER4_C            : MIMG_Gather_WQM <mimgopc<0x48>, AMDGPUSample_c>;
defm IMAGE_GATHER4_C_CL         : MIMG_Gather_WQM <mimgopc<0x49>, AMDGPUSample_c_cl>;
defm IMAGE_GATHER4_C_L          : MIMG_Gather <mimgopc<0x4c>, AMDGPUSample_c_l>;
defm IMAGE_GATHER4_C_B          : MIMG_Gather_WQM <mimgopc<0x4d>, AMDGPUSample_c_b>;
defm IMAGE_GATHER4_C_B_CL       : MIMG_Gather_WQM <mimgopc<0x4e>, AMDGPUSample_c_b_cl>;
defm IMAGE_GATHER4_C_LZ         : MIMG_Gather <mimgopc<0x4f>, AMDGPUSample_c_lz>;
defm IMAGE_GATHER4_O            : MIMG_Gather_WQM <mimgopc<0x50>, AMDGPUSample_o>;
defm IMAGE_GATHER4_CL_O         : MIMG_Gather_WQM <mimgopc<0x51>, AMDGPUSample_cl_o>;
defm IMAGE_GATHER4_L_O          : MIMG_Gather <mimgopc<0x54>, AMDGPUSample_l_o>;
defm IMAGE_GATHER4_B_O          : MIMG_Gather_WQM <mimgopc<0x55>, AMDGPUSample_b_o>;
defm IMAGE_GATHER4_B_CL_O       : MIMG_Gather <mimgopc<0x56>, AMDGPUSample_b_cl_o>;
defm IMAGE_GATHER4_LZ_O         : MIMG_Gather <mimgopc<0x57>, AMDGPUSample_lz_o>;
defm IMAGE_GATHER4_C_O          : MIMG_Gather_WQM <mimgopc<0x58>, AMDGPUSample_c_o>;
defm IMAGE_GATHER4_C_CL_O       : MIMG_Gather_WQM <mimgopc<0x59>, AMDGPUSample_c_cl_o>;
defm IMAGE_GATHER4_C_L_O        : MIMG_Gather <mimgopc<0x5c>, AMDGPUSample_c_l_o>;
defm IMAGE_GATHER4_C_B_O        : MIMG_Gather_WQM <mimgopc<0x5d>, AMDGPUSample_c_b_o>;
defm IMAGE_GATHER4_C_B_CL_O     : MIMG_Gather_WQM <mimgopc<0x5e>, AMDGPUSample_c_b_cl_o>;
defm IMAGE_GATHER4_C_LZ_O       : MIMG_Gather <mimgopc<0x5f>, AMDGPUSample_c_lz_o>;
//defm IMAGE_GATHER4H             : MIMG_Gather_WQM <mimgopc<0x61>, ?>;

defm IMAGE_GET_LOD              : MIMG_Sampler <mimgopc<0x60>, AMDGPUSample, 1, 0, 1, "image_get_lod">;

defm IMAGE_SAMPLE_CD            : MIMG_Sampler <mimgopc<0x68>, AMDGPUSample_cd>;
defm IMAGE_SAMPLE_CD_CL         : MIMG_Sampler <mimgopc<0x69>, AMDGPUSample_cd_cl>;
defm IMAGE_SAMPLE_C_CD          : MIMG_Sampler <mimgopc<0x6a>, AMDGPUSample_c_cd>;
defm IMAGE_SAMPLE_C_CD_CL       : MIMG_Sampler <mimgopc<0x6b>, AMDGPUSample_c_cd_cl>;
defm IMAGE_SAMPLE_CD_O          : MIMG_Sampler <mimgopc<0x6c>, AMDGPUSample_cd_o>;
defm IMAGE_SAMPLE_CD_CL_O       : MIMG_Sampler <mimgopc<0x6d>, AMDGPUSample_cd_cl_o>;
defm IMAGE_SAMPLE_C_CD_O        : MIMG_Sampler <mimgopc<0x6e>, AMDGPUSample_c_cd_o>;
defm IMAGE_SAMPLE_C_CD_CL_O     : MIMG_Sampler <mimgopc<0x6f>, AMDGPUSample_c_cd_cl_o>;
defm IMAGE_SAMPLE_CD_G16        : MIMG_Sampler <mimgopc<0xe8>, AMDGPUSample_cd, 0, 1>;
defm IMAGE_SAMPLE_CD_CL_G16     : MIMG_Sampler <mimgopc<0xe9>, AMDGPUSample_cd_cl, 0, 1>;
defm IMAGE_SAMPLE_C_CD_G16      : MIMG_Sampler <mimgopc<0xea>, AMDGPUSample_c_cd, 0, 1>;
defm IMAGE_SAMPLE_C_CD_CL_G16   : MIMG_Sampler <mimgopc<0xeb>, AMDGPUSample_c_cd_cl, 0, 1>;
defm IMAGE_SAMPLE_CD_O_G16      : MIMG_Sampler <mimgopc<0xec>, AMDGPUSample_cd_o, 0, 1>;
defm IMAGE_SAMPLE_CD_CL_O_G16   : MIMG_Sampler <mimgopc<0xed>, AMDGPUSample_cd_cl_o, 0, 1>;
defm IMAGE_SAMPLE_C_CD_O_G16    : MIMG_Sampler <mimgopc<0xee>, AMDGPUSample_c_cd_o, 0, 1>;
defm IMAGE_SAMPLE_C_CD_CL_O_G16 : MIMG_Sampler <mimgopc<0xef>, AMDGPUSample_c_cd_cl_o, 0, 1>;
} // End OtherPredicates = [HasExtendedImageInsts]
//def IMAGE_RSRC256 : MIMG_NoPattern_RSRC256 <"image_rsrc256", 0x0000007e>;
//def IMAGE_SAMPLER : MIMG_NoPattern_ <"image_sampler", 0x0000007f>;

let OtherPredicates = [HasGFX10_AEncoding] in
defm IMAGE_MSAA_LOAD_X : MIMG_NoSampler <mimgopc<0x80>, "image_msaa_load", 1, 0, 0, 1>;

let OtherPredicates = [HasGFX10_AEncoding] in {
defm IMAGE_BVH_INTERSECT_RAY       : MIMG_IntersectRay<mimgopc<0xe6>, "image_bvh_intersect_ray", 0, 0>;
defm IMAGE_BVH_INTERSECT_RAY_a16   : MIMG_IntersectRay<mimgopc<0xe6>, "image_bvh_intersect_ray", 0, 1>;
defm IMAGE_BVH64_INTERSECT_RAY     : MIMG_IntersectRay<mimgopc<0xe7>, "image_bvh64_intersect_ray", 1, 0>;
defm IMAGE_BVH64_INTERSECT_RAY_a16 : MIMG_IntersectRay<mimgopc<0xe7>, "image_bvh64_intersect_ray", 1, 1>;
} // End OtherPredicates = [HasGFX10_AEncoding]

} // End let OtherPredicates = [HasImageInsts]

/********** ========================================= **********/
/********** Table of dimension-aware image intrinsics **********/
/********** ========================================= **********/

class ImageDimIntrinsicInfo<AMDGPUImageDimIntrinsic I> {
  Intrinsic Intr = I;
  MIMGBaseOpcode BaseOpcode = !cast<MIMGBaseOpcode>(!strconcat("IMAGE_", I.P.OpMod));
  AMDGPUDimProps Dim = I.P.Dim;
  AMDGPUImageDimIntrinsicEval DimEval = AMDGPUImageDimIntrinsicEval<I.P>;

  bits<8> NumOffsetArgs = DimEval.NumOffsetArgs;
  bits<8> NumBiasArgs = DimEval.NumBiasArgs;
  bits<8> NumZCompareArgs = DimEval.NumZCompareArgs;
  bits<8> NumGradients = DimEval.NumGradientArgs;
  bits<8> NumDmask = DimEval.NumDmaskArgs;
  bits<8> NumData = DimEval.NumDataArgs;
  bits<8> NumVAddrs = DimEval.NumVAddrArgs;
  bits<8> NumArgs = !add(DimEval.CachePolicyArgIndex, 1);

  bits<8> DMaskIndex = DimEval.DmaskArgIndex;
  bits<8> VAddrStart = DimEval.VAddrArgIndex;
  bits<8> OffsetIndex = DimEval.OffsetArgIndex;
  bits<8> BiasIndex = DimEval.BiasArgIndex;
  bits<8> ZCompareIndex = DimEval.ZCompareArgIndex;
  bits<8> GradientStart = DimEval.GradientArgIndex;
  bits<8> CoordStart = DimEval.CoordArgIndex;
  bits<8> LodIndex = DimEval.LodArgIndex;
  bits<8> MipIndex = DimEval.MipArgIndex;
  bits<8> VAddrEnd = !add(DimEval.VAddrArgIndex, DimEval.NumVAddrArgs);
  bits<8> RsrcIndex = DimEval.RsrcArgIndex;
  bits<8> SampIndex = DimEval.SampArgIndex;
  bits<8> UnormIndex = DimEval.UnormArgIndex;
  bits<8> TexFailCtrlIndex = DimEval.TexFailCtrlArgIndex;
  bits<8> CachePolicyIndex = DimEval.CachePolicyArgIndex;

  bits<8> BiasTyArg = !add(I.P.NumRetAndDataAnyTypes,
    !if(!eq(NumOffsetArgs, 0), 0, I.P.ExtraAddrArgs[0].Type.isAny));
  bits<8> GradientTyArg = !add(I.P.NumRetAndDataAnyTypes,
    !foldl(0, I.P.ExtraAddrArgs, cnt, arg, !add(cnt, arg.Type.isAny)));
  bits<8> CoordTyArg = !add(GradientTyArg, !if(I.P.Gradients, 1, 0));
}

def ImageDimIntrinsicTable : GenericTable {
  let FilterClass = "ImageDimIntrinsicInfo";
  let Fields = ["Intr", "BaseOpcode", "Dim", "NumOffsetArgs", "NumBiasArgs", "NumZCompareArgs", "NumGradients", "NumDmask", "NumData", "NumVAddrs", "NumArgs",
    "DMaskIndex", "VAddrStart", "OffsetIndex", "BiasIndex", "ZCompareIndex", "GradientStart", "CoordStart", "LodIndex", "MipIndex", "VAddrEnd",
    "RsrcIndex", "SampIndex", "UnormIndex", "TexFailCtrlIndex", "CachePolicyIndex",
    "BiasTyArg", "GradientTyArg", "CoordTyArg"];
  string TypeOf_BaseOpcode = "MIMGBaseOpcode";
  string TypeOf_Dim = "MIMGDim";

  let PrimaryKey = ["Intr"];
  let PrimaryKeyName = "getImageDimIntrinsicInfo";
  let PrimaryKeyEarlyOut = 1;
}

def getImageDimIntrinsicByBaseOpcode : SearchIndex {
  let Table = ImageDimIntrinsicTable;
  let Key = ["BaseOpcode", "Dim"];
}

foreach intr = !listconcat(AMDGPUImageDimIntrinsics,
                           AMDGPUImageDimAtomicIntrinsics) in {
  def : ImageDimIntrinsicInfo<intr>;
}

// L to LZ Optimization Mapping
def : MIMGLZMapping<IMAGE_SAMPLE_L, IMAGE_SAMPLE_LZ>;
def : MIMGLZMapping<IMAGE_SAMPLE_C_L, IMAGE_SAMPLE_C_LZ>;
def : MIMGLZMapping<IMAGE_SAMPLE_L_O, IMAGE_SAMPLE_LZ_O>;
def : MIMGLZMapping<IMAGE_SAMPLE_C_L_O, IMAGE_SAMPLE_C_LZ_O>;
def : MIMGLZMapping<IMAGE_GATHER4_L, IMAGE_GATHER4_LZ>;
def : MIMGLZMapping<IMAGE_GATHER4_C_L, IMAGE_GATHER4_C_LZ>;
def : MIMGLZMapping<IMAGE_GATHER4_L_O, IMAGE_GATHER4_LZ_O>;
def : MIMGLZMapping<IMAGE_GATHER4_C_L_O, IMAGE_GATHER4_C_LZ_O>;

// MIP to NONMIP Optimization Mapping
def : MIMGMIPMapping<IMAGE_LOAD_MIP, IMAGE_LOAD>;
def : MIMGMIPMapping<IMAGE_STORE_MIP, IMAGE_STORE>;

// Bias to NoBias Optimization Mapping
def : MIMGBiasMapping<IMAGE_SAMPLE_B, IMAGE_SAMPLE>;
def : MIMGBiasMapping<IMAGE_SAMPLE_B_CL, IMAGE_SAMPLE_CL>;
def : MIMGBiasMapping<IMAGE_SAMPLE_C_B, IMAGE_SAMPLE_C>;
def : MIMGBiasMapping<IMAGE_SAMPLE_C_B_CL, IMAGE_SAMPLE_C_CL>;
def : MIMGBiasMapping<IMAGE_SAMPLE_B_O, IMAGE_SAMPLE_O>;
def : MIMGBiasMapping<IMAGE_SAMPLE_B_CL_O, IMAGE_SAMPLE_CL_O>;
def : MIMGBiasMapping<IMAGE_SAMPLE_C_B_O, IMAGE_SAMPLE_C_O>;
def : MIMGBiasMapping<IMAGE_SAMPLE_C_B_CL_O, IMAGE_SAMPLE_C_CL_O>;
def : MIMGBiasMapping<IMAGE_GATHER4_B, IMAGE_GATHER4>;
def : MIMGBiasMapping<IMAGE_GATHER4_B_CL, IMAGE_GATHER4_CL>;
def : MIMGBiasMapping<IMAGE_GATHER4_C_B, IMAGE_GATHER4_C>;
def : MIMGBiasMapping<IMAGE_GATHER4_C_B_CL, IMAGE_GATHER4_C_CL>;
def : MIMGBiasMapping<IMAGE_GATHER4_B_O, IMAGE_GATHER4_O>;
def : MIMGBiasMapping<IMAGE_GATHER4_B_CL_O, IMAGE_GATHER4_CL_O>;
def : MIMGBiasMapping<IMAGE_GATHER4_C_B_O, IMAGE_GATHER4_C_O>;
def : MIMGBiasMapping<IMAGE_GATHER4_C_B_CL_O, IMAGE_GATHER4_C_CL_O>;

// Offset to NoOffset Optimization Mapping
def : MIMGOffsetMapping<IMAGE_SAMPLE_O, IMAGE_SAMPLE>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_CL_O, IMAGE_SAMPLE_CL>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_D_O, IMAGE_SAMPLE_D>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_D_CL_O, IMAGE_SAMPLE_D_CL>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_D_O_G16, IMAGE_SAMPLE_D_G16>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_D_CL_O_G16, IMAGE_SAMPLE_D_CL_G16>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_L_O, IMAGE_SAMPLE_L>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_B_O, IMAGE_SAMPLE_B>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_B_CL_O, IMAGE_SAMPLE_B_CL>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_LZ_O, IMAGE_SAMPLE_LZ>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_C_O, IMAGE_SAMPLE_C>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_C_CL_O, IMAGE_SAMPLE_C_CL>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_C_D_O, IMAGE_SAMPLE_C_D>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_C_D_CL_O, IMAGE_SAMPLE_C_D_CL>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_C_D_O_G16, IMAGE_SAMPLE_C_D_G16>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_C_D_CL_O_G16, IMAGE_SAMPLE_C_D_CL_G16>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_C_L_O, IMAGE_SAMPLE_C_L>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_C_B_CL_O, IMAGE_SAMPLE_C_B_CL>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_C_B_O, IMAGE_SAMPLE_C_B>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_C_LZ_O, IMAGE_SAMPLE_C_LZ>;
def : MIMGOffsetMapping<IMAGE_GATHER4_O, IMAGE_GATHER4>;
def : MIMGOffsetMapping<IMAGE_GATHER4_CL_O, IMAGE_GATHER4_CL>;
def : MIMGOffsetMapping<IMAGE_GATHER4_L_O, IMAGE_GATHER4_L>;
def : MIMGOffsetMapping<IMAGE_GATHER4_B_O, IMAGE_GATHER4_B>;
def : MIMGOffsetMapping<IMAGE_GATHER4_B_CL_O, IMAGE_GATHER4_B_CL>;
def : MIMGOffsetMapping<IMAGE_GATHER4_LZ_O, IMAGE_GATHER4_LZ>;
def : MIMGOffsetMapping<IMAGE_GATHER4_C_O, IMAGE_GATHER4_C>;
def : MIMGOffsetMapping<IMAGE_GATHER4_C_CL_O, IMAGE_GATHER4_C_CL>;
def : MIMGOffsetMapping<IMAGE_GATHER4_C_L_O, IMAGE_GATHER4_C_L>;
def : MIMGOffsetMapping<IMAGE_GATHER4_C_B_O, IMAGE_GATHER4_C_B>;
def : MIMGOffsetMapping<IMAGE_GATHER4_C_B_CL_O, IMAGE_GATHER4_C_B_CL>;
def : MIMGOffsetMapping<IMAGE_GATHER4_C_LZ_O, IMAGE_GATHER4_C_LZ>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_CD_O, IMAGE_SAMPLE_CD>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_CD_CL_O, IMAGE_SAMPLE_CD_CL>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_C_CD_O, IMAGE_SAMPLE_C_CD>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_C_CD_CL_O, IMAGE_SAMPLE_C_CD_CL>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_CD_O_G16, IMAGE_SAMPLE_CD_G16>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_CD_CL_O_G16, IMAGE_SAMPLE_CD_CL_G16>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_C_CD_O_G16, IMAGE_SAMPLE_C_CD_G16>;
def : MIMGOffsetMapping<IMAGE_SAMPLE_C_CD_CL_O_G16, IMAGE_SAMPLE_C_CD_CL_G16>;

// G to G16 Optimization Mapping
def : MIMGG16Mapping<IMAGE_SAMPLE_D, IMAGE_SAMPLE_D_G16>;
def : MIMGG16Mapping<IMAGE_SAMPLE_D_CL, IMAGE_SAMPLE_D_CL_G16>;
def : MIMGG16Mapping<IMAGE_SAMPLE_C_D, IMAGE_SAMPLE_C_D_G16>;
def : MIMGG16Mapping<IMAGE_SAMPLE_C_D_CL, IMAGE_SAMPLE_C_D_CL_G16>;
def : MIMGG16Mapping<IMAGE_SAMPLE_D_O, IMAGE_SAMPLE_D_O_G16>;
def : MIMGG16Mapping<IMAGE_SAMPLE_D_CL_O, IMAGE_SAMPLE_D_CL_O_G16>;
def : MIMGG16Mapping<IMAGE_SAMPLE_C_D_O, IMAGE_SAMPLE_C_D_O_G16>;
def : MIMGG16Mapping<IMAGE_SAMPLE_C_D_CL_O, IMAGE_SAMPLE_C_D_CL_O_G16>;
def : MIMGG16Mapping<IMAGE_SAMPLE_CD, IMAGE_SAMPLE_CD_G16>;
def : MIMGG16Mapping<IMAGE_SAMPLE_CD_CL, IMAGE_SAMPLE_CD_CL_G16>;
def : MIMGG16Mapping<IMAGE_SAMPLE_C_CD, IMAGE_SAMPLE_C_CD_G16>;
def : MIMGG16Mapping<IMAGE_SAMPLE_C_CD_CL, IMAGE_SAMPLE_C_CD_CL_G16>;
def : MIMGG16Mapping<IMAGE_SAMPLE_CD_O, IMAGE_SAMPLE_CD_O_G16>;
def : MIMGG16Mapping<IMAGE_SAMPLE_CD_CL_O, IMAGE_SAMPLE_CD_CL_O_G16>;
def : MIMGG16Mapping<IMAGE_SAMPLE_C_CD_O, IMAGE_SAMPLE_C_CD_O_G16>;
def : MIMGG16Mapping<IMAGE_SAMPLE_C_CD_CL_O, IMAGE_SAMPLE_C_CD_CL_O_G16>;
