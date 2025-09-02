//===-- RISCVInstrInfoXsfmm.td - SiFive matrix multiply ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the Xsfmm* vendor extensions defined by SiFive.
//
//===----------------------------------------------------------------------===//

def XSfmmVTypeAsmOperand : AsmOperandClass {
  let Name = "XSfmmVType";
  let ParserMethod = "parseXSfmmVType";
  let DiagnosticType = "InvalidXSfmmVType";
  let RenderMethod = "addVTypeIOperands";
}

def XSfmmVTypeOp : RISCVOp {
  let ParserMatchClass = XSfmmVTypeAsmOperand;
  let PrintMethod = "printXSfmmVType";
  let OperandType = "OPERAND_XSFMM_VTYPE";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    if (!isUInt<32>(Imm))
      return false;
    return RISCVVType::isValidXSfmmVType(Imm);
  }];
}

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
class SFInstSetSingle<dag outs, dag ins, bits<5> rs2, string opcodestr,
                      string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatI> {
  bits<5> rs1;
  bits<5> rd;

  let Inst{31-25} = 0b1000010;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = OPCFG.Value;
  let Inst{11-7} = rd;
  let Inst{6-0} = OPC_OP_V.Value;

  let Defs = [VTYPE, VL];
}

class SFInstTileMemOp<dag outs, dag ins, bits<3> nf, RISCVOpcode opcode,
                      string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> rs2;
  bits<5> rs1;

  let Inst{31-29} = nf;
  let Inst{28} = 1;
  let Inst{27-26} = MOPLDUnitStride.Value;
  let Inst{25} = 1;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b111;
  let Inst{11-7} = 0b00000;
  let Inst{6-0} = opcode.Value;

  let Uses = [VTYPE, VL];
  let ReadsPastVL = 1;
}

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
class SFInstTileLoad<bits<3> nf, string opcodestr>
    : SFInstTileMemOp<(outs), (ins GPR:$rs2, GPRMemZeroOffset:$rs1), nf,
                      OPC_LOAD_FP, opcodestr, "$rs2, ${rs1}">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
class SFInstTileStore<bits<3> nf, string opcodestr>
    : SFInstTileMemOp<(outs), (ins GPR:$rs2, GPRMemZeroOffset:$rs1), nf,
                      OPC_STORE_FP, opcodestr, "$rs2, ${rs1}">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class SFInstTileMoveOp<bits<6> funct6, dag outs, dag ins, string opcodestr,
                       string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> rs2;
  bits<5> rs1;
  bits<5> vd;

  let Inst{31-26} = funct6;
  let Inst{25} = 1;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = OPMVX.Value;
  let Inst{11-7} = vd;
  let Inst{6-0} = OPC_OP_V.Value;

  let Uses = [VTYPE, VL];
  let ReadsPastVL = 1;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class SFInstMatmulF<dag outs, dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<5> vs1;
  bits<4> rd;

  let Inst{31-26} = 0b111100;
  let Inst{25} = 1;
  let Inst{24-20} = vs2;
  let Inst{19-15} = vs1;
  let Inst{14-12} = OPFVV.Value;
  let Inst{11-9} = rd{3-1};
  let Inst{8-7} = 0b00;
  let Inst{6-0} = OPC_OP_VE.Value;

  let Uses = [VTYPE, VL];
  let ReadsPastVL = 1;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class SFInstMatmulF8<bit a, bit b, dag outs, dag ins,
                     string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<5> vs1;
  bits<4> rd;

  let Inst{31-27} = 0b11111;
  let Inst{26} = a;
  let Inst{25} = 1;
  let Inst{24-20} = vs2;
  let Inst{19-15} = vs1;
  let Inst{14-12} = OPFVV.Value;
  let Inst{11-10} = rd{3-2};
  let Inst{9-8} = 0b00;
  let Inst{7} = b;
  let Inst{6-0} = OPC_OP_VE.Value;

  let Uses = [VTYPE, VL];
  let ReadsPastVL = 1;
}


class F8Encode<bit encoding, string name> {
  bit Encoding = encoding;
  string Name = name;
}

defvar F8Encodes = [F8Encode<0b0, "e5m2">,
                    F8Encode<0b1, "e4m3">];

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class SFInstMatmulI8<bit funct6_1, bit a, bit b, dag outs, dag ins,
                     string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<5> vs1;
  bits<4> rd;

  let Inst{31-28} = 0b1111;
  let Inst{27} = funct6_1;
  let Inst{26} = a;
  let Inst{25} = 1;
  let Inst{24-20} = vs2;
  let Inst{19-15} = vs1;
  let Inst{14-12} = OPIVV.Value;
  let Inst{11-10} = rd{3-2};
  let Inst{9-8} = 0b00;
  let Inst{7} = b;
  let Inst{6-0} = OPC_OP_VE.Value;

  let Uses = [VTYPE, VL];
  let ReadsPastVL = 1;
}

class I8Encode<bit encoding, string name> {
  bit Encoding = encoding;
  string Name = name;
}

defvar I8Encodes = [I8Encode<0, "u">,
                    I8Encode<1, "s">];

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class SFInstSetZero<dag outs, dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<5> vs1;
  bits<4> rd;

  let Inst{31-26} = 0b010000;
  let Inst{25} = 1;
  let Inst{24-20} = 0b11110;
  let Inst{19-15} = 0b00000;
  let Inst{14-12} = OPMVX.Value;
  let Inst{11-8} = rd;
  let Inst{7} = 0;
  let Inst{6-0} = OPC_OP_V.Value;

  let Uses = [VTYPE, VL];
}

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
class SFInstVtDiscard<string opcodestr>
    : RVInst<(outs), (ins), opcodestr, "", [], InstFormatR> {
  let Inst{31-26} = 0b010000;
  let Inst{25} = 1;
  let Inst{24-20} = 0b11100;
  let Inst{19-15} = 0b00000;
  let Inst{14-12} = OPMVX.Value;
  let Inst{11-7} = 0b00000;
  let Inst{6-0} = OPC_OP_V.Value;
}

let Predicates = [HasVendorXSfmmbase] in
def : InstAlias<"sf.vsettnt $rd, $rs1, $vtypei",
                (VSETVLI GPR:$rd, GPR:$rs1, XSfmmVTypeOp:$vtypei)>;

let DecoderNamespace = "XSfvector" in {

let Predicates = [HasVendorXSfmmbase] in {
  def SF_VSETTN : SFInstSetSingle<(outs GPR:$rd), (ins GPR:$rs1), 0b00000,
                                  "sf.vsettn", "$rd, $rs1">;
  def SF_VSETTM : SFInstSetSingle<(outs GPR:$rd), (ins GPR:$rs1), 0b00001,
                                  "sf.vsettm", "$rd, $rs1">;
  def SF_VSETTK : SFInstSetSingle<(outs GPR:$rd), (ins GPR:$rs1), 0b00010,
                                   "sf.vsettk", "$rd, $rs1">;
  def SF_VTDISCARD : SFInstVtDiscard<"sf.vtdiscard">;

  def SF_VTMV_V_T : SFInstTileMoveOp<0b010000, (outs VR:$vd), (ins GPR:$rs1),
                                     "sf.vtmv.v.t", "$vd, $rs1"> {
    let rs2 = 0b11111;
  }
  def SF_VTMV_T_V : SFInstTileMoveOp<0b010111, (outs), (ins GPR:$rs1, VR:$rs2),
                                     "sf.vtmv.t.v", "$rs1, $rs2"> {
    let vd = 0b00000;
  }

  def SF_VTZERO_T : SFInstSetZero<(outs), (ins TR:$rd), "sf.vtzero.t", "$rd">;

  def SF_VLTE8  : SFInstTileLoad<0b000, "sf.vlte8">;
  def SF_VLTE16 : SFInstTileLoad<0b001, "sf.vlte16">;
  def SF_VLTE32 : SFInstTileLoad<0b010, "sf.vlte32">;
  def SF_VLTE64 : SFInstTileLoad<0b011, "sf.vlte64">;

  def SF_VSTE8  : SFInstTileStore<0b000, "sf.vste8">;
  def SF_VSTE16 : SFInstTileStore<0b001, "sf.vste16">;
  def SF_VSTE32 : SFInstTileStore<0b010, "sf.vste32">;
  def SF_VSTE64 : SFInstTileStore<0b011, "sf.vste64">;
} // Predicates = [HasVendorXSfmmbase]

let Predicates = [HasVendorXSfmm32a16fOrXSfmm32a32fOrXSfmm64a64f] in {
  let Uses = [FRM], mayRaiseFPException = true in
  def SF_MM_F_F   : SFInstMatmulF<(outs), (ins TRM2:$rd, VR:$vs2, VR:$vs1),
                                  "sf.mm.f.f", "$rd, $vs2, $vs1">;
} // Predicates = [HasVendorXSfmm32a16fOrXSfmm64a32fOrXSfmm64a64f]

let Predicates = [HasVendorXSfmm32a8i] in {
  foreach a = I8Encodes in
    foreach b = I8Encodes in
      def SF_MM_#!toupper(a.Name)#_#!toupper(b.Name)
          : SFInstMatmulI8<0, a.Encoding, b.Encoding,
                           (outs), (ins TRM4:$rd, VR:$vs2, VR:$vs1),
                           "sf.mm."#a.Name#"."#b.Name, "$rd, $vs2, $vs1">;
} // Predicates = [HasVendorXSfmm32a8i]

let Predicates = [HasVendorXSfmm32a8f] in {
let Uses = [FRM], mayRaiseFPException = true in {
  foreach a = F8Encodes in
    foreach b = F8Encodes in
      def SF_MM_#!toupper(a.Name)#_#!toupper(b.Name)
          : SFInstMatmulF8<a.Encoding, b.Encoding,
                           (outs), (ins TRM4:$rd, VR:$vs2, VR:$vs1),
                           "sf.mm."#a.Name#"."#b.Name, "$rd, $vs2, $vs1">;
}
} // Predicates = [HasVendorXSfmm32a8f]

} // DecoderNamespace = "XSfvector"
