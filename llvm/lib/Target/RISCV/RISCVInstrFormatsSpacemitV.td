//===----------------------------------------------------------------------===//
//
//  This file describes the RISC-V SPACEMIT RVV extension instruction formats.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RISC-V SPACEMIT RVV class templates
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// Operand definitions.
//===----------------------------------------------------------------------===//

class SMTVDotOpcode<bits<7> val> { bits<7> Value = val; }
def OPMMA : SMTVDotOpcode<0b1110001>;
def OPMMA_SLIDE : SMTVDotOpcode<0b1110011>;

//===----------------------------------------------------------------------===//
// Vector Dot-Product Sign Encoding
// Defines the signed/unsigned mixing modes for vector dot-product operations.
// Encoding format: [1:0] bits
//   00: UU (Unsigned x Unsigned)
//   01: US (Unsigned x Signed)
//   10: SU (Signed x Unsigned)
//   11: SS (Signed x Signed)
//===----------------------------------------------------------------------===//

class VTypeEncode<bits<2> encoding, string name> {
  bits<2> Encoding = encoding;
  string Name = name;
}
defvar VTypeEncodes = [VTypeEncode<0b00, "u">,
                       VTypeEncode<0b01, "us">,
                       VTypeEncode<0b10, "su">,
                       VTypeEncode<0b11, "">];

//===----------------------------------------------------------------------===//
// Vector Dot-Product Sliding Window Modes
// Encoding format: [1:0] bits
//   00: Slide1 (1-element sliding stride)
//   01: Slide2 (2-element sliding stride)
//   10: Slide3 (3-element sliding stride)
//   11: Reserved
//
// Used in sliding-window dot-product operations:
//   vd = vs1 • vs2.slide{1|2|3}  // • = dot product
//===----------------------------------------------------------------------===//
class SlideEncode<bits<2> encoding, string name> {
  bits<2> Encoding = encoding;
  string Name = name;
}

defvar SlideEncodes = [SlideEncode<0b00, "1">,
                       SlideEncode<0b01, "2">,
                       SlideEncode<0b10, "3">];

class HPEncode<bits<3> encoding, string name> {
  bits<3> Encoding = encoding;
  string Name = name;
}
defvar HPEncodes = [HPEncode<0b011, "u">,
                    HPEncode<0b110, "us">,
                    HPEncode<0b101, "su">,
                    HPEncode<0b100, "">];

// The SpacemiT support element Type
class SMTVType<bit isFP> : AsmOperandClass {
  let Name = "SMTVType"#isFP;
  let RenderMethod = "addSMTVTypeOperand";
  let ParserMethod = "parseSMTVType";
  let DiagnosticType = "InvalidSMTType"#isFP;
  let DiagnosticString = !if(isFP,
      "This Inst only supports bfp16 and fp16",
      "This Inst only supports i4 and i8, i8 is the default type"
  );
  let IsOptional = 1;
  let DefaultMethod = "defaultSMTVType";
}

class SMT_VTYPE<bit isFP> : Operand<XLenVT> {
  let ParserMatchClass = SMTVType<isFP>;
  let PrintMethod = "printSMTVType";
  let DecoderMethod = "decodeUImmOperand<2>";
  let OperandType = "OPERAND_SMTVType";
  let OperandNamespace = "RISCVOp";
}

// Supports i4 and i8
def SMT_INT : SMT_VTYPE<0>;
// Supports bfp16 and fp16
def SMT_FP  : SMT_VTYPE<1>;

def SMTI8 : AsmOperandClass {
  let Name = "SMTI8";
  let RenderMethod = "addSMTVTypeOperand";
  let DiagnosticType = "InvalidSMTI8";
  let DiagnosticString = "smt.vmadot with slide only supports i8 type";
  let ParserMethod = "parseSMTVType";
  let IsOptional = 1;
  let DefaultMethod = "defaultSMTVType";
}

def SMT_I8  : Operand<XLenVT> {
  let ParserMatchClass = SMTI8;
  let PrintMethod = "printSMTVType";
  let DecoderMethod = "decodeUImmOperand<2>";
  let OperandType = "OPERAND_SMTI8";
  let OperandNamespace = "RISCVOp";
}

//===----------------------------------------------------------------------===//
// Instruction formats
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  // Base vector dot product (no slide) format.
  class SMTVMADot<bits<2> sign, string opcodestr, bit EarlyClobber = 0>
      : RVInst<(outs VRM2:$vd_wb), (ins VRM2:$vd, VR:$vs1, VR:$vs2),
               opcodestr, "$vd, $vs1, $vs2", [], InstFormatR> {
    bits<5> vd;
    bits<5> vs1;
    bits<5> vs2;

    let Inst{31-25} = OPMMA.Value;
    let Inst{24-20} = vs2;
    let Inst{19-15} = vs1;
    let Inst{14} = 0b0;
    let Inst{13-12} = sign;
    let Inst{11-7} = vd;
    let Inst{6-0} = OPC_CUSTOM_1.Value;
    let Constraints =
        !if(EarlyClobber, "@earlyclobber $vd_wb, $vd = $vd_wb", "$vd = $vd_wb");
    let UseNamedOperandTable = true;
  }

  // Sliding-window vector dot product format.
  class SMTVMADotSlide<bits<2> funct2, bits<2> sign, string opcodestr,
                       bit EarlyClobber = 0>
      : RVInst<(outs VRM2:$vd_wb), (ins VRM2:$vd, VRM2:$vs1, VR:$vs2),
               opcodestr, "$vd, $vs1, $vs2", [], InstFormatR> {
    bits<5> vd;
    bits<5> vs1;
    bits<5> vs2;

    let Inst{31-25} = OPMMA_SLIDE.Value;
    let Inst{24-20} = vs2;
    let Inst{19-16} = vs1{4-1};
    let Inst{15-14} = funct2;
    let Inst{13-12} = sign;
    let Inst{11-7} = vd;
    let Inst{6-0} = OPC_CUSTOM_1.Value;
    let Constraints =
        !if(EarlyClobber, "@earlyclobber $vd_wb, $vd = $vd_wb", "$vd = $vd_wb");
    let UseNamedOperandTable = true;
  }

  class SMTVMADotII<bits<2> sign, string opcodestr, bit EarlyClobber = 0>
      : RVInst<(outs VRM2:$vd_wb),
               (ins VRM2:$vd, VR:$vs1, VR:$vs2, SMT_INT:$vtype), opcodestr,
               "$vd, $vs1, $vs2$vtype", [], InstFormatR> {
    bits<5> vd;
    bits<5> vs1;
    bits<5> vs2;
    bits<2> vtype;

    let Inst{31} = 0b1;
    let Inst{30-29} = vtype;
    let Inst{28-25} = 0b0001;
    let Inst{24-20} = vs2;
    let Inst{19-15} = vs1;
    let Inst{14} = 0b0;
    let Inst{13-12} = sign;
    let Inst{11-7} = vd;
    let Inst{6-0} = OPC_CUSTOM_1.Value;
    let Constraints =
        !if(EarlyClobber, "@earlyclobber $vd_wb, $vd = $vd_wb", "$vd = $vd_wb");
    let UseNamedOperandTable = true;
  }

  // Currently only i8 element type is supported for sliding-window dot-product
  // instructions
  class SMTVMADotSlideII<bits<2> slide, bits<2> sign, string opcodestr,
                         bit EarlyClobber = 0>
      : RVInst<(outs VRM2:$vd_wb),
               (ins VRM2:$vd, VRM2:$vs1, VR:$vs2, SMT_I8:$vtype), opcodestr,
               "$vd, $vs1, $vs2$vtype", [], InstFormatR> {
    bits<5> vd;
    bits<5> vs1;
    bits<5> vs2;
    bits<2> vtype;

    let Inst{31} = 1;
    let Inst{30-29} = vtype;
    let Inst{28-25} = 0b0011;
    let Inst{24-20} = vs2;
    let Inst{19-16} = vs1{4-1};
    let Inst{15-14} = slide;
    let Inst{13-12} = sign;
    let Inst{11-7} = vd;
    let Inst{6-0} = OPC_CUSTOM_1.Value;
    let Constraints =
        !if(EarlyClobber, "@earlyclobber $vd_wb, $vd = $vd_wb", "$vd = $vd_wb");
    let UseNamedOperandTable = true;
  }

  class SMTVMADOTSP<bits<2> sign, string opcodestr, string argstr,
                    bit EarlyClobber = 0>
      : RVInst<(outs VRM2NoV0:$vd_wb),
               (ins VRM2NoV0:$vd, VRM2NoV0:$vs1, VRNoV0V1:$vs2, VRV0V1:$vmask, uimm2:$imm2,
                   SMT_INT:$vtype),
               opcodestr, argstr, [], InstFormatR> {
    bits<5> vd;
    bits<5> vs1;
    bits<5> vs2;
    bits<2> vtype;
    bit vmask;
    bits<2> imm2;

    let Inst{31} = 0b1;
    let Inst{30-29} = vtype;
    let Inst{28-26} = 0b010;
    let Inst{25} = vmask;
    let Inst{24-20} = vs2;
    let Inst{19-16} = vs1{4-1};
    let Inst{15} = imm2{1};
    let Inst{14} = 0b0;
    let Inst{13-12} = sign;
    let Inst{11-8} = vd{4-1};
    let Inst{7} = imm2{0};
    let Inst{6-0} = OPC_CUSTOM_1.Value;
    let Constraints =
        !if(EarlyClobber, "@earlyclobber $vd_wb, $vd = $vd_wb", "$vd = $vd_wb");
    let UseNamedOperandTable = true;
  }

  class SMTVMADOTHP<bits<3> funct3, string opcodestr, string argstr,
                    bit EarlyClobber = 0>
      : RVInst<(outs VRNoV0V1:$vd_wb),
               (ins VRNoV0V1:$vd, VRNoV0V1:$vs1, VRNoV0V1:$vs2, VRV0V1:$vmask, uimm3:$imm3,
                   SMT_INT:$vtype),
               opcodestr, argstr, [], InstFormatR> {
    bits<5> vd;
    bits<5> vs1;
    bits<5> vs2;
    bits<2> vtype;
    bit vmask;
    bits<3> imm3;

    let Inst{31} = 0b1;
    let Inst{30-29} = vtype;
    let Inst{28-26} = funct3;
    let Inst{25} = vmask;
    let Inst{24-20} = vs2;
    let Inst{19-15} = vs1;
    let Inst{14-12} = imm3;
    let Inst{11-7} = vd;
    let Inst{6-0} = OPC_CUSTOM_1.Value;
    let Constraints =
        !if(EarlyClobber, "@earlyclobber $vd_wb, $vd = $vd_wb", "$vd = $vd_wb");
    let UseNamedOperandTable = true;
  }

  class SMTVFWMADOT<bits<3> funct3, dag outs, dag ins, string opcodestr,
                    string argstr>
      : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
    bits<5> vd;
    bits<5> vs1;
    bits<5> vs2;
    bits<2> vtype;

    let Inst{31} = 0b1;
    let Inst{30-29} = vtype;
    let Inst{28-25} = 0b1111;
    let Inst{24-20} = vs2;
    let Inst{19-15} = vs1;
    let Inst{14-12} = funct3;
    let Inst{11-7} = vd;
    let Inst{6-0} = OPC_CUSTOM_1.Value;
    let UseNamedOperandTable = true;
  }

  class SMTVPACK<bits<7> funct7, bit funct1, dag outs, dag ins,
                 string opcodestr, string argstr>
      : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
    bits<5> vd;
    bits<5> vs1;
    bits<5> vs2;
    bits<2> imm2;

    let Inst{31-25} = funct7;
    let Inst{24-20} = vs2;
    let Inst{19-15} = vs1;
    let Inst{14} = funct1;
    let Inst{13-12} = imm2;
    let Inst{11-7} = vd;
    let Inst{6-0} = OPC_CUSTOM_1.Value;
    let UseNamedOperandTable = true;
  }
}
