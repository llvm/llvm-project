//===----------------------------------------------------------------------===//
//
//  This file describes the RISCV SPACEMIT RVV extension instruction formats.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RISCV SPACEMIT RVV class templates
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
  }
}

