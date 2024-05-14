//===----------------------------------------------------------------------===//
//  Describe Sim instructions format
//
//  CPU INSTRUCTION FORMATS
//
//  FF FF FF FF
//        ^^ ^^ - r3_imm - third reg or imm
//      ^       - r2     - second reg
//     ^        - r1     - first reg
//  ^^          - opcode - operation code
//
//===----------------------------------------------------------------------===//

// Generic Sim Format
class InArchInst<bits<8> op, dag outs, dag ins, string asmstr, list<dag> pattern>: Instruction
{
  // Inst and Size: for tablegen(... -gen-emitter) and
  // tablegen(... -gen-disassembler) in CMakeLists.txt
  let Namespace = "InArch";
  let DecoderNamespace = "InArch";
  let Size = 4;

  field bits<32> Inst;

  bits<8> Opcode = op;
  bits<4>  r1;
  bits<4>  r2;
  bits<16> r3_imm;

  let Inst{31-24} = Opcode;
  let Inst{23-20} = r1;
  let Inst{19-16} = r2;
  let Inst{15-0}  = r3_imm;

  let OutOperandList = outs;
  let InOperandList  = ins;
  let AsmString   = asmstr;
  let Pattern     = pattern;

}

// Pseudo instructions format
class InArchPseudoInst<dag outs, dag ins, string asmstr, list<dag> pattern>
   : InArchInst<0, outs, ins, asmstr, pattern> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}