class InstNM<dag outs, dag ins, string asmstr, list<dag> pattern>:
  Instruction, PredicateControl
{
  let Namespace = "Mips";
  let DecoderNamespace = "NanoMips";
  let EncodingPredicates = [HasNanoMips];

  let OutOperandList = outs;
  let InOperandList  = ins;

  let AsmString = asmstr;
  let Pattern   = pattern;

  bit isCTI = 0; // Any form of Control Transfer Instruction.
}

class InstSize16 {
  field bits<16> Inst;
  int Size = 2;
}

class InstSize32 {
  field bits<32> Inst;
  int Size = 4;
}

class InstSize48 {
  field bits<48> Inst;
  int Size = 6;
}

// nanoMIPS Pseudo Instructions Format
class PseudoInstNM<dag outs, dag ins, list<dag> pattern> :
    InstNM<outs, ins, "", pattern> {
  let isCodeGenOnly = 1;
  let isPseudo = 1;
}
