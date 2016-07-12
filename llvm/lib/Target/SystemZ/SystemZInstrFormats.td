//==- SystemZInstrFormats.td - SystemZ Instruction Formats --*- tablegen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Basic SystemZ instruction definition
//===----------------------------------------------------------------------===//

class InstSystemZ<int size, dag outs, dag ins, string asmstr,
                  list<dag> pattern> : Instruction {
  let Namespace = "SystemZ";

  dag OutOperandList = outs;
  dag InOperandList = ins;
  let Size = size;
  let Pattern = pattern;
  let AsmString = asmstr;

  // Some instructions come in pairs, one having a 12-bit displacement
  // and the other having a 20-bit displacement.  Both instructions in
  // the pair have the same DispKey and their DispSizes are "12" and "20"
  // respectively.
  string DispKey = "";
  string DispSize = "none";

  // Many register-based <INSN>R instructions have a memory-based <INSN>
  // counterpart.  OpKey uniquely identifies <INSN>, while OpType is
  // "reg" for <INSN>R and "mem" for <INSN>.
  string OpKey = "";
  string OpType = "none";

  // Many distinct-operands instructions have older 2-operand equivalents.
  // NumOpsKey uniquely identifies one of these 2-operand and 3-operand pairs,
  // with NumOpsValue being "2" or "3" as appropriate.
  string NumOpsKey = "";
  string NumOpsValue = "none";

  // True if this instruction is a simple D(X,B) load of a register
  // (with no sign or zero extension).
  bit SimpleBDXLoad = 0;

  // True if this instruction is a simple D(X,B) store of a register
  // (with no truncation).
  bit SimpleBDXStore = 0;

  // True if this instruction has a 20-bit displacement field.
  bit Has20BitOffset = 0;

  // True if addresses in this instruction have an index register.
  bit HasIndex = 0;

  // True if this is a 128-bit pseudo instruction that combines two 64-bit
  // operations.
  bit Is128Bit = 0;

  // The access size of all memory operands in bytes, or 0 if not known.
  bits<5> AccessBytes = 0;

  // If the instruction sets CC to a useful value, this gives the mask
  // of all possible CC results.  The mask has the same form as
  // SystemZ::CCMASK_*.
  bits<4> CCValues = 0;

  // The subset of CCValues that have the same meaning as they would after
  // a comparison of the first operand against zero.
  bits<4> CompareZeroCCMask = 0;

  // True if the instruction is conditional and if the CC mask operand
  // comes first (as for BRC, etc.).
  bit CCMaskFirst = 0;

  // Similar, but true if the CC mask operand comes last (as for LOC, etc.).
  bit CCMaskLast = 0;

  // True if the instruction is the "logical" rather than "arithmetic" form,
  // in cases where a distinction exists.
  bit IsLogical = 0;

  let TSFlags{0}     = SimpleBDXLoad;
  let TSFlags{1}     = SimpleBDXStore;
  let TSFlags{2}     = Has20BitOffset;
  let TSFlags{3}     = HasIndex;
  let TSFlags{4}     = Is128Bit;
  let TSFlags{9-5}   = AccessBytes;
  let TSFlags{13-10} = CCValues;
  let TSFlags{17-14} = CompareZeroCCMask;
  let TSFlags{18}    = CCMaskFirst;
  let TSFlags{19}    = CCMaskLast;
  let TSFlags{20}    = IsLogical;
}

//===----------------------------------------------------------------------===//
// Mappings between instructions
//===----------------------------------------------------------------------===//

// Return the version of an instruction that has an unsigned 12-bit
// displacement.
def getDisp12Opcode : InstrMapping {
  let FilterClass = "InstSystemZ";
  let RowFields = ["DispKey"];
  let ColFields = ["DispSize"];
  let KeyCol = ["20"];
  let ValueCols = [["12"]];
}

// Return the version of an instruction that has a signed 20-bit displacement.
def getDisp20Opcode : InstrMapping {
  let FilterClass = "InstSystemZ";
  let RowFields = ["DispKey"];
  let ColFields = ["DispSize"];
  let KeyCol = ["12"];
  let ValueCols = [["20"]];
}

// Return the memory form of a register instruction.
def getMemOpcode : InstrMapping {
  let FilterClass = "InstSystemZ";
  let RowFields = ["OpKey"];
  let ColFields = ["OpType"];
  let KeyCol = ["reg"];
  let ValueCols = [["mem"]];
}

// Return the 3-operand form of a 2-operand instruction.
def getThreeOperandOpcode : InstrMapping {
  let FilterClass = "InstSystemZ";
  let RowFields = ["NumOpsKey"];
  let ColFields = ["NumOpsValue"];
  let KeyCol = ["2"];
  let ValueCols = [["3"]];
}

//===----------------------------------------------------------------------===//
// Instruction formats
//===----------------------------------------------------------------------===//
//
// Formats are specified using operand field declarations of the form:
//
//   bits<4> Rn   : register input or output for operand n
//   bits<5> Vn   : vector register input or output for operand n
//   bits<m> In   : immediate value of width m for operand n
//   bits<4> BDn  : address operand n, which has a base and a displacement
//   bits<m> XBDn : address operand n, which has an index, a base and a
//                  displacement
//   bits<m> VBDn : address operand n, which has a vector index, a base and a
//                  displacement
//   bits<4> Xn   : index register for address operand n
//   bits<4> Mn   : mode value for operand n
//
// The operand numbers ("n" in the list above) follow the architecture manual.
// Assembly operands sometimes have a different order; in particular, R3 often
// is often written between operands 1 and 2.
//
//===----------------------------------------------------------------------===//

class InstI<bits<8> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<2, outs, ins, asmstr, pattern> {
  field bits<16> Inst;
  field bits<16> SoftFail = 0;

  bits<8> I1;

  let Inst{15-8} = op;
  let Inst{7-0}  = I1;
}

class InstRI<bits<12> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<4, outs, ins, asmstr, pattern> {
  field bits<32> Inst;
  field bits<32> SoftFail = 0;

  bits<4> R1;
  bits<16> I2;

  let Inst{31-24} = op{11-4};
  let Inst{23-20} = R1;
  let Inst{19-16} = op{3-0};
  let Inst{15-0}  = I2;
}

class InstRIEa<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<4> R1;
  bits<16> I2;
  bits<4> M3;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = R1;
  let Inst{35-32} = 0;
  let Inst{31-16} = I2;
  let Inst{15-12} = M3;
  let Inst{11-8}  = 0;
  let Inst{7-0}   = op{7-0};
}

class InstRIEb<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<4> R1;
  bits<4> R2;
  bits<4> M3;
  bits<16> RI4;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = R1;
  let Inst{35-32} = R2;
  let Inst{31-16} = RI4;
  let Inst{15-12} = M3;
  let Inst{11-8}  = 0;
  let Inst{7-0}   = op{7-0};
}

class InstRIEc<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<4> R1;
  bits<8> I2;
  bits<4> M3;
  bits<16> RI4;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = R1;
  let Inst{35-32} = M3;
  let Inst{31-16} = RI4;
  let Inst{15-8}  = I2;
  let Inst{7-0}   = op{7-0};
}

class InstRIEd<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<4> R1;
  bits<4> R3;
  bits<16> I2;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = R1;
  let Inst{35-32} = R3;
  let Inst{31-16} = I2;
  let Inst{15-8}  = 0;
  let Inst{7-0}   = op{7-0};
}

class InstRIEf<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<4> R1;
  bits<4> R2;
  bits<8> I3;
  bits<8> I4;
  bits<8> I5;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = R1;
  let Inst{35-32} = R2;
  let Inst{31-24} = I3;
  let Inst{23-16} = I4;
  let Inst{15-8}  = I5;
  let Inst{7-0}   = op{7-0};
}

class InstRIL<bits<12> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<4> R1;
  bits<32> I2;

  let Inst{47-40} = op{11-4};
  let Inst{39-36} = R1;
  let Inst{35-32} = op{3-0};
  let Inst{31-0}  = I2;
}

class InstRIS<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<4> R1;
  bits<8> I2;
  bits<4> M3;
  bits<16> BD4;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = R1;
  let Inst{35-32} = M3;
  let Inst{31-16} = BD4;
  let Inst{15-8}  = I2;
  let Inst{7-0}   = op{7-0};
}

class InstRR<bits<8> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<2, outs, ins, asmstr, pattern> {
  field bits<16> Inst;
  field bits<16> SoftFail = 0;

  bits<4> R1;
  bits<4> R2;

  let Inst{15-8} = op;
  let Inst{7-4}  = R1;
  let Inst{3-0}  = R2;
}

class InstRRD<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<4, outs, ins, asmstr, pattern> {
  field bits<32> Inst;
  field bits<32> SoftFail = 0;

  bits<4> R1;
  bits<4> R3;
  bits<4> R2;

  let Inst{31-16} = op;
  let Inst{15-12} = R1;
  let Inst{11-8}  = 0;
  let Inst{7-4}   = R3;
  let Inst{3-0}   = R2;
}

class InstRRE<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<4, outs, ins, asmstr, pattern> {
  field bits<32> Inst;
  field bits<32> SoftFail = 0;

  bits<4> R1;
  bits<4> R2;

  let Inst{31-16} = op;
  let Inst{15-8}  = 0;
  let Inst{7-4}   = R1;
  let Inst{3-0}   = R2;
}

class InstRRF<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<4, outs, ins, asmstr, pattern> {
  field bits<32> Inst;
  field bits<32> SoftFail = 0;

  bits<4> R1;
  bits<4> R2;
  bits<4> R3;
  bits<4> R4;

  let Inst{31-16} = op;
  let Inst{15-12} = R3;
  let Inst{11-8}  = R4;
  let Inst{7-4}   = R1;
  let Inst{3-0}   = R2;
}

class InstRRFc<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<4, outs, ins, asmstr, pattern> {
  field bits<32> Inst;
  field bits<32> SoftFail = 0;

  bits<4> R1;
  bits<4> R2;
  bits<4> M3;

  let Inst{31-16} = op;
  let Inst{15-12} = M3;
  let Inst{11-8}  = 0;
  let Inst{7-4}   = R1;
  let Inst{3-0}   = R2;
}

class InstRRS<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<4> R1;
  bits<4> R2;
  bits<4> M3;
  bits<16> BD4;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = R1;
  let Inst{35-32} = R2;
  let Inst{31-16} = BD4;
  let Inst{15-12} = M3;
  let Inst{11-8}  = 0;
  let Inst{7-0}   = op{7-0};
}

class InstRX<bits<8> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<4, outs, ins, asmstr, pattern> {
  field bits<32> Inst;
  field bits<32> SoftFail = 0;

  bits<4> R1;
  bits<20> XBD2;

  let Inst{31-24} = op;
  let Inst{23-20} = R1;
  let Inst{19-0}  = XBD2;

  let HasIndex = 1;
}

class InstRXE<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<4> R1;
  bits<20> XBD2;
  bits<4> M3;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = R1;
  let Inst{35-16} = XBD2;
  let Inst{15-12} = M3;
  let Inst{11-8}  = 0;
  let Inst{7-0}   = op{7-0};

  let HasIndex = 1;
}

class InstRXF<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<4> R1;
  bits<4> R3;
  bits<20> XBD2;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = R3;
  let Inst{35-16} = XBD2;
  let Inst{15-12} = R1;
  let Inst{11-8}  = 0;
  let Inst{7-0}   = op{7-0};

  let HasIndex = 1;
}

class InstRXY<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<4> R1;
  bits<28> XBD2;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = R1;
  let Inst{35-8}  = XBD2;
  let Inst{7-0}   = op{7-0};

  let Has20BitOffset = 1;
  let HasIndex = 1;
}

class InstRS<bits<8> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<4, outs, ins, asmstr, pattern> {
  field bits<32> Inst;
  field bits<32> SoftFail = 0;

  bits<4> R1;
  bits<4> R3;
  bits<16> BD2;

  let Inst{31-24} = op;
  let Inst{23-20} = R1;
  let Inst{19-16} = R3;
  let Inst{15-0}  = BD2;
}

class InstRSY<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<4> R1;
  bits<4> R3;
  bits<24> BD2;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = R1;
  let Inst{35-32} = R3;
  let Inst{31-8}  = BD2;
  let Inst{7-0}   = op{7-0};

  let Has20BitOffset = 1;
}

class InstSI<bits<8> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<4, outs, ins, asmstr, pattern> {
  field bits<32> Inst;
  field bits<32> SoftFail = 0;

  bits<16> BD1;
  bits<8> I2;

  let Inst{31-24} = op;
  let Inst{23-16} = I2;
  let Inst{15-0}  = BD1;
}

class InstSIL<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<16> BD1;
  bits<16> I2;

  let Inst{47-32} = op;
  let Inst{31-16} = BD1;
  let Inst{15-0}  = I2;
}

class InstSIY<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<24> BD1;
  bits<8> I2;

  let Inst{47-40} = op{15-8};
  let Inst{39-32} = I2;
  let Inst{31-8}  = BD1;
  let Inst{7-0}   = op{7-0};

  let Has20BitOffset = 1;
}

class InstSS<bits<8> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<24> BDL1;
  bits<16> BD2;

  let Inst{47-40} = op;
  let Inst{39-16} = BDL1;
  let Inst{15-0}  = BD2;
}

class InstS<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<4, outs, ins, asmstr, pattern> {
  field bits<32> Inst;
  field bits<32> SoftFail = 0;

  bits<16> BD2;

  let Inst{31-16} = op;
  let Inst{15-0}  = BD2;
}

class InstVRIa<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<5> V1;
  bits<16> I2;
  bits<4> M3;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = V1{3-0};
  let Inst{35-32} = 0;
  let Inst{31-16} = I2;
  let Inst{15-12} = M3;
  let Inst{11}    = V1{4};
  let Inst{10-8}  = 0;
  let Inst{7-0}   = op{7-0};
}

class InstVRIb<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<5> V1;
  bits<8> I2;
  bits<8> I3;
  bits<4> M4;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = V1{3-0};
  let Inst{35-32} = 0;
  let Inst{31-24} = I2;
  let Inst{23-16} = I3;
  let Inst{15-12} = M4;
  let Inst{11}    = V1{4};
  let Inst{10-8}  = 0;
  let Inst{7-0}   = op{7-0};
}

class InstVRIc<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<5> V1;
  bits<5> V3;
  bits<16> I2;
  bits<4> M4;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = V1{3-0};
  let Inst{35-32} = V3{3-0};
  let Inst{31-16} = I2;
  let Inst{15-12} = M4;
  let Inst{11}    = V1{4};
  let Inst{10}    = V3{4};
  let Inst{9-8}   = 0;
  let Inst{7-0}   = op{7-0};
}

class InstVRId<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<5> V1;
  bits<5> V2;
  bits<5> V3;
  bits<8> I4;
  bits<4> M5;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = V1{3-0};
  let Inst{35-32} = V2{3-0};
  let Inst{31-28} = V3{3-0};
  let Inst{27-24} = 0;
  let Inst{23-16} = I4;
  let Inst{15-12} = M5;
  let Inst{11}    = V1{4};
  let Inst{10}    = V2{4};
  let Inst{9}     = V3{4};
  let Inst{8}     = 0;
  let Inst{7-0}   = op{7-0};
}

class InstVRIe<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<5> V1;
  bits<5> V2;
  bits<12> I3;
  bits<4> M4;
  bits<4> M5;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = V1{3-0};
  let Inst{35-32} = V2{3-0};
  let Inst{31-20} = I3;
  let Inst{19-16} = M5;
  let Inst{15-12} = M4;
  let Inst{11}    = V1{4};
  let Inst{10}    = V2{4};
  let Inst{9-8}   = 0;
  let Inst{7-0}   = op{7-0};
}

// Depending on the instruction mnemonic, certain bits may be or-ed into
// the M4 value provided as explicit operand.  These are passed as m4or.
class InstVRRa<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern,
               bits<4> m4or = 0>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<5> V1;
  bits<5> V2;
  bits<4> M3;
  bits<4> M4;
  bits<4> M5;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = V1{3-0};
  let Inst{35-32} = V2{3-0};
  let Inst{31-24} = 0;
  let Inst{23-20} = M5;
  let Inst{19}    = !if (!eq (m4or{3}, 1), 1, M4{3});
  let Inst{18}    = !if (!eq (m4or{2}, 1), 1, M4{2});
  let Inst{17}    = !if (!eq (m4or{1}, 1), 1, M4{1});
  let Inst{16}    = !if (!eq (m4or{0}, 1), 1, M4{0});
  let Inst{15-12} = M3;
  let Inst{11}    = V1{4};
  let Inst{10}    = V2{4};
  let Inst{9-8}   = 0;
  let Inst{7-0}   = op{7-0};
}

// Depending on the instruction mnemonic, certain bits may be or-ed into
// the M5 value provided as explicit operand.  These are passed as m5or.
class InstVRRb<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern,
               bits<4> m5or = 0>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<5> V1;
  bits<5> V2;
  bits<5> V3;
  bits<4> M4;
  bits<4> M5;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = V1{3-0};
  let Inst{35-32} = V2{3-0};
  let Inst{31-28} = V3{3-0};
  let Inst{27-24} = 0;
  let Inst{23}    = !if (!eq (m5or{3}, 1), 1, M5{3});
  let Inst{22}    = !if (!eq (m5or{2}, 1), 1, M5{2});
  let Inst{21}    = !if (!eq (m5or{1}, 1), 1, M5{1});
  let Inst{20}    = !if (!eq (m5or{0}, 1), 1, M5{0});
  let Inst{19-16} = 0;
  let Inst{15-12} = M4;
  let Inst{11}    = V1{4};
  let Inst{10}    = V2{4};
  let Inst{9}     = V3{4};
  let Inst{8}     = 0;
  let Inst{7-0}   = op{7-0};
}

class InstVRRc<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<5> V1;
  bits<5> V2;
  bits<5> V3;
  bits<4> M4;
  bits<4> M5;
  bits<4> M6;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = V1{3-0};
  let Inst{35-32} = V2{3-0};
  let Inst{31-28} = V3{3-0};
  let Inst{27-24} = 0;
  let Inst{23-20} = M6;
  let Inst{19-16} = M5;
  let Inst{15-12} = M4;
  let Inst{11}    = V1{4};
  let Inst{10}    = V2{4};
  let Inst{9}     = V3{4};
  let Inst{8}     = 0;
  let Inst{7-0}   = op{7-0};
}

// Depending on the instruction mnemonic, certain bits may be or-ed into
// the M6 value provided as explicit operand.  These are passed as m6or.
class InstVRRd<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern,
               bits<4> m6or = 0>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<5> V1;
  bits<5> V2;
  bits<5> V3;
  bits<5> V4;
  bits<4> M5;
  bits<4> M6;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = V1{3-0};
  let Inst{35-32} = V2{3-0};
  let Inst{31-28} = V3{3-0};
  let Inst{27-24} = M5;
  let Inst{23}    = !if (!eq (m6or{3}, 1), 1, M6{3});
  let Inst{22}    = !if (!eq (m6or{2}, 1), 1, M6{2});
  let Inst{21}    = !if (!eq (m6or{1}, 1), 1, M6{1});
  let Inst{20}    = !if (!eq (m6or{0}, 1), 1, M6{0});
  let Inst{19-16} = 0;
  let Inst{15-12} = V4{3-0};
  let Inst{11}    = V1{4};
  let Inst{10}    = V2{4};
  let Inst{9}     = V3{4};
  let Inst{8}     = V4{4};
  let Inst{7-0}   = op{7-0};
}

class InstVRRe<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<5> V1;
  bits<5> V2;
  bits<5> V3;
  bits<5> V4;
  bits<4> M5;
  bits<4> M6;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = V1{3-0};
  let Inst{35-32} = V2{3-0};
  let Inst{31-28} = V3{3-0};
  let Inst{27-24} = M6;
  let Inst{23-20} = 0;
  let Inst{19-16} = M5;
  let Inst{15-12} = V4{3-0};
  let Inst{11}    = V1{4};
  let Inst{10}    = V2{4};
  let Inst{9}     = V3{4};
  let Inst{8}     = V4{4};
  let Inst{7-0}   = op{7-0};
}

class InstVRRf<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<5> V1;
  bits<4> R2;
  bits<4> R3;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = V1{3-0};
  let Inst{35-32} = R2;
  let Inst{31-28} = R3;
  let Inst{27-12} = 0;
  let Inst{11}    = V1{4};
  let Inst{10-8}  = 0;
  let Inst{7-0}   = op{7-0};
}

class InstVRSa<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<5> V1;
  bits<16> BD2;
  bits<5> V3;
  bits<4> M4;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = V1{3-0};
  let Inst{35-32} = V3{3-0};
  let Inst{31-16} = BD2;
  let Inst{15-12} = M4;
  let Inst{11}    = V1{4};
  let Inst{10}    = V3{4};
  let Inst{9-8}   = 0;
  let Inst{7-0}   = op{7-0};
}

class InstVRSb<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<5> V1;
  bits<16> BD2;
  bits<4> R3;
  bits<4> M4;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = V1{3-0};
  let Inst{35-32} = R3;
  let Inst{31-16} = BD2;
  let Inst{15-12} = M4;
  let Inst{11}    = V1{4};
  let Inst{10-8}  = 0;
  let Inst{7-0}   = op{7-0};
}

class InstVRSc<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<4> R1;
  bits<16> BD2;
  bits<5> V3;
  bits<4> M4;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = R1;
  let Inst{35-32} = V3{3-0};
  let Inst{31-16} = BD2;
  let Inst{15-12} = M4;
  let Inst{11}    = 0;
  let Inst{10}    = V3{4};
  let Inst{9-8}   = 0;
  let Inst{7-0}   = op{7-0};
}

class InstVRV<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<5> V1;
  bits<21> VBD2;
  bits<4> M3;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = V1{3-0};
  let Inst{35-16} = VBD2{19-0};
  let Inst{15-12} = M3;
  let Inst{11}    = V1{4};
  let Inst{10}    = VBD2{20};
  let Inst{9-8}   = 0;
  let Inst{7-0}   = op{7-0};
}

class InstVRX<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstSystemZ<6, outs, ins, asmstr, pattern> {
  field bits<48> Inst;
  field bits<48> SoftFail = 0;

  bits<5> V1;
  bits<20> XBD2;
  bits<4> M3;

  let Inst{47-40} = op{15-8};
  let Inst{39-36} = V1{3-0};
  let Inst{35-16} = XBD2;
  let Inst{15-12} = M3;
  let Inst{11}    = V1{4};
  let Inst{10-8}  = 0;
  let Inst{7-0}   = op{7-0};
}

//===----------------------------------------------------------------------===//
// Instruction definitions with semantics
//===----------------------------------------------------------------------===//
//
// These classes have the form [Cond]<Category><Format>, where <Format> is one
// of the formats defined above and where <Category> describes the inputs
// and outputs.  "Cond" is used if the instruction is conditional,
// in which case the 4-bit condition-code mask is added as a final operand.
// <Category> can be one of:
//
//   Inherent:
//     One register output operand and no input operands.
//
//   BranchUnary:
//     One register output operand, one register input operand and
//     one branch displacement.  The instructions stores a modified
//     form of the source register in the destination register and
//     branches on the result.
//
//   LoadMultiple:
//     One address input operand and two explicit output operands.
//     The instruction loads a range of registers from the address,
//     with the explicit operands giving the first and last register
//     to load.  Other loaded registers are added as implicit definitions.
//
//   StoreMultiple:
//     Two explicit input register operands and an address operand.
//     The instruction stores a range of registers to the address,
//     with the explicit operands giving the first and last register
//     to store.  Other stored registers are added as implicit uses.
//
//   StoreLength:
//     One value operand, one length operand and one address operand.
//     The instruction stores the value operand to the address but
//     doesn't write more than the number of bytes specified by the
//     length operand.
//
//   Unary:
//     One register output operand and one input operand.
//
//   Store:
//     One address operand and one other input operand.  The instruction
//     stores to the address.
//
//   Binary:
//     One register output operand and two input operands.
//
//   StoreBinary:
//     One address operand and two other input operands.  The instruction
//     stores to the address.
//
//   Compare:
//     Two input operands and an implicit CC output operand.
//
//   Test:
//     Two input operands and an implicit CC output operand.  The second
//     input operand is an "address" operand used as a test class mask.
//
//   Ternary:
//     One register output operand and three input operands.
//
//   Quaternary:
//     One register output operand and four input operands.
//
//   LoadAndOp:
//     One output operand and two input operands, one of which is an address.
//     The instruction both reads from and writes to the address.
//
//   CmpSwap:
//     One output operand and three input operands, one of which is an address.
//     The instruction both reads from and writes to the address.
//
//   RotateSelect:
//     One output operand and five input operands.  The first two operands
//     are registers and the other three are immediates.
//
//   Prefetch:
//     One 4-bit immediate operand and one address operand.  The immediate
//     operand is 1 for a load prefetch and 2 for a store prefetch.
//
// The format determines which input operands are tied to output operands,
// and also determines the shape of any address operand.
//
// Multiclasses of the form <Category><Format>Pair define two instructions,
// one with <Category><Format> and one with <Category><Format>Y.  The name
// of the first instruction has no suffix, the name of the second has
// an extra "y".
//
//===----------------------------------------------------------------------===//

class InherentRRE<string mnemonic, bits<16> opcode, RegisterOperand cls,
                  dag src>
  : InstRRE<opcode, (outs cls:$R1), (ins),
            mnemonic#"\t$R1",
            [(set cls:$R1, src)]> {
  let R2 = 0;
}

class InherentVRIa<string mnemonic, bits<16> opcode, bits<16> value>
  : InstVRIa<opcode, (outs VR128:$V1), (ins), mnemonic#"\t$V1", []> {
  let I2 = value;
  let M3 = 0;
}

class BranchUnaryRI<string mnemonic, bits<12> opcode, RegisterOperand cls>
  : InstRI<opcode, (outs cls:$R1), (ins cls:$R1src, brtarget16:$I2),
           mnemonic##"\t$R1, $I2", []> {
  let isBranch = 1;
  let isTerminator = 1;
  let Constraints = "$R1 = $R1src";
  let DisableEncoding = "$R1src";
}

class LoadMultipleRS<string mnemonic, bits<8> opcode, RegisterOperand cls,
                     AddressingMode mode = bdaddr12only>
  : InstRS<opcode, (outs cls:$R1, cls:$R3), (ins mode:$BD2),
           mnemonic#"\t$R1, $R3, $BD2", []> {
  let mayLoad = 1;
}

class LoadMultipleRSY<string mnemonic, bits<16> opcode, RegisterOperand cls,
                      AddressingMode mode = bdaddr20only>
  : InstRSY<opcode, (outs cls:$R1, cls:$R3), (ins mode:$BD2),
            mnemonic#"\t$R1, $R3, $BD2", []> {
  let mayLoad = 1;
}

multiclass LoadMultipleRSPair<string mnemonic, bits<8> rsOpcode,
                              bits<16> rsyOpcode, RegisterOperand cls> {
  let DispKey = mnemonic ## #cls in {
    let DispSize = "12" in
      def "" : LoadMultipleRS<mnemonic, rsOpcode, cls, bdaddr12pair>;
    let DispSize = "20" in
      def Y  : LoadMultipleRSY<mnemonic#"y", rsyOpcode, cls, bdaddr20pair>;
  }
}

class LoadMultipleVRSa<string mnemonic, bits<16> opcode>
  : InstVRSa<opcode, (outs VR128:$V1, VR128:$V3), (ins bdaddr12only:$BD2),
             mnemonic#"\t$V1, $V3, $BD2", []> {
  let M4 = 0;
  let mayLoad = 1;
}

class StoreRILPC<string mnemonic, bits<12> opcode, SDPatternOperator operator,
                 RegisterOperand cls>
  : InstRIL<opcode, (outs), (ins cls:$R1, pcrel32:$I2),
            mnemonic#"\t$R1, $I2",
            [(operator cls:$R1, pcrel32:$I2)]> {
  let mayStore = 1;
  // We want PC-relative addresses to be tried ahead of BD and BDX addresses.
  // However, BDXs have two extra operands and are therefore 6 units more
  // complex.
  let AddedComplexity = 7;
}

class StoreRX<string mnemonic, bits<8> opcode, SDPatternOperator operator,
              RegisterOperand cls, bits<5> bytes,
              AddressingMode mode = bdxaddr12only>
  : InstRX<opcode, (outs), (ins cls:$R1, mode:$XBD2),
           mnemonic#"\t$R1, $XBD2",
           [(operator cls:$R1, mode:$XBD2)]> {
  let OpKey = mnemonic ## cls;
  let OpType = "mem";
  let mayStore = 1;
  let AccessBytes = bytes;
}

class StoreRXY<string mnemonic, bits<16> opcode, SDPatternOperator operator,
               RegisterOperand cls, bits<5> bytes,
               AddressingMode mode = bdxaddr20only>
  : InstRXY<opcode, (outs), (ins cls:$R1, mode:$XBD2),
            mnemonic#"\t$R1, $XBD2",
            [(operator cls:$R1, mode:$XBD2)]> {
  let OpKey = mnemonic ## cls;
  let OpType = "mem";
  let mayStore = 1;
  let AccessBytes = bytes;
}

multiclass StoreRXPair<string mnemonic, bits<8> rxOpcode, bits<16> rxyOpcode,
                       SDPatternOperator operator, RegisterOperand cls,
                       bits<5> bytes> {
  let DispKey = mnemonic ## #cls in {
    let DispSize = "12" in
      def "" : StoreRX<mnemonic, rxOpcode, operator, cls, bytes, bdxaddr12pair>;
    let DispSize = "20" in
      def Y  : StoreRXY<mnemonic#"y", rxyOpcode, operator, cls, bytes,
                        bdxaddr20pair>;
  }
}

class StoreVRX<string mnemonic, bits<16> opcode, SDPatternOperator operator,
               TypedReg tr, bits<5> bytes, bits<4> type = 0>
  : InstVRX<opcode, (outs), (ins tr.op:$V1, bdxaddr12only:$XBD2),
            mnemonic#"\t$V1, $XBD2",
            [(set tr.op:$V1, (tr.vt (operator bdxaddr12only:$XBD2)))]> {
  let M3 = type;
  let mayStore = 1;
  let AccessBytes = bytes;
}

class StoreLengthVRSb<string mnemonic, bits<16> opcode,
                      SDPatternOperator operator, bits<5> bytes>
  : InstVRSb<opcode, (outs), (ins VR128:$V1, GR32:$R3, bdaddr12only:$BD2),
             mnemonic#"\t$V1, $R3, $BD2",
             [(operator VR128:$V1, GR32:$R3, bdaddr12only:$BD2)]> {
  let M4 = 0;
  let mayStore = 1;
  let AccessBytes = bytes;
}

class StoreMultipleRS<string mnemonic, bits<8> opcode, RegisterOperand cls,
                      AddressingMode mode = bdaddr12only>
  : InstRS<opcode, (outs), (ins cls:$R1, cls:$R3, mode:$BD2),
           mnemonic#"\t$R1, $R3, $BD2", []> {
  let mayStore = 1;
}

class StoreMultipleRSY<string mnemonic, bits<16> opcode, RegisterOperand cls,
                       AddressingMode mode = bdaddr20only>
  : InstRSY<opcode, (outs), (ins cls:$R1, cls:$R3, mode:$BD2),
            mnemonic#"\t$R1, $R3, $BD2", []> {
  let mayStore = 1;
}

multiclass StoreMultipleRSPair<string mnemonic, bits<8> rsOpcode,
                               bits<16> rsyOpcode, RegisterOperand cls> {
  let DispKey = mnemonic ## #cls in {
    let DispSize = "12" in
      def "" : StoreMultipleRS<mnemonic, rsOpcode, cls, bdaddr12pair>;
    let DispSize = "20" in
      def Y  : StoreMultipleRSY<mnemonic#"y", rsyOpcode, cls, bdaddr20pair>;
  }
}

class StoreMultipleVRSa<string mnemonic, bits<16> opcode>
  : InstVRSa<opcode, (outs), (ins VR128:$V1, VR128:$V3, bdaddr12only:$BD2),
             mnemonic#"\t$V1, $V3, $BD2", []> {
  let M4 = 0;
  let mayStore = 1;
}

// StoreSI* instructions are used to store an integer to memory, but the
// addresses are more restricted than for normal stores.  If we are in the
// situation of having to force either the address into a register or the
// constant into a register, it's usually better to do the latter.
// We therefore match the address in the same way as a normal store and
// only use the StoreSI* instruction if the matched address is suitable.
class StoreSI<string mnemonic, bits<8> opcode, SDPatternOperator operator,
              Immediate imm>
  : InstSI<opcode, (outs), (ins mviaddr12pair:$BD1, imm:$I2),
           mnemonic#"\t$BD1, $I2",
           [(operator imm:$I2, mviaddr12pair:$BD1)]> {
  let mayStore = 1;
}

class StoreSIY<string mnemonic, bits<16> opcode, SDPatternOperator operator,
               Immediate imm>
  : InstSIY<opcode, (outs), (ins mviaddr20pair:$BD1, imm:$I2),
            mnemonic#"\t$BD1, $I2",
            [(operator imm:$I2, mviaddr20pair:$BD1)]> {
  let mayStore = 1;
}

class StoreSIL<string mnemonic, bits<16> opcode, SDPatternOperator operator,
               Immediate imm>
  : InstSIL<opcode, (outs), (ins mviaddr12pair:$BD1, imm:$I2),
            mnemonic#"\t$BD1, $I2",
            [(operator imm:$I2, mviaddr12pair:$BD1)]> {
  let mayStore = 1;
}

multiclass StoreSIPair<string mnemonic, bits<8> siOpcode, bits<16> siyOpcode,
                       SDPatternOperator operator, Immediate imm> {
  let DispKey = mnemonic in {
    let DispSize = "12" in
      def "" : StoreSI<mnemonic, siOpcode, operator, imm>;
    let DispSize = "20" in
      def Y  : StoreSIY<mnemonic#"y", siyOpcode, operator, imm>;
  }
}

class CondStoreRSY<string mnemonic, bits<16> opcode,
                   RegisterOperand cls, bits<5> bytes,
                   AddressingMode mode = bdaddr20only>
  : InstRSY<opcode, (outs), (ins cls:$R1, mode:$BD2, cond4:$valid, cond4:$R3),
            mnemonic#"$R3\t$R1, $BD2", []>,
    Requires<[FeatureLoadStoreOnCond]> {
  let mayStore = 1;
  let AccessBytes = bytes;
  let CCMaskLast = 1;
}

// Like CondStoreRSY, but used for the raw assembly form.  The condition-code
// mask is the third operand rather than being part of the mnemonic.
class AsmCondStoreRSY<string mnemonic, bits<16> opcode,
                      RegisterOperand cls, bits<5> bytes,
                      AddressingMode mode = bdaddr20only>
  : InstRSY<opcode, (outs), (ins cls:$R1, mode:$BD2, imm32zx4:$R3),
            mnemonic#"\t$R1, $BD2, $R3", []>,
    Requires<[FeatureLoadStoreOnCond]> {
  let mayStore = 1;
  let AccessBytes = bytes;
}

// Like CondStoreRSY, but with a fixed CC mask.
class FixedCondStoreRSY<string mnemonic, bits<16> opcode,
                        RegisterOperand cls, bits<4> ccmask, bits<5> bytes,
                        AddressingMode mode = bdaddr20only>
  : InstRSY<opcode, (outs), (ins cls:$R1, mode:$BD2),
            mnemonic#"\t$R1, $BD2", []>,
    Requires<[FeatureLoadStoreOnCond]> {
  let mayStore = 1;
  let AccessBytes = bytes;
  let R3 = ccmask;
}

class UnaryRR<string mnemonic, bits<8> opcode, SDPatternOperator operator,
              RegisterOperand cls1, RegisterOperand cls2>
  : InstRR<opcode, (outs cls1:$R1), (ins cls2:$R2),
           mnemonic#"r\t$R1, $R2",
           [(set cls1:$R1, (operator cls2:$R2))]> {
  let OpKey = mnemonic ## cls1;
  let OpType = "reg";
}

class UnaryRRE<string mnemonic, bits<16> opcode, SDPatternOperator operator,
               RegisterOperand cls1, RegisterOperand cls2>
  : InstRRE<opcode, (outs cls1:$R1), (ins cls2:$R2),
            mnemonic#"r\t$R1, $R2",
            [(set cls1:$R1, (operator cls2:$R2))]> {
  let OpKey = mnemonic ## cls1;
  let OpType = "reg";
}

class UnaryRRF<string mnemonic, bits<16> opcode, RegisterOperand cls1,
               RegisterOperand cls2>
  : InstRRF<opcode, (outs cls1:$R1), (ins imm32zx4:$R3, cls2:$R2),
            mnemonic#"r\t$R1, $R3, $R2", []> {
  let OpKey = mnemonic ## cls1;
  let OpType = "reg";
  let R4 = 0;
}

class UnaryRRF4<string mnemonic, bits<16> opcode, RegisterOperand cls1,
                RegisterOperand cls2>
  : InstRRF<opcode, (outs cls1:$R1), (ins imm32zx4:$R3, cls2:$R2, imm32zx4:$R4),
            mnemonic#"\t$R1, $R3, $R2, $R4", []>;

// These instructions are generated by if conversion.  The old value of R1
// is added as an implicit use.
class CondUnaryRRF<string mnemonic, bits<16> opcode, RegisterOperand cls1,
                   RegisterOperand cls2>
  : InstRRF<opcode, (outs cls1:$R1), (ins cls2:$R2, cond4:$valid, cond4:$R3),
            mnemonic#"r$R3\t$R1, $R2", []>,
    Requires<[FeatureLoadStoreOnCond]> {
  let CCMaskLast = 1;
  let R4 = 0;
}

class CondUnaryRIE<string mnemonic, bits<16> opcode, RegisterOperand cls,
                   Immediate imm>
  : InstRIEd<opcode, (outs cls:$R1),
                     (ins imm:$I2, cond4:$valid, cond4:$R3),
             mnemonic#"$R3\t$R1, $I2", []>,
    Requires<[FeatureLoadStoreOnCond2]> {
  let CCMaskLast = 1;
}

// Like CondUnaryRRF, but used for the raw assembly form.  The condition-code
// mask is the third operand rather than being part of the mnemonic.
class AsmCondUnaryRRF<string mnemonic, bits<16> opcode, RegisterOperand cls1,
                      RegisterOperand cls2>
  : InstRRF<opcode, (outs cls1:$R1), (ins cls1:$R1src, cls2:$R2, imm32zx4:$R3),
            mnemonic#"r\t$R1, $R2, $R3", []>,
    Requires<[FeatureLoadStoreOnCond]> {
  let Constraints = "$R1 = $R1src";
  let DisableEncoding = "$R1src";
  let R4 = 0;
}

class AsmCondUnaryRIE<string mnemonic, bits<16> opcode, RegisterOperand cls,
                   Immediate imm>
  : InstRIEd<opcode, (outs cls:$R1),
                     (ins cls:$R1src, imm:$I2, imm32zx4:$R3),
             mnemonic#"\t$R1, $I2, $R3", []>,
    Requires<[FeatureLoadStoreOnCond2]> {
  let Constraints = "$R1 = $R1src";
  let DisableEncoding = "$R1src";
}

// Like CondUnaryRRF, but with a fixed CC mask.
class FixedCondUnaryRRF<string mnemonic, bits<16> opcode, RegisterOperand cls1,
                        RegisterOperand cls2, bits<4> ccmask>
  : InstRRF<opcode, (outs cls1:$R1), (ins cls1:$R1src, cls2:$R2),
            mnemonic#"\t$R1, $R2", []>,
    Requires<[FeatureLoadStoreOnCond]> {
  let Constraints = "$R1 = $R1src";
  let DisableEncoding = "$R1src";
  let R3 = ccmask;
  let R4 = 0;
}

class FixedCondUnaryRIE<string mnemonic, bits<16> opcode, RegisterOperand cls,
                   Immediate imm, bits<4> ccmask>
  : InstRIEd<opcode, (outs cls:$R1),
                     (ins cls:$R1src, imm:$I2),
             mnemonic#"\t$R1, $I2", []>,
    Requires<[FeatureLoadStoreOnCond2]> {
  let Constraints = "$R1 = $R1src";
  let DisableEncoding = "$R1src";
  let R3 = ccmask;
}

class UnaryRI<string mnemonic, bits<12> opcode, SDPatternOperator operator,
              RegisterOperand cls, Immediate imm>
  : InstRI<opcode, (outs cls:$R1), (ins imm:$I2),
           mnemonic#"\t$R1, $I2",
           [(set cls:$R1, (operator imm:$I2))]>;

class UnaryRIL<string mnemonic, bits<12> opcode, SDPatternOperator operator,
               RegisterOperand cls, Immediate imm>
  : InstRIL<opcode, (outs cls:$R1), (ins imm:$I2),
            mnemonic#"\t$R1, $I2",
            [(set cls:$R1, (operator imm:$I2))]>;

class UnaryRILPC<string mnemonic, bits<12> opcode, SDPatternOperator operator,
                 RegisterOperand cls>
  : InstRIL<opcode, (outs cls:$R1), (ins pcrel32:$I2),
            mnemonic#"\t$R1, $I2",
            [(set cls:$R1, (operator pcrel32:$I2))]> {
  let mayLoad = 1;
  // We want PC-relative addresses to be tried ahead of BD and BDX addresses.
  // However, BDXs have two extra operands and are therefore 6 units more
  // complex.
  let AddedComplexity = 7;
}

class CondUnaryRSY<string mnemonic, bits<16> opcode,
                   SDPatternOperator operator, RegisterOperand cls,
                   bits<5> bytes, AddressingMode mode = bdaddr20only>
  : InstRSY<opcode, (outs cls:$R1),
            (ins cls:$R1src, mode:$BD2, cond4:$valid, cond4:$R3),
            mnemonic#"$R3\t$R1, $BD2",
            [(set cls:$R1,
                  (z_select_ccmask (load bdaddr20only:$BD2), cls:$R1src,
                                   cond4:$valid, cond4:$R3))]>,
    Requires<[FeatureLoadStoreOnCond]> {
  let Constraints = "$R1 = $R1src";
  let DisableEncoding = "$R1src";
  let mayLoad = 1;
  let AccessBytes = bytes;
  let CCMaskLast = 1;
}

// Like CondUnaryRSY, but used for the raw assembly form.  The condition-code
// mask is the third operand rather than being part of the mnemonic.
class AsmCondUnaryRSY<string mnemonic, bits<16> opcode,
                      RegisterOperand cls, bits<5> bytes,
                      AddressingMode mode = bdaddr20only>
  : InstRSY<opcode, (outs cls:$R1), (ins cls:$R1src, mode:$BD2, imm32zx4:$R3),
            mnemonic#"\t$R1, $BD2, $R3", []>,
    Requires<[FeatureLoadStoreOnCond]> {
  let mayLoad = 1;
  let AccessBytes = bytes;
  let Constraints = "$R1 = $R1src";
  let DisableEncoding = "$R1src";
}

// Like CondUnaryRSY, but with a fixed CC mask.
class FixedCondUnaryRSY<string mnemonic, bits<16> opcode,
                        RegisterOperand cls, bits<4> ccmask, bits<5> bytes,
                        AddressingMode mode = bdaddr20only>
  : InstRSY<opcode, (outs cls:$R1), (ins cls:$R1src, mode:$BD2),
            mnemonic#"\t$R1, $BD2", []>,
    Requires<[FeatureLoadStoreOnCond]> {
  let Constraints = "$R1 = $R1src";
  let DisableEncoding = "$R1src";
  let R3 = ccmask;
  let mayLoad = 1;
  let AccessBytes = bytes;
}

class UnaryRX<string mnemonic, bits<8> opcode, SDPatternOperator operator,
              RegisterOperand cls, bits<5> bytes,
              AddressingMode mode = bdxaddr12only>
  : InstRX<opcode, (outs cls:$R1), (ins mode:$XBD2),
           mnemonic#"\t$R1, $XBD2",
           [(set cls:$R1, (operator mode:$XBD2))]> {
  let OpKey = mnemonic ## cls;
  let OpType = "mem";
  let mayLoad = 1;
  let AccessBytes = bytes;
}

class UnaryRXE<string mnemonic, bits<16> opcode, SDPatternOperator operator,
               RegisterOperand cls, bits<5> bytes>
  : InstRXE<opcode, (outs cls:$R1), (ins bdxaddr12only:$XBD2),
            mnemonic#"\t$R1, $XBD2",
            [(set cls:$R1, (operator bdxaddr12only:$XBD2))]> {
  let OpKey = mnemonic ## cls;
  let OpType = "mem";
  let mayLoad = 1;
  let AccessBytes = bytes;
  let M3 = 0;
}

class UnaryRXY<string mnemonic, bits<16> opcode, SDPatternOperator operator,
               RegisterOperand cls, bits<5> bytes,
               AddressingMode mode = bdxaddr20only>
  : InstRXY<opcode, (outs cls:$R1), (ins mode:$XBD2),
            mnemonic#"\t$R1, $XBD2",
            [(set cls:$R1, (operator mode:$XBD2))]> {
  let OpKey = mnemonic ## cls;
  let OpType = "mem";
  let mayLoad = 1;
  let AccessBytes = bytes;
}

multiclass UnaryRXPair<string mnemonic, bits<8> rxOpcode, bits<16> rxyOpcode,
                       SDPatternOperator operator, RegisterOperand cls,
                       bits<5> bytes> {
  let DispKey = mnemonic ## #cls in {
    let DispSize = "12" in
      def "" : UnaryRX<mnemonic, rxOpcode, operator, cls, bytes, bdxaddr12pair>;
    let DispSize = "20" in
      def Y  : UnaryRXY<mnemonic#"y", rxyOpcode, operator, cls, bytes,
                        bdxaddr20pair>;
  }
}

class UnaryVRIa<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                TypedReg tr, Immediate imm, bits<4> type = 0>
  : InstVRIa<opcode, (outs tr.op:$V1), (ins imm:$I2),
             mnemonic#"\t$V1, $I2",
             [(set tr.op:$V1, (tr.vt (operator imm:$I2)))]> {
  let M3 = type;
}

class UnaryVRRa<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                TypedReg tr1, TypedReg tr2, bits<4> type = 0, bits<4> m4 = 0,
                bits<4> m5 = 0>
  : InstVRRa<opcode, (outs tr1.op:$V1), (ins tr2.op:$V2),
             mnemonic#"\t$V1, $V2",
             [(set tr1.op:$V1, (tr1.vt (operator (tr2.vt tr2.op:$V2))))]> {
  let M3 = type;
  let M4 = m4;
  let M5 = m5;
}

multiclass UnaryVRRaSPair<string mnemonic, bits<16> opcode,
                          SDPatternOperator operator,
                          SDPatternOperator operator_cc, TypedReg tr1,
                          TypedReg tr2, bits<4> type, bits<4> modifier = 0,
                          bits<4> modifier_cc = 1> {
  def "" : UnaryVRRa<mnemonic, opcode, operator, tr1, tr2, type, 0, modifier>;
  let Defs = [CC] in
    def S : UnaryVRRa<mnemonic##"s", opcode, operator_cc, tr1, tr2, type, 0,
                      modifier_cc>;
}

class UnaryVRX<string mnemonic, bits<16> opcode, SDPatternOperator operator,
               TypedReg tr, bits<5> bytes, bits<4> type = 0>
  : InstVRX<opcode, (outs tr.op:$V1), (ins bdxaddr12only:$XBD2),
            mnemonic#"\t$V1, $XBD2",
            [(set tr.op:$V1, (tr.vt (operator bdxaddr12only:$XBD2)))]> {
  let M3 = type;
  let mayLoad = 1;
  let AccessBytes = bytes;
}

class BinaryRR<string mnemonic, bits<8> opcode, SDPatternOperator operator,
               RegisterOperand cls1, RegisterOperand cls2>
  : InstRR<opcode, (outs cls1:$R1), (ins cls1:$R1src, cls2:$R2),
           mnemonic#"r\t$R1, $R2",
           [(set cls1:$R1, (operator cls1:$R1src, cls2:$R2))]> {
  let OpKey = mnemonic ## cls1;
  let OpType = "reg";
  let Constraints = "$R1 = $R1src";
  let DisableEncoding = "$R1src";
}

class BinaryRRE<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                RegisterOperand cls1, RegisterOperand cls2>
  : InstRRE<opcode, (outs cls1:$R1), (ins cls1:$R1src, cls2:$R2),
            mnemonic#"r\t$R1, $R2",
            [(set cls1:$R1, (operator cls1:$R1src, cls2:$R2))]> {
  let OpKey = mnemonic ## cls1;
  let OpType = "reg";
  let Constraints = "$R1 = $R1src";
  let DisableEncoding = "$R1src";
}

class BinaryRRF<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                RegisterOperand cls1, RegisterOperand cls2>
  : InstRRF<opcode, (outs cls1:$R1), (ins cls1:$R2, cls2:$R3),
            mnemonic#"r\t$R1, $R3, $R2",
            [(set cls1:$R1, (operator cls1:$R2, cls2:$R3))]> {
  let OpKey = mnemonic ## cls1;
  let OpType = "reg";
  let R4 = 0;
}

class BinaryRRFK<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                 RegisterOperand cls1, RegisterOperand cls2>
  : InstRRF<opcode, (outs cls1:$R1), (ins cls1:$R2, cls2:$R3),
            mnemonic#"rk\t$R1, $R2, $R3",
            [(set cls1:$R1, (operator cls1:$R2, cls2:$R3))]> {
  let R4 = 0;
}

multiclass BinaryRRAndK<string mnemonic, bits<8> opcode1, bits<16> opcode2,
                        SDPatternOperator operator, RegisterOperand cls1,
                        RegisterOperand cls2> {
  let NumOpsKey = mnemonic in {
    let NumOpsValue = "3" in
      def K : BinaryRRFK<mnemonic, opcode2, null_frag, cls1, cls2>,
              Requires<[FeatureDistinctOps]>;
    let NumOpsValue = "2", isConvertibleToThreeAddress = 1 in
      def "" : BinaryRR<mnemonic, opcode1, operator, cls1, cls2>;
  }
}

multiclass BinaryRREAndK<string mnemonic, bits<16> opcode1, bits<16> opcode2,
                         SDPatternOperator operator, RegisterOperand cls1,
                         RegisterOperand cls2> {
  let NumOpsKey = mnemonic in {
    let NumOpsValue = "3" in
      def K : BinaryRRFK<mnemonic, opcode2, null_frag, cls1, cls2>,
              Requires<[FeatureDistinctOps]>;
    let NumOpsValue = "2", isConvertibleToThreeAddress = 1 in
      def "" : BinaryRRE<mnemonic, opcode1, operator, cls1, cls2>;
  }
}

class BinaryRI<string mnemonic, bits<12> opcode, SDPatternOperator operator,
               RegisterOperand cls, Immediate imm>
  : InstRI<opcode, (outs cls:$R1), (ins cls:$R1src, imm:$I2),
           mnemonic#"\t$R1, $I2",
           [(set cls:$R1, (operator cls:$R1src, imm:$I2))]> {
  let Constraints = "$R1 = $R1src";
  let DisableEncoding = "$R1src";
}

class BinaryRIE<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                RegisterOperand cls, Immediate imm>
  : InstRIEd<opcode, (outs cls:$R1), (ins cls:$R3, imm:$I2),
             mnemonic#"\t$R1, $R3, $I2",
             [(set cls:$R1, (operator cls:$R3, imm:$I2))]>;

multiclass BinaryRIAndK<string mnemonic, bits<12> opcode1, bits<16> opcode2,
                        SDPatternOperator operator, RegisterOperand cls,
                        Immediate imm> {
  let NumOpsKey = mnemonic in {
    let NumOpsValue = "3" in
      def K : BinaryRIE<mnemonic##"k", opcode2, null_frag, cls, imm>,
              Requires<[FeatureDistinctOps]>;
    let NumOpsValue = "2", isConvertibleToThreeAddress = 1 in
      def "" : BinaryRI<mnemonic, opcode1, operator, cls, imm>;
  }
}

class BinaryRIL<string mnemonic, bits<12> opcode, SDPatternOperator operator,
                RegisterOperand cls, Immediate imm>
  : InstRIL<opcode, (outs cls:$R1), (ins cls:$R1src, imm:$I2),
            mnemonic#"\t$R1, $I2",
            [(set cls:$R1, (operator cls:$R1src, imm:$I2))]> {
  let Constraints = "$R1 = $R1src";
  let DisableEncoding = "$R1src";
}

class BinaryRS<string mnemonic, bits<8> opcode, SDPatternOperator operator,
               RegisterOperand cls>
  : InstRS<opcode, (outs cls:$R1), (ins cls:$R1src, shift12only:$BD2),
           mnemonic#"\t$R1, $BD2",
           [(set cls:$R1, (operator cls:$R1src, shift12only:$BD2))]> {
  let R3 = 0;
  let Constraints = "$R1 = $R1src";
  let DisableEncoding = "$R1src";
}

class BinaryRSY<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                RegisterOperand cls>
  : InstRSY<opcode, (outs cls:$R1), (ins cls:$R3, shift20only:$BD2),
            mnemonic#"\t$R1, $R3, $BD2",
            [(set cls:$R1, (operator cls:$R3, shift20only:$BD2))]>;

multiclass BinaryRSAndK<string mnemonic, bits<8> opcode1, bits<16> opcode2,
                        SDPatternOperator operator, RegisterOperand cls> {
  let NumOpsKey = mnemonic in {
    let NumOpsValue = "3" in
      def K  : BinaryRSY<mnemonic##"k", opcode2, null_frag, cls>,
               Requires<[FeatureDistinctOps]>;
    let NumOpsValue = "2", isConvertibleToThreeAddress = 1 in
      def "" : BinaryRS<mnemonic, opcode1, operator, cls>;
  }
}

class BinaryRX<string mnemonic, bits<8> opcode, SDPatternOperator operator,
               RegisterOperand cls, SDPatternOperator load, bits<5> bytes,
               AddressingMode mode = bdxaddr12only>
  : InstRX<opcode, (outs cls:$R1), (ins cls:$R1src, mode:$XBD2),
           mnemonic#"\t$R1, $XBD2",
           [(set cls:$R1, (operator cls:$R1src, (load mode:$XBD2)))]> {
  let OpKey = mnemonic ## cls;
  let OpType = "mem";
  let Constraints = "$R1 = $R1src";
  let DisableEncoding = "$R1src";
  let mayLoad = 1;
  let AccessBytes = bytes;
}

class BinaryRXE<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                  RegisterOperand cls, SDPatternOperator load, bits<5> bytes>
  : InstRXE<opcode, (outs cls:$R1), (ins cls:$R1src, bdxaddr12only:$XBD2),
            mnemonic#"\t$R1, $XBD2",
            [(set cls:$R1, (operator cls:$R1src,
                                     (load bdxaddr12only:$XBD2)))]> {
  let OpKey = mnemonic ## cls;
  let OpType = "mem";
  let Constraints = "$R1 = $R1src";
  let DisableEncoding = "$R1src";
  let mayLoad = 1;
  let AccessBytes = bytes;
  let M3 = 0;
}

class BinaryRXY<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                RegisterOperand cls, SDPatternOperator load, bits<5> bytes,
                AddressingMode mode = bdxaddr20only>
  : InstRXY<opcode, (outs cls:$R1), (ins cls:$R1src, mode:$XBD2),
            mnemonic#"\t$R1, $XBD2",
            [(set cls:$R1, (operator cls:$R1src, (load mode:$XBD2)))]> {
  let OpKey = mnemonic ## cls;
  let OpType = "mem";
  let Constraints = "$R1 = $R1src";
  let DisableEncoding = "$R1src";
  let mayLoad = 1;
  let AccessBytes = bytes;
}

multiclass BinaryRXPair<string mnemonic, bits<8> rxOpcode, bits<16> rxyOpcode,
                        SDPatternOperator operator, RegisterOperand cls,
                        SDPatternOperator load, bits<5> bytes> {
  let DispKey = mnemonic ## #cls in {
    let DispSize = "12" in
      def "" : BinaryRX<mnemonic, rxOpcode, operator, cls, load, bytes,
                        bdxaddr12pair>;
    let DispSize = "20" in
      def Y  : BinaryRXY<mnemonic#"y", rxyOpcode, operator, cls, load, bytes,
                         bdxaddr20pair>;
  }
}

class BinarySI<string mnemonic, bits<8> opcode, SDPatternOperator operator,
               Operand imm, AddressingMode mode = bdaddr12only>
  : InstSI<opcode, (outs), (ins mode:$BD1, imm:$I2),
           mnemonic#"\t$BD1, $I2",
           [(store (operator (load mode:$BD1), imm:$I2), mode:$BD1)]> {
  let mayLoad = 1;
  let mayStore = 1;
}

class BinarySIY<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                Operand imm, AddressingMode mode = bdaddr20only>
  : InstSIY<opcode, (outs), (ins mode:$BD1, imm:$I2),
            mnemonic#"\t$BD1, $I2",
            [(store (operator (load mode:$BD1), imm:$I2), mode:$BD1)]> {
  let mayLoad = 1;
  let mayStore = 1;
}

multiclass BinarySIPair<string mnemonic, bits<8> siOpcode,
                        bits<16> siyOpcode, SDPatternOperator operator,
                        Operand imm> {
  let DispKey = mnemonic ## #cls in {
    let DispSize = "12" in
      def "" : BinarySI<mnemonic, siOpcode, operator, imm, bdaddr12pair>;
    let DispSize = "20" in
      def Y  : BinarySIY<mnemonic#"y", siyOpcode, operator, imm, bdaddr20pair>;
  }
}

class BinaryVRIb<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                 TypedReg tr, bits<4> type>
  : InstVRIb<opcode, (outs tr.op:$V1), (ins imm32zx8:$I2, imm32zx8:$I3),
             mnemonic#"\t$V1, $I2, $I3",
             [(set tr.op:$V1, (tr.vt (operator imm32zx8:$I2, imm32zx8:$I3)))]> {
  let M4 = type;
}

class BinaryVRIc<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                 TypedReg tr1, TypedReg tr2, bits<4> type>
  : InstVRIc<opcode, (outs tr1.op:$V1), (ins tr2.op:$V3, imm32zx16:$I2),
             mnemonic#"\t$V1, $V3, $I2",
             [(set tr1.op:$V1, (tr1.vt (operator (tr2.vt tr2.op:$V3),
                                                 imm32zx16:$I2)))]> {
  let M4 = type;
}

class BinaryVRIe<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                 TypedReg tr1, TypedReg tr2, bits<4> type, bits<4> m5>
  : InstVRIe<opcode, (outs tr1.op:$V1), (ins tr2.op:$V2, imm32zx12:$I3),
             mnemonic#"\t$V1, $V2, $I3",
             [(set tr1.op:$V1, (tr1.vt (operator (tr2.vt tr2.op:$V2),
                                                 imm32zx12:$I3)))]> {
  let M4 = type;
  let M5 = m5;
}

class BinaryVRRa<string mnemonic, bits<16> opcode>
  : InstVRRa<opcode, (outs VR128:$V1), (ins VR128:$V2, imm32zx4:$M3),
             mnemonic#"\t$V1, $V2, $M3", []> {
  let M4 = 0;
  let M5 = 0;
}

class BinaryVRRb<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                 TypedReg tr1, TypedReg tr2, bits<4> type = 0,
                 bits<4> modifier = 0>
  : InstVRRb<opcode, (outs tr1.op:$V1), (ins tr2.op:$V2, tr2.op:$V3),
             mnemonic#"\t$V1, $V2, $V3",
             [(set tr1.op:$V1, (tr1.vt (operator (tr2.vt tr2.op:$V2),
                                                 (tr2.vt tr2.op:$V3))))]> {
  let M4 = type;
  let M5 = modifier;
}

// Declare a pair of instructions, one which sets CC and one which doesn't.
// The CC-setting form ends with "S" and sets the low bit of M5.
multiclass BinaryVRRbSPair<string mnemonic, bits<16> opcode,
                           SDPatternOperator operator,
                           SDPatternOperator operator_cc, TypedReg tr1,
                           TypedReg tr2, bits<4> type,
                           bits<4> modifier = 0, bits<4> modifier_cc = 1> {
  def "" : BinaryVRRb<mnemonic, opcode, operator, tr1, tr2, type, modifier>;
  let Defs = [CC] in
    def S : BinaryVRRb<mnemonic##"s", opcode, operator_cc, tr1, tr2, type,
                       modifier_cc>;
}

class BinaryVRRc<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                 TypedReg tr1, TypedReg tr2, bits<4> type = 0, bits<4> m5 = 0,
                 bits<4> m6 = 0>
  : InstVRRc<opcode, (outs tr1.op:$V1), (ins tr2.op:$V2, tr2.op:$V3),
             mnemonic#"\t$V1, $V2, $V3",
             [(set tr1.op:$V1, (tr1.vt (operator (tr2.vt tr2.op:$V2),
                                                 (tr2.vt tr2.op:$V3))))]> {
  let M4 = type;
  let M5 = m5;
  let M6 = m6;
}

multiclass BinaryVRRcSPair<string mnemonic, bits<16> opcode,
                           SDPatternOperator operator,
                           SDPatternOperator operator_cc, TypedReg tr1,
                           TypedReg tr2, bits<4> type, bits<4> m5,
                           bits<4> modifier = 0, bits<4> modifier_cc = 1> {
  def "" : BinaryVRRc<mnemonic, opcode, operator, tr1, tr2, type, m5, modifier>;
  let Defs = [CC] in
    def S : BinaryVRRc<mnemonic##"s", opcode, operator_cc, tr1, tr2, type,
                       m5, modifier_cc>;
}

class BinaryVRRf<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                 TypedReg tr>
  : InstVRRf<opcode, (outs tr.op:$V1), (ins GR64:$R2, GR64:$R3),
             mnemonic#"\t$V1, $R2, $R3",
             [(set tr.op:$V1, (tr.vt (operator GR64:$R2, GR64:$R3)))]>;

class BinaryVRSa<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                 TypedReg tr1, TypedReg tr2, bits<4> type>
  : InstVRSa<opcode, (outs tr1.op:$V1), (ins tr2.op:$V3, shift12only:$BD2),
             mnemonic#"\t$V1, $V3, $BD2",
             [(set tr1.op:$V1, (tr1.vt (operator (tr2.vt tr2.op:$V3),
                                                 shift12only:$BD2)))]> {
  let M4 = type;
}

class BinaryVRSb<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                 bits<5> bytes>
  : InstVRSb<opcode, (outs VR128:$V1), (ins GR32:$R3, bdaddr12only:$BD2),
             mnemonic#"\t$V1, $R3, $BD2",
             [(set VR128:$V1, (operator GR32:$R3, bdaddr12only:$BD2))]> {
  let M4 = 0;
  let mayLoad = 1;
  let AccessBytes = bytes;
}

class BinaryVRSc<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                 TypedReg tr, bits<4> type>
  : InstVRSc<opcode, (outs GR64:$R1), (ins tr.op:$V3, shift12only:$BD2),
           mnemonic#"\t$R1, $V3, $BD2",
           [(set GR64:$R1, (operator (tr.vt tr.op:$V3), shift12only:$BD2))]> {
  let M4 = type;
}

class BinaryVRX<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                TypedReg tr, bits<5> bytes>
  : InstVRX<opcode, (outs VR128:$V1), (ins bdxaddr12only:$XBD2, imm32zx4:$M3),
            mnemonic#"\t$V1, $XBD2, $M3",
            [(set tr.op:$V1, (tr.vt (operator bdxaddr12only:$XBD2,
                                              imm32zx4:$M3)))]> {
  let mayLoad = 1;
  let AccessBytes = bytes;
}

class StoreBinaryVRV<string mnemonic, bits<16> opcode, bits<5> bytes,
                     Immediate index>
  : InstVRV<opcode, (outs), (ins VR128:$V1, bdvaddr12only:$VBD2, index:$M3),
            mnemonic#"\t$V1, $VBD2, $M3", []> {
  let mayStore = 1;
  let AccessBytes = bytes;
}

class StoreBinaryVRX<string mnemonic, bits<16> opcode,
                     SDPatternOperator operator, TypedReg tr, bits<5> bytes,
                     Immediate index>
  : InstVRX<opcode, (outs), (ins tr.op:$V1, bdxaddr12only:$XBD2, index:$M3),
            mnemonic#"\t$V1, $XBD2, $M3",
            [(operator (tr.vt tr.op:$V1), bdxaddr12only:$XBD2, index:$M3)]> {
  let mayStore = 1;
  let AccessBytes = bytes;
}

class CompareRR<string mnemonic, bits<8> opcode, SDPatternOperator operator,
                RegisterOperand cls1, RegisterOperand cls2>
  : InstRR<opcode, (outs), (ins cls1:$R1, cls2:$R2),
           mnemonic#"r\t$R1, $R2",
           [(operator cls1:$R1, cls2:$R2)]> {
  let OpKey = mnemonic ## cls1;
  let OpType = "reg";
  let isCompare = 1;
}

class CompareRRE<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                 RegisterOperand cls1, RegisterOperand cls2>
  : InstRRE<opcode, (outs), (ins cls1:$R1, cls2:$R2),
            mnemonic#"r\t$R1, $R2",
            [(operator cls1:$R1, cls2:$R2)]> {
  let OpKey = mnemonic ## cls1;
  let OpType = "reg";
  let isCompare = 1;
}

class CompareRI<string mnemonic, bits<12> opcode, SDPatternOperator operator,
                RegisterOperand cls, Immediate imm>
  : InstRI<opcode, (outs), (ins cls:$R1, imm:$I2),
           mnemonic#"\t$R1, $I2",
           [(operator cls:$R1, imm:$I2)]> {
  let isCompare = 1;
}

class CompareRIL<string mnemonic, bits<12> opcode, SDPatternOperator operator,
                 RegisterOperand cls, Immediate imm>
  : InstRIL<opcode, (outs), (ins cls:$R1, imm:$I2),
            mnemonic#"\t$R1, $I2",
            [(operator cls:$R1, imm:$I2)]> {
  let isCompare = 1;
}

class CompareRILPC<string mnemonic, bits<12> opcode, SDPatternOperator operator,
                   RegisterOperand cls, SDPatternOperator load>
  : InstRIL<opcode, (outs), (ins cls:$R1, pcrel32:$I2),
            mnemonic#"\t$R1, $I2",
            [(operator cls:$R1, (load pcrel32:$I2))]> {
  let isCompare = 1;
  let mayLoad = 1;
  // We want PC-relative addresses to be tried ahead of BD and BDX addresses.
  // However, BDXs have two extra operands and are therefore 6 units more
  // complex.
  let AddedComplexity = 7;
}

class CompareRX<string mnemonic, bits<8> opcode, SDPatternOperator operator,
                RegisterOperand cls, SDPatternOperator load, bits<5> bytes,
                AddressingMode mode = bdxaddr12only>
  : InstRX<opcode, (outs), (ins cls:$R1, mode:$XBD2),
           mnemonic#"\t$R1, $XBD2",
           [(operator cls:$R1, (load mode:$XBD2))]> {
  let OpKey = mnemonic ## cls;
  let OpType = "mem";
  let isCompare = 1;
  let mayLoad = 1;
  let AccessBytes = bytes;
}

class CompareRXE<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                 RegisterOperand cls, SDPatternOperator load, bits<5> bytes>
  : InstRXE<opcode, (outs), (ins cls:$R1, bdxaddr12only:$XBD2),
            mnemonic#"\t$R1, $XBD2",
            [(operator cls:$R1, (load bdxaddr12only:$XBD2))]> {
  let OpKey = mnemonic ## cls;
  let OpType = "mem";
  let isCompare = 1;
  let mayLoad = 1;
  let AccessBytes = bytes;
  let M3 = 0;
}

class CompareRXY<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                 RegisterOperand cls, SDPatternOperator load, bits<5> bytes,
                 AddressingMode mode = bdxaddr20only>
  : InstRXY<opcode, (outs), (ins cls:$R1, mode:$XBD2),
            mnemonic#"\t$R1, $XBD2",
            [(operator cls:$R1, (load mode:$XBD2))]> {
  let OpKey = mnemonic ## cls;
  let OpType = "mem";
  let isCompare = 1;
  let mayLoad = 1;
  let AccessBytes = bytes;
}

multiclass CompareRXPair<string mnemonic, bits<8> rxOpcode, bits<16> rxyOpcode,
                         SDPatternOperator operator, RegisterOperand cls,
                         SDPatternOperator load, bits<5> bytes> {
  let DispKey = mnemonic ## #cls in {
    let DispSize = "12" in
      def "" : CompareRX<mnemonic, rxOpcode, operator, cls,
                         load, bytes, bdxaddr12pair>;
    let DispSize = "20" in
      def Y  : CompareRXY<mnemonic#"y", rxyOpcode, operator, cls,
                          load, bytes, bdxaddr20pair>;
  }
}

class CompareSI<string mnemonic, bits<8> opcode, SDPatternOperator operator,
                SDPatternOperator load, Immediate imm,
                AddressingMode mode = bdaddr12only>
  : InstSI<opcode, (outs), (ins mode:$BD1, imm:$I2),
           mnemonic#"\t$BD1, $I2",
           [(operator (load mode:$BD1), imm:$I2)]> {
  let isCompare = 1;
  let mayLoad = 1;
}

class CompareSIL<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                 SDPatternOperator load, Immediate imm>
  : InstSIL<opcode, (outs), (ins bdaddr12only:$BD1, imm:$I2),
            mnemonic#"\t$BD1, $I2",
            [(operator (load bdaddr12only:$BD1), imm:$I2)]> {
  let isCompare = 1;
  let mayLoad = 1;
}

class CompareSIY<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                 SDPatternOperator load, Immediate imm,
                 AddressingMode mode = bdaddr20only>
  : InstSIY<opcode, (outs), (ins mode:$BD1, imm:$I2),
            mnemonic#"\t$BD1, $I2",
            [(operator (load mode:$BD1), imm:$I2)]> {
  let isCompare = 1;
  let mayLoad = 1;
}

multiclass CompareSIPair<string mnemonic, bits<8> siOpcode, bits<16> siyOpcode,
                         SDPatternOperator operator, SDPatternOperator load,
                         Immediate imm> {
  let DispKey = mnemonic in {
    let DispSize = "12" in
      def "" : CompareSI<mnemonic, siOpcode, operator, load, imm, bdaddr12pair>;
    let DispSize = "20" in
      def Y  : CompareSIY<mnemonic#"y", siyOpcode, operator, load, imm,
                          bdaddr20pair>;
  }
}

class CompareVRRa<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                  TypedReg tr, bits<4> type>
  : InstVRRa<opcode, (outs), (ins tr.op:$V1, tr.op:$V2),
             mnemonic#"\t$V1, $V2",
             [(operator (tr.vt tr.op:$V1), (tr.vt tr.op:$V2))]> {
  let isCompare = 1;
  let M3 = type;
  let M4 = 0;
  let M5 = 0;
}

class TestRXE<string mnemonic, bits<16> opcode, SDPatternOperator operator,
              RegisterOperand cls>
  : InstRXE<opcode, (outs), (ins cls:$R1, bdxaddr12only:$XBD2),
            mnemonic#"\t$R1, $XBD2",
            [(operator cls:$R1, bdxaddr12only:$XBD2)]> {
  let M3 = 0;
}

class TernaryRRD<string mnemonic, bits<16> opcode,
                 SDPatternOperator operator, RegisterOperand cls>
  : InstRRD<opcode, (outs cls:$R1), (ins cls:$R1src, cls:$R3, cls:$R2),
            mnemonic#"r\t$R1, $R3, $R2",
            [(set cls:$R1, (operator cls:$R1src, cls:$R3, cls:$R2))]> {
  let OpKey = mnemonic ## cls;
  let OpType = "reg";
  let Constraints = "$R1 = $R1src";
  let DisableEncoding = "$R1src";
}

class TernaryRS<string mnemonic, bits<8> opcode, RegisterOperand cls,
                bits<5> bytes, AddressingMode mode = bdaddr12only>
  : InstRS<opcode, (outs cls:$R1),
          (ins cls:$R1src, imm32zx4:$R3, mode:$BD2),
           mnemonic#"\t$R1, $R3, $BD2", []> {

  let Constraints = "$R1 = $R1src";
  let DisableEncoding = "$R1src";
  let mayLoad = 1;
  let AccessBytes = bytes;
}

class TernaryRSY<string mnemonic, bits<16> opcode, RegisterOperand cls,
                bits<5> bytes, AddressingMode mode = bdaddr20only>
  : InstRSY<opcode, (outs cls:$R1),
           (ins cls:$R1src, imm32zx4:$R3, mode:$BD2),
            mnemonic#"\t$R1, $R3, $BD2", []> {

  let Constraints = "$R1 = $R1src";
  let DisableEncoding = "$R1src";
  let mayLoad = 1;
  let AccessBytes = bytes;
}

multiclass TernaryRSPair<string mnemonic, bits<8> rsOpcode, bits<16> rsyOpcode,
                         RegisterOperand cls, bits<5> bytes> {
  let DispKey = mnemonic ## #cls in {
    let DispSize = "12" in
      def "" : TernaryRS<mnemonic, rsOpcode, cls, bytes, bdaddr12pair>;
    let DispSize = "20" in
      def Y  : TernaryRSY<mnemonic#"y", rsyOpcode, cls, bytes, bdaddr20pair>;
  }
}

class TernaryRXF<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                 RegisterOperand cls, SDPatternOperator load, bits<5> bytes>
  : InstRXF<opcode, (outs cls:$R1),
            (ins cls:$R1src, cls:$R3, bdxaddr12only:$XBD2),
            mnemonic#"\t$R1, $R3, $XBD2",
            [(set cls:$R1, (operator cls:$R1src, cls:$R3,
                                     (load bdxaddr12only:$XBD2)))]> {
  let OpKey = mnemonic ## cls;
  let OpType = "mem";
  let Constraints = "$R1 = $R1src";
  let DisableEncoding = "$R1src";
  let mayLoad = 1;
  let AccessBytes = bytes;
}

class TernaryVRIa<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                  TypedReg tr1, TypedReg tr2, Immediate imm, Immediate index>
  : InstVRIa<opcode, (outs tr1.op:$V1), (ins tr2.op:$V1src, imm:$I2, index:$M3),
             mnemonic#"\t$V1, $I2, $M3",
             [(set tr1.op:$V1, (tr1.vt (operator (tr2.vt tr2.op:$V1src),
                                                 imm:$I2, index:$M3)))]> {
  let Constraints = "$V1 = $V1src";
  let DisableEncoding = "$V1src";
}

class TernaryVRId<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                  TypedReg tr1, TypedReg tr2, bits<4> type>
  : InstVRId<opcode, (outs tr1.op:$V1),
             (ins tr2.op:$V2, tr2.op:$V3, imm32zx8:$I4),
             mnemonic#"\t$V1, $V2, $V3, $I4",
             [(set tr1.op:$V1, (tr1.vt (operator (tr2.vt tr2.op:$V2),
                                                 (tr2.vt tr2.op:$V3),
                                                 imm32zx8:$I4)))]> {
  let M5 = type;
}

class TernaryVRRa<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                  TypedReg tr1, TypedReg tr2, bits<4> type, bits<4> m4or>
  : InstVRRa<opcode, (outs tr1.op:$V1),
             (ins tr2.op:$V2, imm32zx4:$M4, imm32zx4:$M5),
             mnemonic#"\t$V1, $V2, $M4, $M5",
             [(set tr1.op:$V1, (tr1.vt (operator (tr2.vt tr2.op:$V2),
                                                 imm32zx4:$M4,
                                                 imm32zx4:$M5)))],
             m4or> {
  let M3 = type;
}

class TernaryVRRb<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                  TypedReg tr1, TypedReg tr2, bits<4> type,
                  SDPatternOperator m5mask, bits<4> m5or>
  : InstVRRb<opcode, (outs tr1.op:$V1),
             (ins tr2.op:$V2, tr2.op:$V3, m5mask:$M5),
             mnemonic#"\t$V1, $V2, $V3, $M5",
             [(set tr1.op:$V1, (tr1.vt (operator (tr2.vt tr2.op:$V2),
                                                 (tr2.vt tr2.op:$V3),
                                                 m5mask:$M5)))],
             m5or> {
  let M4 = type;
}

multiclass TernaryVRRbSPair<string mnemonic, bits<16> opcode,
                            SDPatternOperator operator,
                            SDPatternOperator operator_cc, TypedReg tr1,
                            TypedReg tr2, bits<4> type, bits<4> m5or> {
  def "" : TernaryVRRb<mnemonic, opcode, operator, tr1, tr2, type,
                       imm32zx4even, !and (m5or, 14)>;
  def : InstAlias<mnemonic#"\t$V1, $V2, $V3",
                  (!cast<Instruction>(NAME) tr1.op:$V1, tr2.op:$V2,
                                            tr2.op:$V3, 0)>;
  let Defs = [CC] in
    def S : TernaryVRRb<mnemonic##"s", opcode, operator_cc, tr1, tr2, type,
                        imm32zx4even, !add(!and (m5or, 14), 1)>;
  def : InstAlias<mnemonic#"s\t$V1, $V2, $V3",
                  (!cast<Instruction>(NAME#"S") tr1.op:$V1, tr2.op:$V2,
                                                tr2.op:$V3, 0)>;
}

class TernaryVRRc<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                  TypedReg tr1, TypedReg tr2>
  : InstVRRc<opcode, (outs tr1.op:$V1),
             (ins tr2.op:$V2, tr2.op:$V3, imm32zx4:$M4),
             mnemonic#"\t$V1, $V2, $V3, $M4",
             [(set tr1.op:$V1, (tr1.vt (operator (tr2.vt tr2.op:$V2),
                                                 (tr2.vt tr2.op:$V3),
                                                 imm32zx4:$M4)))]> {
  let M5 = 0;
  let M6 = 0;
}

class TernaryVRRd<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                  TypedReg tr1, TypedReg tr2, bits<4> type = 0>
  : InstVRRd<opcode, (outs tr1.op:$V1),
             (ins tr2.op:$V2, tr2.op:$V3, tr1.op:$V4),
             mnemonic#"\t$V1, $V2, $V3, $V4",
             [(set tr1.op:$V1, (tr1.vt (operator (tr2.vt tr2.op:$V2),
                                                 (tr2.vt tr2.op:$V3),
                                                 (tr1.vt tr1.op:$V4))))]> {
  let M5 = type;
  let M6 = 0;
}

class TernaryVRRe<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                  TypedReg tr1, TypedReg tr2, bits<4> m5 = 0, bits<4> type = 0>
  : InstVRRe<opcode, (outs tr1.op:$V1),
             (ins tr2.op:$V2, tr2.op:$V3, tr1.op:$V4),
             mnemonic#"\t$V1, $V2, $V3, $V4",
             [(set tr1.op:$V1, (tr1.vt (operator (tr2.vt tr2.op:$V2),
                                                 (tr2.vt tr2.op:$V3),
                                                 (tr1.vt tr1.op:$V4))))]> {
  let M5 = m5;
  let M6 = type;
}

class TernaryVRSb<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                  TypedReg tr1, TypedReg tr2, RegisterOperand cls, bits<4> type>
  : InstVRSb<opcode, (outs tr1.op:$V1),
             (ins tr2.op:$V1src, cls:$R3, shift12only:$BD2),
             mnemonic#"\t$V1, $R3, $BD2",
             [(set tr1.op:$V1, (tr1.vt (operator (tr2.vt tr2.op:$V1src),
                                                 cls:$R3,
                                                 shift12only:$BD2)))]> {
  let Constraints = "$V1 = $V1src";
  let DisableEncoding = "$V1src";
  let M4 = type;
}

class TernaryVRV<string mnemonic, bits<16> opcode, bits<5> bytes,
                 Immediate index>
  : InstVRV<opcode, (outs VR128:$V1),
           (ins VR128:$V1src, bdvaddr12only:$VBD2, index:$M3),
           mnemonic#"\t$V1, $VBD2, $M3", []> {
  let Constraints = "$V1 = $V1src";
  let DisableEncoding = "$V1src";
  let mayLoad = 1;
  let AccessBytes = bytes;
}

class TernaryVRX<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                 TypedReg tr1, TypedReg tr2, bits<5> bytes, Immediate index>
  : InstVRX<opcode, (outs tr1.op:$V1),
           (ins tr2.op:$V1src, bdxaddr12only:$XBD2, index:$M3),
           mnemonic#"\t$V1, $XBD2, $M3",
           [(set tr1.op:$V1, (tr1.vt (operator (tr2.vt tr2.op:$V1src),
                                               bdxaddr12only:$XBD2,
                                               index:$M3)))]> {
  let Constraints = "$V1 = $V1src";
  let DisableEncoding = "$V1src";
  let mayLoad = 1;
  let AccessBytes = bytes;
}

class QuaternaryVRId<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                     TypedReg tr1, TypedReg tr2, bits<4> type>
  : InstVRId<opcode, (outs tr1.op:$V1),
             (ins tr2.op:$V1src, tr2.op:$V2, tr2.op:$V3, imm32zx8:$I4),
             mnemonic#"\t$V1, $V2, $V3, $I4",
             [(set tr1.op:$V1, (tr1.vt (operator (tr2.vt tr2.op:$V1src),
                                                 (tr2.vt tr2.op:$V2),
                                                 (tr2.vt tr2.op:$V3),
                                                 imm32zx8:$I4)))]> {
  let Constraints = "$V1 = $V1src";
  let DisableEncoding = "$V1src";
  let M5 = type;
}

class QuaternaryVRRd<string mnemonic, bits<16> opcode,
                     SDPatternOperator operator, TypedReg tr1, TypedReg tr2,
                     bits<4> type, SDPatternOperator m6mask, bits<4> m6or>
  : InstVRRd<opcode, (outs tr1.op:$V1),
             (ins tr2.op:$V2, tr2.op:$V3, tr2.op:$V4, m6mask:$M6),
             mnemonic#"\t$V1, $V2, $V3, $V4, $M6",
             [(set tr1.op:$V1, (tr1.vt (operator (tr2.vt tr2.op:$V2),
                                                 (tr2.vt tr2.op:$V3),
                                                 (tr2.vt tr2.op:$V4),
                                                 m6mask:$M6)))],
             m6or> {
  let M5 = type;
}

multiclass QuaternaryVRRdSPair<string mnemonic, bits<16> opcode,
                               SDPatternOperator operator,
                               SDPatternOperator operator_cc, TypedReg tr1,
                               TypedReg tr2, bits<4> type, bits<4> m6or> {
  def "" : QuaternaryVRRd<mnemonic, opcode, operator, tr1, tr2, type,
                          imm32zx4even, !and (m6or, 14)>;
  def : InstAlias<mnemonic#"\t$V1, $V2, $V3, $V4",
                  (!cast<Instruction>(NAME) tr1.op:$V1, tr2.op:$V2,
                                            tr2.op:$V3, tr2.op:$V4, 0)>;
  let Defs = [CC] in
    def S : QuaternaryVRRd<mnemonic##"s", opcode, operator_cc, tr1, tr2, type,
                           imm32zx4even, !add (!and (m6or, 14), 1)>;
  def : InstAlias<mnemonic#"s\t$V1, $V2, $V3, $V4",
                  (!cast<Instruction>(NAME#"S") tr1.op:$V1, tr2.op:$V2,
                                                tr2.op:$V3, tr2.op:$V4, 0)>;
}

class LoadAndOpRSY<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                  RegisterOperand cls, AddressingMode mode = bdaddr20only>
  : InstRSY<opcode, (outs cls:$R1), (ins cls:$R3, mode:$BD2),
            mnemonic#"\t$R1, $R3, $BD2",
            [(set cls:$R1, (operator mode:$BD2, cls:$R3))]> {
  let mayLoad = 1;
  let mayStore = 1;
}

class CmpSwapRS<string mnemonic, bits<8> opcode, SDPatternOperator operator,
                RegisterOperand cls, AddressingMode mode = bdaddr12only>
  : InstRS<opcode, (outs cls:$R1), (ins cls:$R1src, cls:$R3, mode:$BD2),
           mnemonic#"\t$R1, $R3, $BD2",
           [(set cls:$R1, (operator mode:$BD2, cls:$R1src, cls:$R3))]> {
  let Constraints = "$R1 = $R1src";
  let DisableEncoding = "$R1src";
  let mayLoad = 1;
  let mayStore = 1;
}

class CmpSwapRSY<string mnemonic, bits<16> opcode, SDPatternOperator operator,
                 RegisterOperand cls, AddressingMode mode = bdaddr20only>
  : InstRSY<opcode, (outs cls:$R1), (ins cls:$R1src, cls:$R3, mode:$BD2),
            mnemonic#"\t$R1, $R3, $BD2",
            [(set cls:$R1, (operator mode:$BD2, cls:$R1src, cls:$R3))]> {
  let Constraints = "$R1 = $R1src";
  let DisableEncoding = "$R1src";
  let mayLoad = 1;
  let mayStore = 1;
}

multiclass CmpSwapRSPair<string mnemonic, bits<8> rsOpcode, bits<16> rsyOpcode,
                         SDPatternOperator operator, RegisterOperand cls> {
  let DispKey = mnemonic ## #cls in {
    let DispSize = "12" in
      def "" : CmpSwapRS<mnemonic, rsOpcode, operator, cls, bdaddr12pair>;
    let DispSize = "20" in
      def Y  : CmpSwapRSY<mnemonic#"y", rsyOpcode, operator, cls, bdaddr20pair>;
  }
}

class RotateSelectRIEf<string mnemonic, bits<16> opcode, RegisterOperand cls1,
                       RegisterOperand cls2>
  : InstRIEf<opcode, (outs cls1:$R1),
             (ins cls1:$R1src, cls2:$R2, imm32zx8:$I3, imm32zx8:$I4,
                  imm32zx6:$I5),
             mnemonic#"\t$R1, $R2, $I3, $I4, $I5", []> {
  let Constraints = "$R1 = $R1src";
  let DisableEncoding = "$R1src";
}

class PrefetchRXY<string mnemonic, bits<16> opcode, SDPatternOperator operator>
  : InstRXY<opcode, (outs), (ins imm32zx4:$R1, bdxaddr20only:$XBD2),
            mnemonic##"\t$R1, $XBD2",
            [(operator imm32zx4:$R1, bdxaddr20only:$XBD2)]>;

class PrefetchRILPC<string mnemonic, bits<12> opcode,
                    SDPatternOperator operator>
  : InstRIL<opcode, (outs), (ins imm32zx4:$R1, pcrel32:$I2),
            mnemonic##"\t$R1, $I2",
            [(operator imm32zx4:$R1, pcrel32:$I2)]> {
  // We want PC-relative addresses to be tried ahead of BD and BDX addresses.
  // However, BDXs have two extra operands and are therefore 6 units more
  // complex.
  let AddedComplexity = 7;
}

// A floating-point load-and test operation.  Create both a normal unary
// operation and one that acts as a comparison against zero.
// Note that the comparison against zero operation is not available if we
// have vector support, since load-and-test instructions will partially
// clobber the target (vector) register.
multiclass LoadAndTestRRE<string mnemonic, bits<16> opcode,
                          RegisterOperand cls> {
  def "" : UnaryRRE<mnemonic, opcode, null_frag, cls, cls>;
  let isCodeGenOnly = 1, Predicates = [FeatureNoVector] in
    def Compare : CompareRRE<mnemonic, opcode, null_frag, cls, cls>;
}

//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//
//
// Convenience instructions that get lowered to real instructions
// by either SystemZTargetLowering::EmitInstrWithCustomInserter()
// or SystemZInstrInfo::expandPostRAPseudo().
//
//===----------------------------------------------------------------------===//

class Pseudo<dag outs, dag ins, list<dag> pattern>
  : InstSystemZ<0, outs, ins, "", pattern> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

// Like UnaryRI, but expanded after RA depending on the choice of register.
class UnaryRIPseudo<SDPatternOperator operator, RegisterOperand cls,
                    Immediate imm>
  : Pseudo<(outs cls:$R1), (ins imm:$I2),
           [(set cls:$R1, (operator imm:$I2))]>;

// Like UnaryRXY, but expanded after RA depending on the choice of register.
class UnaryRXYPseudo<string key, SDPatternOperator operator,
                     RegisterOperand cls, bits<5> bytes,
                     AddressingMode mode = bdxaddr20only>
  : Pseudo<(outs cls:$R1), (ins mode:$XBD2),
           [(set cls:$R1, (operator mode:$XBD2))]> {
  let OpKey = key ## cls;
  let OpType = "mem";
  let mayLoad = 1;
  let Has20BitOffset = 1;
  let HasIndex = 1;
  let AccessBytes = bytes;
}

// Like UnaryRR, but expanded after RA depending on the choice of registers.
class UnaryRRPseudo<string key, SDPatternOperator operator,
                    RegisterOperand cls1, RegisterOperand cls2>
  : Pseudo<(outs cls1:$R1), (ins cls2:$R2),
           [(set cls1:$R1, (operator cls2:$R2))]> {
  let OpKey = key ## cls1;
  let OpType = "reg";
}

// Like BinaryRI, but expanded after RA depending on the choice of register.
class BinaryRIPseudo<SDPatternOperator operator, RegisterOperand cls,
                     Immediate imm>
  : Pseudo<(outs cls:$R1), (ins cls:$R1src, imm:$I2),
           [(set cls:$R1, (operator cls:$R1src, imm:$I2))]> {
  let Constraints = "$R1 = $R1src";
}

// Like BinaryRIE, but expanded after RA depending on the choice of register.
class BinaryRIEPseudo<SDPatternOperator operator, RegisterOperand cls,
                      Immediate imm>
  : Pseudo<(outs cls:$R1), (ins cls:$R3, imm:$I2),
           [(set cls:$R1, (operator cls:$R3, imm:$I2))]>;

// Like BinaryRIAndK, but expanded after RA depending on the choice of register.
multiclass BinaryRIAndKPseudo<string key, SDPatternOperator operator,
                              RegisterOperand cls, Immediate imm> {
  let NumOpsKey = key in {
    let NumOpsValue = "3" in
      def K : BinaryRIEPseudo<null_frag, cls, imm>,
              Requires<[FeatureHighWord, FeatureDistinctOps]>;
    let NumOpsValue = "2", isConvertibleToThreeAddress = 1 in
      def "" : BinaryRIPseudo<operator, cls, imm>,
               Requires<[FeatureHighWord]>;
  }
}

// Like CompareRI, but expanded after RA depending on the choice of register.
class CompareRIPseudo<SDPatternOperator operator, RegisterOperand cls,
                      Immediate imm>
  : Pseudo<(outs), (ins cls:$R1, imm:$I2), [(operator cls:$R1, imm:$I2)]>;

// Like CompareRXY, but expanded after RA depending on the choice of register.
class CompareRXYPseudo<SDPatternOperator operator, RegisterOperand cls,
                       SDPatternOperator load, bits<5> bytes,
                       AddressingMode mode = bdxaddr20only>
  : Pseudo<(outs), (ins cls:$R1, mode:$XBD2),
           [(operator cls:$R1, (load mode:$XBD2))]> {
  let mayLoad = 1;
  let Has20BitOffset = 1;
  let HasIndex = 1;
  let AccessBytes = bytes;
}

// Like StoreRXY, but expanded after RA depending on the choice of register.
class StoreRXYPseudo<SDPatternOperator operator, RegisterOperand cls,
                     bits<5> bytes, AddressingMode mode = bdxaddr20only>
  : Pseudo<(outs), (ins cls:$R1, mode:$XBD2),
           [(operator cls:$R1, mode:$XBD2)]> {
  let mayStore = 1;
  let Has20BitOffset = 1;
  let HasIndex = 1;
  let AccessBytes = bytes;
}

// Like RotateSelectRIEf, but expanded after RA depending on the choice
// of registers.
class RotateSelectRIEfPseudo<RegisterOperand cls1, RegisterOperand cls2>
  : Pseudo<(outs cls1:$R1),
           (ins cls1:$R1src, cls2:$R2, imm32zx8:$I3, imm32zx8:$I4,
                imm32zx6:$I5),
           []> {
  let Constraints = "$R1 = $R1src";
  let DisableEncoding = "$R1src";
}

// Implements "$dst = $cc & (8 >> CC) ? $src1 : $src2", where CC is
// the value of the PSW's 2-bit condition code field.
class SelectWrapper<RegisterOperand cls>
  : Pseudo<(outs cls:$dst),
           (ins cls:$src1, cls:$src2, imm32zx4:$valid, imm32zx4:$cc),
           [(set cls:$dst, (z_select_ccmask cls:$src1, cls:$src2,
                                            imm32zx4:$valid, imm32zx4:$cc))]> {
  let usesCustomInserter = 1;
  // Although the instructions used by these nodes do not in themselves
  // change CC, the insertion requires new blocks, and CC cannot be live
  // across them.
  let Defs = [CC];
  let Uses = [CC];
}

// Stores $new to $addr if $cc is true ("" case) or false (Inv case).
multiclass CondStores<RegisterOperand cls, SDPatternOperator store,
                      SDPatternOperator load, AddressingMode mode> {
  let Defs = [CC], Uses = [CC], usesCustomInserter = 1 in {
    def "" : Pseudo<(outs),
                    (ins cls:$new, mode:$addr, imm32zx4:$valid, imm32zx4:$cc),
                    [(store (z_select_ccmask cls:$new, (load mode:$addr),
                                             imm32zx4:$valid, imm32zx4:$cc),
                            mode:$addr)]>;
    def Inv : Pseudo<(outs),
                     (ins cls:$new, mode:$addr, imm32zx4:$valid, imm32zx4:$cc),
                     [(store (z_select_ccmask (load mode:$addr), cls:$new,
                                              imm32zx4:$valid, imm32zx4:$cc),
                              mode:$addr)]>;
  }
}

// OPERATOR is ATOMIC_SWAP or an ATOMIC_LOAD_* operation.  PAT and OPERAND
// describe the second (non-memory) operand.
class AtomicLoadBinary<SDPatternOperator operator, RegisterOperand cls,
                       dag pat, DAGOperand operand>
  : Pseudo<(outs cls:$dst), (ins bdaddr20only:$ptr, operand:$src2),
           [(set cls:$dst, (operator bdaddr20only:$ptr, pat))]> {
  let Defs = [CC];
  let Has20BitOffset = 1;
  let mayLoad = 1;
  let mayStore = 1;
  let usesCustomInserter = 1;
}

// Specializations of AtomicLoadWBinary.
class AtomicLoadBinaryReg32<SDPatternOperator operator>
  : AtomicLoadBinary<operator, GR32, (i32 GR32:$src2), GR32>;
class AtomicLoadBinaryImm32<SDPatternOperator operator, Immediate imm>
  : AtomicLoadBinary<operator, GR32, (i32 imm:$src2), imm>;
class AtomicLoadBinaryReg64<SDPatternOperator operator>
  : AtomicLoadBinary<operator, GR64, (i64 GR64:$src2), GR64>;
class AtomicLoadBinaryImm64<SDPatternOperator operator, Immediate imm>
  : AtomicLoadBinary<operator, GR64, (i64 imm:$src2), imm>;

// OPERATOR is ATOMIC_SWAPW or an ATOMIC_LOADW_* operation.  PAT and OPERAND
// describe the second (non-memory) operand.
class AtomicLoadWBinary<SDPatternOperator operator, dag pat,
                        DAGOperand operand>
  : Pseudo<(outs GR32:$dst),
           (ins bdaddr20only:$ptr, operand:$src2, ADDR32:$bitshift,
                ADDR32:$negbitshift, uimm32:$bitsize),
           [(set GR32:$dst, (operator bdaddr20only:$ptr, pat, ADDR32:$bitshift,
                                      ADDR32:$negbitshift, uimm32:$bitsize))]> {
  let Defs = [CC];
  let Has20BitOffset = 1;
  let mayLoad = 1;
  let mayStore = 1;
  let usesCustomInserter = 1;
}

// Specializations of AtomicLoadWBinary.
class AtomicLoadWBinaryReg<SDPatternOperator operator>
  : AtomicLoadWBinary<operator, (i32 GR32:$src2), GR32>;
class AtomicLoadWBinaryImm<SDPatternOperator operator, Immediate imm>
  : AtomicLoadWBinary<operator, (i32 imm:$src2), imm>;

// Define an instruction that operates on two fixed-length blocks of memory,
// and associated pseudo instructions for operating on blocks of any size.
// The Sequence form uses a straight-line sequence of instructions and
// the Loop form uses a loop of length-256 instructions followed by
// another instruction to handle the excess.
multiclass MemorySS<string mnemonic, bits<8> opcode,
                    SDPatternOperator sequence, SDPatternOperator loop> {
  def "" : InstSS<opcode, (outs), (ins bdladdr12onlylen8:$BDL1,
                                       bdaddr12only:$BD2),
                  mnemonic##"\t$BDL1, $BD2", []>;
  let usesCustomInserter = 1 in {
    def Sequence : Pseudo<(outs), (ins bdaddr12only:$dest, bdaddr12only:$src,
                                       imm64:$length),
                           [(sequence bdaddr12only:$dest, bdaddr12only:$src,
                                      imm64:$length)]>;
    def Loop : Pseudo<(outs), (ins bdaddr12only:$dest, bdaddr12only:$src,
                                   imm64:$length, GR64:$count256),
                      [(loop bdaddr12only:$dest, bdaddr12only:$src,
                             imm64:$length, GR64:$count256)]>;
  }
}

// Define an instruction that operates on two strings, both terminated
// by the character in R0.  The instruction processes a CPU-determinated
// number of bytes at a time and sets CC to 3 if the instruction needs
// to be repeated.  Also define a pseudo instruction that represents
// the full loop (the main instruction plus the branch on CC==3).
multiclass StringRRE<string mnemonic, bits<16> opcode,
                     SDPatternOperator operator> {
  def "" : InstRRE<opcode, (outs GR64:$R1, GR64:$R2),
                   (ins GR64:$R1src, GR64:$R2src),
                   mnemonic#"\t$R1, $R2", []> {
    let Uses = [R0L];
    let Constraints = "$R1 = $R1src, $R2 = $R2src";
    let DisableEncoding = "$R1src, $R2src";
  }
  let usesCustomInserter = 1 in
    def Loop : Pseudo<(outs GR64:$end),
                      (ins GR64:$start1, GR64:$start2, GR32:$char),
                      [(set GR64:$end, (operator GR64:$start1, GR64:$start2,
                                                 GR32:$char))]>;
}

// A pseudo instruction that is a direct alias of a real instruction.
// These aliases are used in cases where a particular register operand is
// fixed or where the same instruction is used with different register sizes.
// The size parameter is the size in bytes of the associated real instruction.
class Alias<int size, dag outs, dag ins, list<dag> pattern>
  : InstSystemZ<size, outs, ins, "", pattern> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

class UnaryAliasVRS<RegisterOperand cls1, RegisterOperand cls2>
 : Alias<6, (outs cls1:$src1), (ins cls2:$src2), []>;

// An alias of a UnaryVRR*, but with different register sizes.
class UnaryAliasVRR<SDPatternOperator operator, TypedReg tr1, TypedReg tr2>
  : Alias<6, (outs tr1.op:$V1), (ins tr2.op:$V2),
          [(set tr1.op:$V1, (tr1.vt (operator (tr2.vt tr2.op:$V2))))]>;

// An alias of a UnaryVRX, but with different register sizes.
class UnaryAliasVRX<SDPatternOperator operator, TypedReg tr,
                    AddressingMode mode = bdxaddr12only>
  : Alias<6, (outs tr.op:$V1), (ins mode:$XBD2),
          [(set tr.op:$V1, (tr.vt (operator mode:$XBD2)))]>;

// An alias of a StoreVRX, but with different register sizes.
class StoreAliasVRX<SDPatternOperator operator, TypedReg tr,
                    AddressingMode mode = bdxaddr12only>
  : Alias<6, (outs), (ins tr.op:$V1, mode:$XBD2),
          [(operator (tr.vt tr.op:$V1), mode:$XBD2)]>;

// An alias of a BinaryRI, but with different register sizes.
class BinaryAliasRI<SDPatternOperator operator, RegisterOperand cls,
                    Immediate imm>
  : Alias<4, (outs cls:$R1), (ins cls:$R1src, imm:$I2),
          [(set cls:$R1, (operator cls:$R1src, imm:$I2))]> {
  let Constraints = "$R1 = $R1src";
}

// An alias of a BinaryRIL, but with different register sizes.
class BinaryAliasRIL<SDPatternOperator operator, RegisterOperand cls,
                     Immediate imm>
  : Alias<6, (outs cls:$R1), (ins cls:$R1src, imm:$I2),
          [(set cls:$R1, (operator cls:$R1src, imm:$I2))]> {
  let Constraints = "$R1 = $R1src";
}

// An alias of a BinaryVRRf, but with different register sizes.
class BinaryAliasVRRf<RegisterOperand cls>
  : Alias<6, (outs VR128:$V1), (ins cls:$R2, cls:$R3), []>;

// An alias of a CompareRI, but with different register sizes.
class CompareAliasRI<SDPatternOperator operator, RegisterOperand cls,
                     Immediate imm>
  : Alias<4, (outs), (ins cls:$R1, imm:$I2), [(operator cls:$R1, imm:$I2)]> {
  let isCompare = 1;
}

// An alias of a RotateSelectRIEf, but with different register sizes.
class RotateSelectAliasRIEf<RegisterOperand cls1, RegisterOperand cls2>
  : Alias<6, (outs cls1:$R1),
          (ins cls1:$R1src, cls2:$R2, imm32zx8:$I3, imm32zx8:$I4,
               imm32zx6:$I5), []> {
  let Constraints = "$R1 = $R1src";
}
