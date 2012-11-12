//===- CodeEmitterGen.cpp - Code Emitter Generator ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// CodeEmitterGen uses the descriptions of instructions and their fields to
// construct an automated code emitter: a function that, given a MachineInstr,
// returns the (currently, 32-bit unsigned) value of the instruction.
//
//===----------------------------------------------------------------------===//

#include "CodeGenTarget.h"
#include "llvm/TableGen/Record.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <map>
#include <string>
#include <vector>
using namespace llvm;

// FIXME: Somewhat hackish to use a command line option for this. There should
// be a CodeEmitter class in the Target.td that controls this sort of thing
// instead.
static cl::opt<bool>
MCEmitter("mc-emitter",
          cl::desc("Generate CodeEmitter for use with the MC library."),
          cl::init(false));

namespace {

class CodeEmitterGen {
  RecordKeeper &Records;
public:
  CodeEmitterGen(RecordKeeper &R) : Records(R) {}

  void run(raw_ostream &o);
private:
  void emitMachineOpEmitter(raw_ostream &o, const std::string &Namespace);
  void emitGetValueBit(raw_ostream &o, const std::string &Namespace);
  void reverseBits(std::vector<Record*> &Insts);
  int getVariableBit(const std::string &VarName, BitsInit *BI, int bit);
  std::string getInstructionCase(Record *R, CodeGenTarget &Target);
  void AddCodeToMergeInOperand(Record *R, BitsInit *BI,
                               const std::string &VarName,
                               unsigned &NumberedOp,
                               std::string &Case, CodeGenTarget &Target);

};

void CodeEmitterGen::reverseBits(std::vector<Record*> &Insts) {
  for (std::vector<Record*>::iterator I = Insts.begin(), E = Insts.end();
       I != E; ++I) {
    Record *R = *I;
    if (R->getValueAsString("Namespace") == "TargetOpcode" ||
        R->getValueAsBit("isPseudo"))
      continue;

    BitsInit *BI = R->getValueAsBitsInit("Inst");

    unsigned numBits = BI->getNumBits();
 
    SmallVector<Init *, 16> NewBits(numBits);
 
    for (unsigned bit = 0, end = numBits / 2; bit != end; ++bit) {
      unsigned bitSwapIdx = numBits - bit - 1;
      Init *OrigBit = BI->getBit(bit);
      Init *BitSwap = BI->getBit(bitSwapIdx);
      NewBits[bit]        = BitSwap;
      NewBits[bitSwapIdx] = OrigBit;
    }
    if (numBits % 2) {
      unsigned middle = (numBits + 1) / 2;
      NewBits[middle] = BI->getBit(middle);
    }

    BitsInit *NewBI = BitsInit::get(NewBits);

    // Update the bits in reversed order so that emitInstrOpBits will get the
    // correct endianness.
    R->getValue("Inst")->setValue(NewBI);
  }
}

// If the VarBitInit at position 'bit' matches the specified variable then
// return the variable bit position.  Otherwise return -1.
int CodeEmitterGen::getVariableBit(const std::string &VarName,
                                   BitsInit *BI, int bit) {
  if (VarBitInit *VBI = dyn_cast<VarBitInit>(BI->getBit(bit))) {
    if (VarInit *VI = dyn_cast<VarInit>(VBI->getBitVar()))
      if (VI->getName() == VarName)
        return VBI->getBitNum();
  } else if (VarInit *VI = dyn_cast<VarInit>(BI->getBit(bit))) {
    if (VI->getName() == VarName)
      return 0;
  }

  return -1;
}

void CodeEmitterGen::
AddCodeToMergeInOperand(Record *R, BitsInit *BI, const std::string &VarName,
                        unsigned &NumberedOp,
                        std::string &Case, CodeGenTarget &Target) {
  CodeGenInstruction &CGI = Target.getInstruction(R);

  // Determine if VarName actually contributes to the Inst encoding.
  int bit = BI->getNumBits()-1;

  // Scan for a bit that this contributed to.
  for (; bit >= 0; ) {
    if (getVariableBit(VarName, BI, bit) != -1)
      break;
    
    --bit;
  }
  
  // If we found no bits, ignore this value, otherwise emit the call to get the
  // operand encoding.
  if (bit < 0) return;
  
  // If the operand matches by name, reference according to that
  // operand number. Non-matching operands are assumed to be in
  // order.
  unsigned OpIdx;
  if (CGI.Operands.hasOperandNamed(VarName, OpIdx)) {
    // Get the machine operand number for the indicated operand.
    OpIdx = CGI.Operands[OpIdx].MIOperandNo;
    assert(!CGI.Operands.isFlatOperandNotEmitted(OpIdx) &&
           "Explicitly used operand also marked as not emitted!");
  } else {
    unsigned NumberOps = CGI.Operands.size();
    /// If this operand is not supposed to be emitted by the
    /// generated emitter, skip it.
    while (NumberedOp < NumberOps &&
           CGI.Operands.isFlatOperandNotEmitted(NumberedOp))
      ++NumberedOp;

    OpIdx = NumberedOp++;
  }
  
  std::pair<unsigned, unsigned> SO = CGI.Operands.getSubOperandNumber(OpIdx);
  std::string &EncoderMethodName = CGI.Operands[SO.first].EncoderMethodName;
  
  // If the source operand has a custom encoder, use it. This will
  // get the encoding for all of the suboperands.
  if (!EncoderMethodName.empty()) {
    // A custom encoder has all of the information for the
    // sub-operands, if there are more than one, so only
    // query the encoder once per source operand.
    if (SO.second == 0) {
      Case += "      // op: " + VarName + "\n" +
              "      op = " + EncoderMethodName + "(MI, " + utostr(OpIdx);
      if (MCEmitter)
        Case += ", Fixups";
      Case += ");\n";
    }
  } else {
    Case += "      // op: " + VarName + "\n" +
      "      op = getMachineOpValue(MI, MI.getOperand(" + utostr(OpIdx) + ")";
    if (MCEmitter)
      Case += ", Fixups";
    Case += ");\n";
  }
  
  for (; bit >= 0; ) {
    int varBit = getVariableBit(VarName, BI, bit);
    
    // If this bit isn't from a variable, skip it.
    if (varBit == -1) {
      --bit;
      continue;
    }
    
    // Figure out the consecutive range of bits covered by this operand, in
    // order to generate better encoding code.
    int beginInstBit = bit;
    int beginVarBit = varBit;
    int N = 1;
    for (--bit; bit >= 0;) {
      varBit = getVariableBit(VarName, BI, bit);
      if (varBit == -1 || varBit != (beginVarBit - N)) break;
      ++N;
      --bit;
    }
     
    uint64_t opMask = ~(uint64_t)0 >> (64-N);
    int opShift = beginVarBit - N + 1;
    opMask <<= opShift;
    opShift = beginInstBit - beginVarBit;
    
    if (opShift > 0) {
      Case += "      Value |= (op & UINT64_C(" + utostr(opMask) + ")) << " +
              itostr(opShift) + ";\n";
    } else if (opShift < 0) {
      Case += "      Value |= (op & UINT64_C(" + utostr(opMask) + ")) >> " + 
              itostr(-opShift) + ";\n";
    } else {
      Case += "      Value |= op & UINT64_C(" + utostr(opMask) + ");\n";
    }
  }
}


std::string CodeEmitterGen::getInstructionCase(Record *R,
                                               CodeGenTarget &Target) {
  std::string Case;
  
  BitsInit *BI = R->getValueAsBitsInit("Inst");
  const std::vector<RecordVal> &Vals = R->getValues();
  unsigned NumberedOp = 0;

  // Loop over all of the fields in the instruction, determining which are the
  // operands to the instruction.
  for (unsigned i = 0, e = Vals.size(); i != e; ++i) {
    // Ignore fixed fields in the record, we're looking for values like:
    //    bits<5> RST = { ?, ?, ?, ?, ? };
    if (Vals[i].getPrefix() || Vals[i].getValue()->isComplete())
      continue;
    
    AddCodeToMergeInOperand(R, BI, Vals[i].getName(), NumberedOp, Case, Target);
  }
  
  std::string PostEmitter = R->getValueAsString("PostEncoderMethod");
  if (!PostEmitter.empty())
    Case += "      Value = " + PostEmitter + "(MI, Value);\n";
  
  return Case;
}

void CodeEmitterGen::run(raw_ostream &o) {
  CodeGenTarget Target(Records);
  std::vector<Record*> Insts = Records.getAllDerivedDefinitions("Instruction");

  // For little-endian instruction bit encodings, reverse the bit order
  if (Target.isLittleEndianEncoding()) reverseBits(Insts);


  const std::vector<const CodeGenInstruction*> &NumberedInstructions =
    Target.getInstructionsByEnumValue();

  // Emit function declaration
  o << "uint64_t " << Target.getName();
  if (MCEmitter)
    o << "MCCodeEmitter::getBinaryCodeForInstr(const MCInst &MI,\n"
      << "    SmallVectorImpl<MCFixup> &Fixups) const {\n";
  else
    o << "CodeEmitter::getBinaryCodeForInstr(const MachineInstr &MI) const {\n";

  // Emit instruction base values
  o << "  static const uint64_t InstBits[] = {\n";
  for (std::vector<const CodeGenInstruction*>::const_iterator
          IN = NumberedInstructions.begin(),
          EN = NumberedInstructions.end();
       IN != EN; ++IN) {
    const CodeGenInstruction *CGI = *IN;
    Record *R = CGI->TheDef;

    if (R->getValueAsString("Namespace") == "TargetOpcode" ||
        R->getValueAsBit("isPseudo")) {
      o << "    UINT64_C(0),\n";
      continue;
    }

    BitsInit *BI = R->getValueAsBitsInit("Inst");

    // Start by filling in fixed values.
    uint64_t Value = 0;
    for (unsigned i = 0, e = BI->getNumBits(); i != e; ++i) {
      if (BitInit *B = dyn_cast<BitInit>(BI->getBit(e-i-1)))
        Value |= (uint64_t)B->getValue() << (e-i-1);
    }
    o << "    UINT64_C(" << Value << ")," << '\t' << "// " << R->getName() << "\n";
  }
  o << "    UINT64_C(0)\n  };\n";

  // Map to accumulate all the cases.
  std::map<std::string, std::vector<std::string> > CaseMap;

  // Construct all cases statement for each opcode
  for (std::vector<Record*>::iterator IC = Insts.begin(), EC = Insts.end();
        IC != EC; ++IC) {
    Record *R = *IC;
    if (R->getValueAsString("Namespace") == "TargetOpcode" ||
        R->getValueAsBit("isPseudo"))
      continue;
    const std::string &InstName = R->getValueAsString("Namespace") + "::"
      + R->getName();
    std::string Case = getInstructionCase(R, Target);

    CaseMap[Case].push_back(InstName);
  }

  // Emit initial function code
  o << "  const unsigned opcode = MI.getOpcode();\n"
    << "  uint64_t Value = InstBits[opcode];\n"
    << "  uint64_t op = 0;\n"
    << "  (void)op;  // suppress warning\n"
    << "  switch (opcode) {\n";

  // Emit each case statement
  std::map<std::string, std::vector<std::string> >::iterator IE, EE;
  for (IE = CaseMap.begin(), EE = CaseMap.end(); IE != EE; ++IE) {
    const std::string &Case = IE->first;
    std::vector<std::string> &InstList = IE->second;

    for (int i = 0, N = InstList.size(); i < N; i++) {
      if (i) o << "\n";
      o << "    case " << InstList[i]  << ":";
    }
    o << " {\n";
    o << Case;
    o << "      break;\n"
      << "    }\n";
  }

  // Default case: unhandled opcode
  o << "  default:\n"
    << "    std::string msg;\n"
    << "    raw_string_ostream Msg(msg);\n"
    << "    Msg << \"Not supported instr: \" << MI;\n"
    << "    report_fatal_error(Msg.str());\n"
    << "  }\n"
    << "  return Value;\n"
    << "}\n\n";
}

} // End anonymous namespace

namespace llvm {

void EmitCodeEmitter(RecordKeeper &RK, raw_ostream &OS) {
  emitSourceFileHeader("Machine Code Emitter", OS);
  CodeEmitterGen(RK).run(OS);
}

} // End llvm namespace
