//===- utils/TableGen/X86EVEX2VEXTablesEmitter.cpp - X86 backend-*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// This tablegen backend is responsible for emitting the X86 backend EVEX2VEX
/// compression tables.
///
//===----------------------------------------------------------------------===//

#include "CodeGenDAGPatterns.h"
#include "CodeGenTarget.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

namespace {

class X86EVEX2VEXTablesEmitter {
  CodeGenTarget Target;

  // Hold all non-masked & non-broadcasted EVEX encoded instructions
  std::vector<const CodeGenInstruction *> EVEXInsts;
  // Hold all VEX encoded instructions. Divided into groups with same opcodes
  // to make the search more efficient
  std::map<uint64_t, std::vector<const CodeGenInstruction *>> VEXInsts;

  typedef std::pair<const CodeGenInstruction *, const CodeGenInstruction *> Entry;

  // Represent both compress tables
  std::vector<Entry> EVEX2VEX128;
  std::vector<Entry> EVEX2VEX256;

  // Represents a manually added entry to the tables
  struct ManualEntry {
    const char *EVEXInstStr;
    const char *VEXInstStr;
    bool Is128Bit;
  };

public:
  X86EVEX2VEXTablesEmitter(RecordKeeper &R) : Target(R) {}

  // run - Output X86 EVEX2VEX tables.
  void run(raw_ostream &OS);

private:
  // Prints the given table as a C++ array of type
  // X86EvexToVexCompressTableEntry
  void printTable(const std::vector<Entry> &Table, raw_ostream &OS);

  bool inExceptionList(const CodeGenInstruction *Inst) {
    // List of EVEX instructions that match VEX instructions by the encoding
    // but do not perform the same operation.
    static constexpr const char *ExceptionList[] = {
        "VCVTQQ2PD",
        "VCVTQQ2PS",
        "VPMAXSQ",
        "VPMAXUQ",
        "VPMINSQ",
        "VPMINUQ",
        "VPMULLQ",
        "VPSRAQ",
        "VDBPSADBW",
        "VRNDSCALE",
        "VSCALEFPS"
    };
    // Instruction's name starts with one of the entries in the exception list
    for (StringRef InstStr : ExceptionList) {
      if (Inst->TheDef->getName().startswith(InstStr))
        return true;
    }
    return false;
  }

};

void X86EVEX2VEXTablesEmitter::printTable(const std::vector<Entry> &Table,
                                          raw_ostream &OS) {
  std::string Size = (Table == EVEX2VEX128) ? "128" : "256";

  OS << "// X86 EVEX encoded instructions that have a VEX " << Size
     << " encoding\n"
     << "// (table format: <EVEX opcode, VEX-" << Size << " opcode>).\n"
     << "static const X86EvexToVexCompressTableEntry X86EvexToVex" << Size
     << "CompressTable[] = {\n"
     << "  // EVEX scalar with corresponding VEX.\n";

  // Print all entries added to the table
  for (auto Pair : Table) {
    OS << "  { X86::" << Pair.first->TheDef->getName()
       << ", X86::" << Pair.second->TheDef->getName() << " },\n";
  }

  // Some VEX instructions were duplicated to multiple EVEX versions due the
  // introduction of mask variants, and thus some of the EVEX versions have
  // different encoding than the VEX instruction. In order to maximize the
  // compression we add these entries manually.
  static constexpr ManualEntry ManuallyAddedEntries[] = {
      // EVEX-Inst            VEX-Inst           Is128-bit
      {"VMOVDQU8Z128mr",      "VMOVDQUmr",       true},
      {"VMOVDQU8Z128rm",      "VMOVDQUrm",       true},
      {"VMOVDQU8Z128rr",      "VMOVDQUrr",       true},
      {"VMOVDQU8Z128rr_REV",  "VMOVDQUrr_REV",   true},
      {"VMOVDQU16Z128mr",     "VMOVDQUmr",       true},
      {"VMOVDQU16Z128rm",     "VMOVDQUrm",       true},
      {"VMOVDQU16Z128rr",     "VMOVDQUrr",       true},
      {"VMOVDQU16Z128rr_REV", "VMOVDQUrr_REV",   true},
      {"VMOVDQU8Z256mr",      "VMOVDQUYmr",      false},
      {"VMOVDQU8Z256rm",      "VMOVDQUYrm",      false},
      {"VMOVDQU8Z256rr",      "VMOVDQUYrr",      false},
      {"VMOVDQU8Z256rr_REV",  "VMOVDQUYrr_REV",  false},
      {"VMOVDQU16Z256mr",     "VMOVDQUYmr",      false},
      {"VMOVDQU16Z256rm",     "VMOVDQUYrm",      false},
      {"VMOVDQU16Z256rr",     "VMOVDQUYrr",      false},
      {"VMOVDQU16Z256rr_REV", "VMOVDQUYrr_REV",  false},

      {"VPERMILPDZ128mi",     "VPERMILPDmi",     true},
      {"VPERMILPDZ128ri",     "VPERMILPDri",     true},
      {"VPERMILPDZ128rm",     "VPERMILPDrm",     true},
      {"VPERMILPDZ128rr",     "VPERMILPDrr",     true},
      {"VPERMILPDZ256mi",     "VPERMILPDYmi",    false},
      {"VPERMILPDZ256ri",     "VPERMILPDYri",    false},
      {"VPERMILPDZ256rm",     "VPERMILPDYrm",    false},
      {"VPERMILPDZ256rr",     "VPERMILPDYrr",    false},

      {"VPBROADCASTQZ128m",   "VPBROADCASTQrm",  true},
      {"VPBROADCASTQZ128r",   "VPBROADCASTQrr",  true},
      {"VPBROADCASTQZ256m",   "VPBROADCASTQYrm", false},
      {"VPBROADCASTQZ256r",   "VPBROADCASTQYrr", false},

      {"VBROADCASTSDZ256m",   "VBROADCASTSDYrm", false},
      {"VBROADCASTSDZ256r",   "VBROADCASTSDYrr", false},

      {"VBROADCASTF64X2Z128rm", "VBROADCASTF128", false},
      {"VBROADCASTI64X2Z128rm", "VBROADCASTI128", false},

      {"VEXTRACTF64x2Z256mr", "VEXTRACTF128mr",  false},
      {"VEXTRACTF64x2Z256rr", "VEXTRACTF128rr",  false},
      {"VEXTRACTI64x2Z256mr", "VEXTRACTI128mr",  false},
      {"VEXTRACTI64x2Z256rr", "VEXTRACTI128rr",  false},

      {"VINSERTF64x2Z256rm",  "VINSERTF128rm",   false},
      {"VINSERTF64x2Z256rr",  "VINSERTF128rr",   false},
      {"VINSERTI64x2Z256rm",  "VINSERTI128rm",   false},
      {"VINSERTI64x2Z256rr",  "VINSERTI128rr",   false},

      // These will require some custom adjustment in the conversion pass.
      {"VALIGNDZ128rri",      "VPALIGNRrri",     true},
      {"VALIGNQZ128rri",      "VPALIGNRrri",     true},
      {"VALIGNDZ128rmi",      "VPALIGNRrmi",     true},
      {"VALIGNQZ128rmi",      "VPALIGNRrmi",     true},
      {"VSHUFF32X4Z256rmi",   "VPERM2F128rm",    false},
      {"VSHUFF32X4Z256rri",   "VPERM2F128rr",    false},
      {"VSHUFF64X2Z256rmi",   "VPERM2F128rm",    false},
      {"VSHUFF64X2Z256rri",   "VPERM2F128rr",    false},
      {"VSHUFI32X4Z256rmi",   "VPERM2I128rm",    false},
      {"VSHUFI32X4Z256rri",   "VPERM2I128rr",    false},
      {"VSHUFI64X2Z256rmi",   "VPERM2I128rm",    false},
      {"VSHUFI64X2Z256rri",   "VPERM2I128rr",    false},
  };

  // Print the manually added entries
  for (const ManualEntry &Entry : ManuallyAddedEntries) {
    if ((Table == EVEX2VEX128 && Entry.Is128Bit) ||
        (Table == EVEX2VEX256 && !Entry.Is128Bit)) {
      OS << "  { X86::" << Entry.EVEXInstStr << ", X86::" << Entry.VEXInstStr
         << " },\n";
    }
  }

  OS << "};\n\n";
}

// Return true if the 2 BitsInits are equal
static inline bool equalBitsInits(const BitsInit *B1, const BitsInit *B2) {
  if (B1->getNumBits() != B2->getNumBits())
    PrintFatalError("Comparing two BitsInits with different sizes!");

  for (unsigned i = 0, e = B1->getNumBits(); i != e; ++i) {
    if (BitInit *Bit1 = dyn_cast<BitInit>(B1->getBit(i))) {
      if (BitInit *Bit2 = dyn_cast<BitInit>(B2->getBit(i))) {
        if (Bit1->getValue() != Bit2->getValue())
          return false;
      } else
        PrintFatalError("Invalid BitsInit bit");
    } else
      PrintFatalError("Invalid BitsInit bit");
  }
  return true;
}

// Calculates the integer value residing BitsInit object
static inline uint64_t getValueFromBitsInit(const BitsInit *B) {
  uint64_t Value = 0;
  for (unsigned i = 0, e = B->getNumBits(); i != e; ++i) {
    if (BitInit *Bit = dyn_cast<BitInit>(B->getBit(i)))
      Value |= uint64_t(Bit->getValue()) << i;
    else
      PrintFatalError("Invalid VectSize bit");
  }
  return Value;
}

// Function object - Operator() returns true if the given VEX instruction
// matches the EVEX instruction of this object.
class IsMatch {
  const CodeGenInstruction *Inst;

public:
  IsMatch(const CodeGenInstruction *Inst) : Inst(Inst) {}

  bool operator()(const CodeGenInstruction *Inst2) {
    Record *Rec1 = Inst->TheDef;
    Record *Rec2 = Inst2->TheDef;
    uint64_t Rec1WVEX =
        getValueFromBitsInit(Rec1->getValueAsBitsInit("VEX_WPrefix"));
    uint64_t Rec2WVEX =
        getValueFromBitsInit(Rec2->getValueAsBitsInit("VEX_WPrefix"));

    if (Rec2->getValueAsDef("OpEnc")->getName().str() != "EncVEX" ||
        // VEX/EVEX fields
        Rec2->getValueAsDef("OpPrefix") != Rec1->getValueAsDef("OpPrefix") ||
        Rec2->getValueAsDef("OpMap") != Rec1->getValueAsDef("OpMap") ||
        Rec2->getValueAsBit("hasVEX_4V") != Rec1->getValueAsBit("hasVEX_4V") ||
        !equalBitsInits(Rec2->getValueAsBitsInit("EVEX_LL"),
                        Rec1->getValueAsBitsInit("EVEX_LL")) ||
        (Rec1WVEX != 2 && Rec2WVEX != 2 && Rec1WVEX != Rec2WVEX) ||
        // Instruction's format
        Rec2->getValueAsDef("Form") != Rec1->getValueAsDef("Form") ||
        Rec2->getValueAsBit("isAsmParserOnly") !=
            Rec1->getValueAsBit("isAsmParserOnly"))
      return false;

    // This is needed for instructions with intrinsic version (_Int).
    // Where the only difference is the size of the operands.
    // For example: VUCOMISDZrm and Int_VUCOMISDrm
    // Also for instructions that their EVEX version was upgraded to work with
    // k-registers. For example VPCMPEQBrm (xmm output register) and
    // VPCMPEQBZ128rm (k register output register).
    for (unsigned i = 0; i < Inst->Operands.size(); i++) {
      Record *OpRec1 = Inst->Operands[i].Rec;
      Record *OpRec2 = Inst2->Operands[i].Rec;

      if (OpRec1 == OpRec2)
        continue;

      if (isRegisterOperand(OpRec1) && isRegisterOperand(OpRec2)) {
        if (getRegOperandSize(OpRec1) != getRegOperandSize(OpRec2))
          return false;
      } else if (isMemoryOperand(OpRec1) && isMemoryOperand(OpRec2)) {
        return false;
      } else if (isImmediateOperand(OpRec1) && isImmediateOperand(OpRec2)) {
        if (OpRec1->getValueAsDef("Type") != OpRec2->getValueAsDef("Type"))
          return false;
      } else
        return false;
    }

    return true;
  }

private:
  static inline bool isRegisterOperand(const Record *Rec) {
    return Rec->isSubClassOf("RegisterClass") ||
           Rec->isSubClassOf("RegisterOperand");
  }

  static inline bool isMemoryOperand(const Record *Rec) {
    return Rec->isSubClassOf("Operand") &&
           Rec->getValueAsString("OperandType") == "OPERAND_MEMORY";
  }

  static inline bool isImmediateOperand(const Record *Rec) {
    return Rec->isSubClassOf("Operand") &&
           Rec->getValueAsString("OperandType") == "OPERAND_IMMEDIATE";
  }

  static inline unsigned int getRegOperandSize(const Record *RegRec) {
    if (RegRec->isSubClassOf("RegisterClass"))
      return RegRec->getValueAsInt("Alignment");
    if (RegRec->isSubClassOf("RegisterOperand"))
      return RegRec->getValueAsDef("RegClass")->getValueAsInt("Alignment");

    llvm_unreachable("Register operand's size not known!");
  }
};

void X86EVEX2VEXTablesEmitter::run(raw_ostream &OS) {
  emitSourceFileHeader("X86 EVEX2VEX tables", OS);

  ArrayRef<const CodeGenInstruction *> NumberedInstructions =
      Target.getInstructionsByEnumValue();

  for (const CodeGenInstruction *Inst : NumberedInstructions) {
    // Filter non-X86 instructions.
    if (!Inst->TheDef->isSubClassOf("X86Inst"))
      continue;

    // Add VEX encoded instructions to one of VEXInsts vectors according to
    // it's opcode.
    if (Inst->TheDef->getValueAsDef("OpEnc")->getName() == "EncVEX") {
      uint64_t Opcode = getValueFromBitsInit(Inst->TheDef->
                                             getValueAsBitsInit("Opcode"));
      VEXInsts[Opcode].push_back(Inst);
    }
    // Add relevant EVEX encoded instructions to EVEXInsts
    else if (Inst->TheDef->getValueAsDef("OpEnc")->getName() == "EncEVEX" &&
             !Inst->TheDef->getValueAsBit("hasEVEX_K") &&
             !Inst->TheDef->getValueAsBit("hasEVEX_B") &&
             getValueFromBitsInit(Inst->TheDef->
                                        getValueAsBitsInit("EVEX_LL")) != 2 &&
             !inExceptionList(Inst))
      EVEXInsts.push_back(Inst);
  }

  for (const CodeGenInstruction *EVEXInst : EVEXInsts) {
    uint64_t Opcode = getValueFromBitsInit(EVEXInst->TheDef->
                                           getValueAsBitsInit("Opcode"));
    // For each EVEX instruction look for a VEX match in the appropriate vector
    // (instructions with the same opcode) using function object IsMatch.
    auto Match = llvm::find_if(VEXInsts[Opcode], IsMatch(EVEXInst));
    if (Match != VEXInsts[Opcode].end()) {
      const CodeGenInstruction *VEXInst = *Match;

      // In case a match is found add new entry to the appropriate table
      switch (getValueFromBitsInit(
          EVEXInst->TheDef->getValueAsBitsInit("EVEX_LL"))) {
      case 0:
        EVEX2VEX128.push_back(std::make_pair(EVEXInst, VEXInst)); // {0,0}
        break;
      case 1:
        EVEX2VEX256.push_back(std::make_pair(EVEXInst, VEXInst)); // {0,1}
        break;
      default:
        llvm_unreachable("Instruction's size not fit for the mapping!");
      }
    }
  }

  // Print both tables
  printTable(EVEX2VEX128, OS);
  printTable(EVEX2VEX256, OS);
}
}

namespace llvm {
void EmitX86EVEX2VEXTables(RecordKeeper &RK, raw_ostream &OS) {
  X86EVEX2VEXTablesEmitter(RK).run(OS);
}
}
