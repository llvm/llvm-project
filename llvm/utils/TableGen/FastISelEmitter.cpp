///===- FastISelEmitter.cpp - Generate an instruction selector ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits code for use by the "fast" instruction
// selection algorithm. See the comments at the top of
// lib/CodeGen/SelectionDAG/FastISel.cpp for background.
//
// This file scans through the target's tablegen instruction-info files
// and extracts instructions with obvious-looking patterns, and it emits
// code to look up these instructions by type and operator.
//
//===----------------------------------------------------------------------===//

#include "Common/CodeGenDAGPatterns.h"
#include "Common/CodeGenInstruction.h"
#include "Common/CodeGenRegisters.h"
#include "Common/CodeGenTarget.h"
#include "Common/InfoByHwMode.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <set>
#include <utility>
using namespace llvm;

/// InstructionMemo - This class holds additional information about an
/// instruction needed to emit code for it.
///
namespace {
struct InstructionMemo {
  StringRef Name;
  const CodeGenRegisterClass *RC;
  std::string SubRegNo;
  std::vector<std::string> PhysRegs;
  std::string PredicateCheck;

  InstructionMemo(StringRef Name, const CodeGenRegisterClass *RC,
                  std::string SubRegNo, std::vector<std::string> PhysRegs,
                  std::string PredicateCheck)
      : Name(Name), RC(RC), SubRegNo(std::move(SubRegNo)),
        PhysRegs(std::move(PhysRegs)),
        PredicateCheck(std::move(PredicateCheck)) {}

  // Make sure we do not copy InstructionMemo.
  InstructionMemo(const InstructionMemo &Other) = delete;
  InstructionMemo(InstructionMemo &&Other) = default;
};

/// ImmPredicateSet - This uniques predicates (represented as a string) and
/// gives them unique (small) integer ID's that start at 0.
class ImmPredicateSet {
  DenseMap<TreePattern *, unsigned> ImmIDs;
  std::vector<TreePredicateFn> PredsByName;

public:
  unsigned getIDFor(TreePredicateFn Pred) {
    unsigned &Entry = ImmIDs[Pred.getOrigPatFragRecord()];
    if (Entry == 0) {
      PredsByName.push_back(Pred);
      Entry = PredsByName.size();
    }
    return Entry - 1;
  }

  const TreePredicateFn &getPredicate(unsigned Idx) { return PredsByName[Idx]; }

  typedef std::vector<TreePredicateFn>::const_iterator iterator;
  iterator begin() const { return PredsByName.begin(); }
  iterator end() const { return PredsByName.end(); }
};

/// OperandsSignature - This class holds a description of a list of operand
/// types. It has utility methods for emitting text based on the operands.
///
struct OperandsSignature {
  class OpKind {
    enum { OK_Reg, OK_FP, OK_Imm, OK_Invalid = -1 };
    char Repr = OK_Invalid;

  public:
    OpKind() {}

    bool operator<(OpKind RHS) const { return Repr < RHS.Repr; }
    bool operator==(OpKind RHS) const { return Repr == RHS.Repr; }

    static OpKind getReg() {
      OpKind K;
      K.Repr = OK_Reg;
      return K;
    }
    static OpKind getFP() {
      OpKind K;
      K.Repr = OK_FP;
      return K;
    }
    static OpKind getImm(unsigned V) {
      assert((unsigned)OK_Imm + V < 128 &&
             "Too many integer predicates for the 'Repr' char");
      OpKind K;
      K.Repr = OK_Imm + V;
      return K;
    }

    bool isReg() const { return Repr == OK_Reg; }
    bool isFP() const { return Repr == OK_FP; }
    bool isImm() const { return Repr >= OK_Imm; }

    unsigned getImmCode() const {
      assert(isImm());
      return Repr - OK_Imm;
    }

    void printManglingSuffix(raw_ostream &OS, ImmPredicateSet &ImmPredicates,
                             bool StripImmCodes) const {
      if (isReg())
        OS << 'r';
      else if (isFP())
        OS << 'f';
      else {
        OS << 'i';
        if (!StripImmCodes)
          if (unsigned Code = getImmCode())
            OS << "_" << ImmPredicates.getPredicate(Code - 1).getFnName();
      }
    }
  };

  SmallVector<OpKind, 3> Operands;

  bool operator<(const OperandsSignature &O) const {
    return Operands < O.Operands;
  }
  bool operator==(const OperandsSignature &O) const {
    return Operands == O.Operands;
  }

  bool empty() const { return Operands.empty(); }

  bool hasAnyImmediateCodes() const {
    return llvm::any_of(Operands, [](OpKind Kind) {
      return Kind.isImm() && Kind.getImmCode() != 0;
    });
  }

  /// getWithoutImmCodes - Return a copy of this with any immediate codes forced
  /// to zero.
  OperandsSignature getWithoutImmCodes() const {
    OperandsSignature Result;
    Result.Operands.resize(Operands.size());
    llvm::transform(Operands, Result.Operands.begin(), [](OpKind Kind) {
      return Kind.isImm() ? OpKind::getImm(0) : Kind;
    });
    return Result;
  }

  void emitImmediatePredicate(raw_ostream &OS,
                              ImmPredicateSet &ImmPredicates) const {
    ListSeparator LS(" &&\n        ");
    for (auto [Idx, Opnd] : enumerate(Operands)) {
      if (!Opnd.isImm())
        continue;

      unsigned Code = Opnd.getImmCode();
      if (Code == 0)
        continue;

      TreePredicateFn PredFn = ImmPredicates.getPredicate(Code - 1);

      // Emit the type check.
      TreePattern *TP = PredFn.getOrigPatFragRecord();
      ValueTypeByHwMode VVT = TP->getTree(0)->getType(0);
      assert(VVT.isSimple() &&
             "Cannot use variable value types with fast isel");
      OS << LS << "VT == " << getEnumName(VVT.getSimple().SimpleTy) << " && ";

      OS << PredFn.getFnName() << "(imm" << Idx << ')';
    }
  }

  /// initialize - Examine the given pattern and initialize the contents
  /// of the Operands array accordingly. Return true if all the operands
  /// are supported, false otherwise.
  ///
  bool initialize(TreePatternNode &InstPatNode, const CodeGenTarget &Target,
                  MVT::SimpleValueType VT, ImmPredicateSet &ImmediatePredicates,
                  const CodeGenRegisterClass *OrigDstRC) {
    if (InstPatNode.isLeaf())
      return false;

    if (InstPatNode.getOperator()->getName() == "imm") {
      Operands.push_back(OpKind::getImm(0));
      return true;
    }

    if (InstPatNode.getOperator()->getName() == "fpimm") {
      Operands.push_back(OpKind::getFP());
      return true;
    }

    const CodeGenRegisterClass *DstRC = nullptr;

    for (const TreePatternNode &Op : InstPatNode.children()) {
      // Handle imm operands specially.
      if (!Op.isLeaf() && Op.getOperator()->getName() == "imm") {
        unsigned PredNo = 0;
        if (!Op.getPredicateCalls().empty()) {
          TreePredicateFn PredFn = Op.getPredicateCalls()[0].Fn;
          // If there is more than one predicate weighing in on this operand
          // then we don't handle it.  This doesn't typically happen for
          // immediates anyway.
          if (Op.getPredicateCalls().size() > 1 ||
              !PredFn.isImmediatePattern() || PredFn.usesOperands())
            return false;
          // Ignore any instruction with 'FastIselShouldIgnore', these are
          // not needed and just bloat the fast instruction selector.  For
          // example, X86 doesn't need to generate code to match ADD16ri8 since
          // ADD16ri will do just fine.
          const Record *Rec = PredFn.getOrigPatFragRecord()->getRecord();
          if (Rec->getValueAsBit("FastIselShouldIgnore"))
            return false;

          PredNo = ImmediatePredicates.getIDFor(PredFn) + 1;
        }

        Operands.push_back(OpKind::getImm(PredNo));
        continue;
      }

      // For now, filter out any operand with a predicate.
      // For now, filter out any operand with multiple values.
      if (!Op.getPredicateCalls().empty() || Op.getNumTypes() != 1)
        return false;

      if (!Op.isLeaf()) {
        if (Op.getOperator()->getName() == "fpimm") {
          Operands.push_back(OpKind::getFP());
          continue;
        }
        // For now, ignore other non-leaf nodes.
        return false;
      }

      assert(Op.hasConcreteType(0) && "Type infererence not done?");

      // For now, all the operands must have the same type (if they aren't
      // immediates).  Note that this causes us to reject variable sized shifts
      // on X86.
      if (Op.getSimpleType(0) != VT)
        return false;

      const DefInit *OpDI = dyn_cast<DefInit>(Op.getLeafValue());
      if (!OpDI)
        return false;
      const Record *OpLeafRec = OpDI->getDef();

      // For now, the only other thing we accept is register operands.
      const CodeGenRegisterClass *RC = nullptr;
      if (OpLeafRec->isSubClassOf("RegisterOperand"))
        OpLeafRec = OpLeafRec->getValueAsDef("RegClass");
      if (OpLeafRec->isSubClassOf("RegisterClass"))
        RC = &Target.getRegisterClass(OpLeafRec);
      else if (OpLeafRec->isSubClassOf("Register"))
        RC = Target.getRegBank().getRegClassForRegister(OpLeafRec);
      else if (OpLeafRec->isSubClassOf("ValueType"))
        RC = OrigDstRC;
      else
        return false;

      // For now, this needs to be a register class of some sort.
      if (!RC)
        return false;

      // For now, all the operands must have the same register class or be
      // a strict subclass of the destination.
      if (DstRC) {
        if (DstRC != RC && !DstRC->hasSubClass(RC))
          return false;
      } else {
        DstRC = RC;
      }
      Operands.push_back(OpKind::getReg());
    }
    return true;
  }

  void PrintParameters(raw_ostream &OS) const {
    ListSeparator LS;
    for (auto [Idx, Opnd] : enumerate(Operands)) {
      OS << LS;
      if (Opnd.isReg())
        OS << "Register Op" << Idx;
      else if (Opnd.isImm())
        OS << "uint64_t imm" << Idx;
      else if (Opnd.isFP())
        OS << "const ConstantFP *f" << Idx;
      else
        llvm_unreachable("Unknown operand kind!");
    }
  }

  void PrintArguments(raw_ostream &OS, ArrayRef<std::string> PhyRegs) const {
    ListSeparator LS;
    for (auto [Idx, Opnd, PhyReg] : enumerate(Operands, PhyRegs)) {
      if (!PhyReg.empty()) {
        // Implicit physical register operand.
        continue;
      }

      OS << LS;
      if (Opnd.isReg())
        OS << "Op" << Idx;
      else if (Opnd.isImm())
        OS << "imm" << Idx;
      else if (Opnd.isFP())
        OS << "f" << Idx;
      else
        llvm_unreachable("Unknown operand kind!");
    }
  }

  void PrintArguments(raw_ostream &OS) const {
    ListSeparator LS;
    for (auto [Idx, Opnd] : enumerate(Operands)) {
      OS << LS;
      if (Opnd.isReg())
        OS << "Op" << Idx;
      else if (Opnd.isImm())
        OS << "imm" << Idx;
      else if (Opnd.isFP())
        OS << "f" << Idx;
      else
        llvm_unreachable("Unknown operand kind!");
    }
  }

  void PrintManglingSuffix(raw_ostream &OS, ArrayRef<std::string> PhyRegs,
                           ImmPredicateSet &ImmPredicates,
                           bool StripImmCodes = false) const {
    for (auto [PhyReg, Opnd] : zip_equal(PhyRegs, Operands)) {
      if (!PhyReg.empty()) {
        // Implicit physical register operand. e.g. Instruction::Mul expect to
        // select to a binary op. On x86, mul may take a single operand with
        // the other operand being implicit. We must emit something that looks
        // like a binary instruction except for the very inner fastEmitInst_*
        // call.
        continue;
      }
      Opnd.printManglingSuffix(OS, ImmPredicates, StripImmCodes);
    }
  }

  void PrintManglingSuffix(raw_ostream &OS, ImmPredicateSet &ImmPredicates,
                           bool StripImmCodes = false) const {
    for (OpKind Opnd : Operands)
      Opnd.printManglingSuffix(OS, ImmPredicates, StripImmCodes);
  }
};

class FastISelMap {
  // A multimap is needed instead of a "plain" map because the key is
  // the instruction's complexity (an int) and they are not unique.
  typedef std::multimap<int, InstructionMemo> PredMap;
  typedef std::map<MVT::SimpleValueType, PredMap> RetPredMap;
  typedef std::map<MVT::SimpleValueType, RetPredMap> TypeRetPredMap;
  typedef std::map<StringRef, TypeRetPredMap> OpcodeTypeRetPredMap;
  typedef std::map<OperandsSignature, OpcodeTypeRetPredMap>
      OperandsOpcodeTypeRetPredMap;

  OperandsOpcodeTypeRetPredMap SimplePatterns;

  // This is used to check that there are no duplicate predicates
  std::set<std::tuple<OperandsSignature, StringRef, MVT::SimpleValueType,
                      MVT::SimpleValueType, std::string>>
      SimplePatternsCheck;

  std::map<OperandsSignature, std::vector<OperandsSignature>>
      SignaturesWithConstantForms;

  StringRef InstNS;
  ImmPredicateSet ImmediatePredicates;

public:
  explicit FastISelMap(StringRef InstNS);

  void collectPatterns(const CodeGenDAGPatterns &CGP);
  void printImmediatePredicates(raw_ostream &OS);
  void printFunctionDefinitions(raw_ostream &OS);

private:
  void emitInstructionCode(raw_ostream &OS, const OperandsSignature &Operands,
                           const PredMap &PM, StringRef RetVTName);
};
} // End anonymous namespace

static std::string getLegalCName(StringRef OpName) {
  std::string CName = OpName.str();
  std::string::size_type Pos = CName.find("::");
  if (Pos != std::string::npos)
    CName.replace(Pos, 2, "_");
  return CName;
}

FastISelMap::FastISelMap(StringRef instns) : InstNS(instns) {}

static std::string PhysRegForNode(const TreePatternNode &Op,
                                  const CodeGenTarget &Target) {
  std::string PhysReg;

  if (!Op.isLeaf())
    return PhysReg;

  const Record *OpLeafRec = cast<DefInit>(Op.getLeafValue())->getDef();
  if (!OpLeafRec->isSubClassOf("Register"))
    return PhysReg;

  PhysReg += cast<StringInit>(OpLeafRec->getValue("Namespace")->getValue())
                 ->getValue();
  PhysReg += "::";
  PhysReg += Target.getRegBank().getReg(OpLeafRec)->getName();
  return PhysReg;
}

void FastISelMap::collectPatterns(const CodeGenDAGPatterns &CGP) {
  const CodeGenTarget &Target = CGP.getTargetInfo();

  // Scan through all the patterns and record the simple ones.
  for (const PatternToMatch &Pattern : CGP.ptms()) {
    // For now, just look at Instructions, so that we don't have to worry
    // about emitting multiple instructions for a pattern.
    TreePatternNode &Dst = Pattern.getDstPattern();
    if (Dst.isLeaf())
      continue;
    const Record *Op = Dst.getOperator();
    if (!Op->isSubClassOf("Instruction"))
      continue;
    CodeGenInstruction &Inst = CGP.getTargetInfo().getInstruction(Op);
    if (Inst.Operands.empty())
      continue;

    // Allow instructions to be marked as unavailable for FastISel for
    // certain cases, i.e. an ISA has two 'and' instruction which differ
    // by what registers they can use but are otherwise identical for
    // codegen purposes.
    if (Inst.FastISelShouldIgnore)
      continue;

    // For now, ignore multi-instruction patterns.
    bool MultiInsts = false;
    for (const TreePatternNode &ChildOp : Dst.children()) {
      if (ChildOp.isLeaf())
        continue;
      if (ChildOp.getOperator()->isSubClassOf("Instruction")) {
        MultiInsts = true;
        break;
      }
    }
    if (MultiInsts)
      continue;

    // For now, ignore instructions where the first operand is not an
    // output register.
    const CodeGenRegisterClass *DstRC = nullptr;
    std::string SubRegNo;
    if (Op->getName() != "EXTRACT_SUBREG") {
      const Record *Op0Rec = Inst.Operands[0].Rec;
      if (Op0Rec->isSubClassOf("RegisterOperand"))
        Op0Rec = Op0Rec->getValueAsDef("RegClass");
      if (!Op0Rec->isSubClassOf("RegisterClass"))
        continue;
      DstRC = &Target.getRegisterClass(Op0Rec);
      if (!DstRC)
        continue;
    } else {
      // If this isn't a leaf, then continue since the register classes are
      // a bit too complicated for now.
      if (!Dst.getChild(1).isLeaf())
        continue;

      const DefInit *SR = dyn_cast<DefInit>(Dst.getChild(1).getLeafValue());
      if (SR)
        SubRegNo = getQualifiedName(SR->getDef());
      else
        SubRegNo = Dst.getChild(1).getLeafValue()->getAsString();
    }

    // Inspect the pattern.
    TreePatternNode &InstPatNode = Pattern.getSrcPattern();
    if (InstPatNode.isLeaf())
      continue;

    // Ignore multiple result nodes for now.
    if (InstPatNode.getNumTypes() > 1)
      continue;

    const Record *InstPatOp = InstPatNode.getOperator();
    StringRef OpcodeName = CGP.getSDNodeInfo(InstPatOp).getEnumName();
    MVT::SimpleValueType RetVT = MVT::isVoid;
    if (InstPatNode.getNumTypes())
      RetVT = InstPatNode.getSimpleType(0);
    MVT::SimpleValueType VT = RetVT;
    if (InstPatNode.getNumChildren()) {
      assert(InstPatNode.getChild(0).getNumTypes() == 1);
      VT = InstPatNode.getChild(0).getSimpleType(0);
    }

    // For now, filter out any instructions with predicates.
    if (!InstPatNode.getPredicateCalls().empty())
      continue;

    // Check all the operands.
    OperandsSignature Operands;
    if (!Operands.initialize(InstPatNode, Target, VT, ImmediatePredicates,
                             DstRC))
      continue;

    std::vector<std::string> PhysRegInputs;
    if (InstPatNode.getOperator()->getName() == "imm" ||
        InstPatNode.getOperator()->getName() == "fpimm")
      PhysRegInputs.push_back("");
    else {
      // Compute the PhysRegs used by the given pattern, and check that
      // the mapping from the src to dst patterns is simple.
      bool FoundNonSimplePattern = false;
      unsigned DstIndex = 0;
      for (const TreePatternNode &SrcChild : InstPatNode.children()) {
        std::string PhysReg = PhysRegForNode(SrcChild, Target);
        if (PhysReg.empty()) {
          if (DstIndex >= Dst.getNumChildren() ||
              Dst.getChild(DstIndex).getName() != SrcChild.getName()) {
            FoundNonSimplePattern = true;
            break;
          }
          ++DstIndex;
        }

        PhysRegInputs.push_back(std::move(PhysReg));
      }

      if (Op->getName() != "EXTRACT_SUBREG" && DstIndex < Dst.getNumChildren())
        FoundNonSimplePattern = true;

      if (FoundNonSimplePattern)
        continue;
    }

    // Check if the operands match one of the patterns handled by FastISel.
    std::string ManglingSuffix;
    raw_string_ostream SuffixOS(ManglingSuffix);
    Operands.PrintManglingSuffix(SuffixOS, ImmediatePredicates, true);
    if (!StringSwitch<bool>(ManglingSuffix)
             .Cases({"", "r", "rr", "ri", "i", "f"}, true)
             .Default(false))
      continue;

    // Get the predicate that guards this pattern.
    std::string PredicateCheck = Pattern.getPredicateCheck();

    // Ok, we found a pattern that we can handle. Remember it.
    InstructionMemo Memo(Pattern.getDstPattern().getOperator()->getName(),
                         DstRC, std::move(SubRegNo), std::move(PhysRegInputs),
                         PredicateCheck);

    int Complexity = Pattern.getPatternComplexity(CGP);

    auto inserted_simple_pattern = SimplePatternsCheck.insert(
        {Operands, OpcodeName, VT, RetVT, PredicateCheck});
    if (!inserted_simple_pattern.second) {
      PrintFatalError(Pattern.getSrcRecord()->getLoc(),
                      "Duplicate predicate in FastISel table!");
    }

    // Note: Instructions with the same complexity will appear in the order
    // that they are encountered.
    SimplePatterns[Operands][OpcodeName][VT][RetVT].emplace(Complexity,
                                                            std::move(Memo));

    // If any of the operands were immediates with predicates on them, strip
    // them down to a signature that doesn't have predicates so that we can
    // associate them with the stripped predicate version.
    if (Operands.hasAnyImmediateCodes()) {
      SignaturesWithConstantForms[Operands.getWithoutImmCodes()].push_back(
          Operands);
    }
  }
}

void FastISelMap::printImmediatePredicates(raw_ostream &OS) {
  if (ImmediatePredicates.begin() == ImmediatePredicates.end())
    return;

  OS << "\n// FastEmit Immediate Predicate functions.\n";
  for (auto ImmediatePredicate : ImmediatePredicates) {
    OS << "static bool " << ImmediatePredicate.getFnName()
       << "(int64_t Imm) {\n";
    OS << ImmediatePredicate.getImmediatePredicateCode() << "\n}\n";
  }

  OS << "\n\n";
}

void FastISelMap::emitInstructionCode(raw_ostream &OS,
                                      const OperandsSignature &Operands,
                                      const PredMap &PM, StringRef RetVTName) {
  // Emit code for each possible instruction. There may be
  // multiple if there are subtarget concerns.  A reverse iterator
  // is used to produce the ones with highest complexity first.

  bool OneHadNoPredicate = false;
  for (const auto &[_, Memo] : reverse(PM)) {
    std::string PredicateCheck = Memo.PredicateCheck;

    if (PredicateCheck.empty()) {
      assert(!OneHadNoPredicate &&
             "Multiple instructions match and more than one had "
             "no predicate!");
      OneHadNoPredicate = true;
    } else {
      if (OneHadNoPredicate) {
        PrintFatalError("Multiple instructions match and one with no "
                        "predicate came before one with a predicate!  "
                        "name:" +
                        Memo.Name + "  predicate: " + PredicateCheck);
      }
      OS << "  if (" + PredicateCheck + ") {\n";
      OS << "  ";
    }

    for (auto [Idx, PhyReg] : enumerate(Memo.PhysRegs)) {
      if (!PhyReg.empty())
        OS << "  BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, MIMD, "
           << "TII.get(TargetOpcode::COPY), " << PhyReg << ").addReg(Op" << Idx
           << ");\n";
    }

    OS << "  return fastEmitInst_";
    if (Memo.SubRegNo.empty()) {
      Operands.PrintManglingSuffix(OS, Memo.PhysRegs, ImmediatePredicates,
                                   true);
      OS << "(" << InstNS << "::" << Memo.Name << ", ";
      OS << "&" << InstNS << "::" << Memo.RC->getName() << "RegClass";
      if (!Operands.empty())
        OS << ", ";
      Operands.PrintArguments(OS, Memo.PhysRegs);
      OS << ");\n";
    } else {
      OS << "extractsubreg(" << RetVTName << ", Op0, " << Memo.SubRegNo
         << ");\n";
    }

    if (!PredicateCheck.empty())
      OS << "  }\n";
  }
  // Return Register() if all of the possibilities had predicates but none
  // were satisfied.
  if (!OneHadNoPredicate)
    OS << "  return Register();\n";
  OS << "}\n";
  OS << "\n";
}

void FastISelMap::printFunctionDefinitions(raw_ostream &OS) {
  // Now emit code for all the patterns that we collected.
  for (const auto &SimplePattern : SimplePatterns) {
    const OperandsSignature &Operands = SimplePattern.first;
    const OpcodeTypeRetPredMap &OTM = SimplePattern.second;

    for (const auto &[Opcode, TM] : OTM) {
      OS << "// FastEmit functions for " << Opcode << ".\n";
      OS << "\n";

      // Emit one function for each opcode,type pair.
      for (const auto &[VT, RM] : TM) {
        if (RM.size() != 1) {
          for (const auto &[RetVT, PM] : RM) {
            OS << "Register fastEmit_" << getLegalCName(Opcode) << "_"
               << getLegalCName(getEnumName(VT)) << "_"
               << getLegalCName(getEnumName(RetVT)) << "_";
            Operands.PrintManglingSuffix(OS, ImmediatePredicates);
            OS << "(";
            Operands.PrintParameters(OS);
            OS << ") {\n";

            emitInstructionCode(OS, Operands, PM, getEnumName(RetVT));
          }

          // Emit one function for the type that demultiplexes on return type.
          OS << "Register fastEmit_" << getLegalCName(Opcode) << "_"
             << getLegalCName(getEnumName(VT)) << "_";
          Operands.PrintManglingSuffix(OS, ImmediatePredicates);
          OS << "(MVT RetVT";
          if (!Operands.empty())
            OS << ", ";
          Operands.PrintParameters(OS);
          OS << ") {\nswitch (RetVT.SimpleTy) {\n";
          for (const auto &[RetVT, _] : RM) {
            OS << "  case " << getEnumName(RetVT) << ": return fastEmit_"
               << getLegalCName(Opcode) << "_" << getLegalCName(getEnumName(VT))
               << "_" << getLegalCName(getEnumName(RetVT)) << "_";
            Operands.PrintManglingSuffix(OS, ImmediatePredicates);
            OS << "(";
            Operands.PrintArguments(OS);
            OS << ");\n";
          }
          OS << "  default: return Register();\n}\n}\n\n";

        } else {
          // Non-variadic return type.
          OS << "Register fastEmit_" << getLegalCName(Opcode) << "_"
             << getLegalCName(getEnumName(VT)) << "_";
          Operands.PrintManglingSuffix(OS, ImmediatePredicates);
          OS << "(MVT RetVT";
          if (!Operands.empty())
            OS << ", ";
          Operands.PrintParameters(OS);
          OS << ") {\n";

          OS << "  if (RetVT.SimpleTy != " << getEnumName(RM.begin()->first)
             << ")\n    return Register();\n";

          const PredMap &PM = RM.begin()->second;

          emitInstructionCode(OS, Operands, PM, "RetVT");
        }
      }

      // Emit one function for the opcode that demultiplexes based on the type.
      OS << "Register fastEmit_" << getLegalCName(Opcode) << "_";
      Operands.PrintManglingSuffix(OS, ImmediatePredicates);
      OS << "(MVT VT, MVT RetVT";
      if (!Operands.empty())
        OS << ", ";
      Operands.PrintParameters(OS);
      OS << ") {\n";
      OS << "  switch (VT.SimpleTy) {\n";
      for (const auto &[VT, _] : TM) {
        StringRef TypeName = getEnumName(VT);
        OS << "  case " << TypeName << ": return fastEmit_"
           << getLegalCName(Opcode) << "_" << getLegalCName(TypeName) << "_";
        Operands.PrintManglingSuffix(OS, ImmediatePredicates);
        OS << "(RetVT";
        if (!Operands.empty())
          OS << ", ";
        Operands.PrintArguments(OS);
        OS << ");\n";
      }
      OS << "  default: return Register();\n";
      OS << "  }\n";
      OS << "}\n";
      OS << "\n";
    }

    OS << "// Top-level FastEmit function.\n";
    OS << "\n";

    // Emit one function for the operand signature that demultiplexes based
    // on opcode and type.
    OS << "Register fastEmit_";
    Operands.PrintManglingSuffix(OS, ImmediatePredicates);
    OS << "(MVT VT, MVT RetVT, unsigned Opcode";
    if (!Operands.empty())
      OS << ", ";
    Operands.PrintParameters(OS);
    OS << ") ";
    if (!Operands.hasAnyImmediateCodes())
      OS << "override ";
    OS << "{\n";

    // If there are any forms of this signature available that operate on
    // constrained forms of the immediate (e.g., 32-bit sext immediate in a
    // 64-bit operand), check them first.

    std::map<OperandsSignature, std::vector<OperandsSignature>>::iterator MI =
        SignaturesWithConstantForms.find(Operands);
    if (MI != SignaturesWithConstantForms.end()) {
      // Unique any duplicates out of the list.
      llvm::sort(MI->second);
      MI->second.erase(llvm::unique(MI->second), MI->second.end());

      // Check each in order it was seen.  It would be nice to have a good
      // relative ordering between them, but we're not going for optimality
      // here.
      for (const OperandsSignature &Sig : MI->second) {
        OS << "  if (";
        Sig.emitImmediatePredicate(OS, ImmediatePredicates);
        OS << ")\n    if (Register Reg = fastEmit_";
        Sig.PrintManglingSuffix(OS, ImmediatePredicates);
        OS << "(VT, RetVT, Opcode";
        if (!Sig.empty())
          OS << ", ";
        Sig.PrintArguments(OS);
        OS << "))\n      return Reg;\n\n";
      }

      // Done with this, remove it.
      SignaturesWithConstantForms.erase(MI);
    }

    OS << "  switch (Opcode) {\n";
    for (const auto &[Opcode, _] : OTM) {
      OS << "  case " << Opcode << ": return fastEmit_" << getLegalCName(Opcode)
         << "_";
      Operands.PrintManglingSuffix(OS, ImmediatePredicates);
      OS << "(VT, RetVT";
      if (!Operands.empty())
        OS << ", ";
      Operands.PrintArguments(OS);
      OS << ");\n";
    }
    OS << "  default: return Register();\n";
    OS << "  }\n";
    OS << "}\n";
    OS << "\n";
  }

  // TODO: SignaturesWithConstantForms should be empty here.
}

static void EmitFastISel(const RecordKeeper &RK, raw_ostream &OS) {
  const CodeGenDAGPatterns CGP(RK);
  const CodeGenTarget &Target = CGP.getTargetInfo();
  emitSourceFileHeader("\"Fast\" Instruction Selector for the " +
                           Target.getName().str() + " target",
                       OS);

  // Determine the target's namespace name.
  StringRef InstNS = Target.getInstNamespace();
  assert(!InstNS.empty() && "Can't determine target-specific namespace!");

  FastISelMap F(InstNS);
  F.collectPatterns(CGP);
  F.printImmediatePredicates(OS);
  F.printFunctionDefinitions(OS);
}

static TableGen::Emitter::Opt X("gen-fast-isel", EmitFastISel,
                                "Generate a \"fast\" instruction selector");
