//===- DAGISelMatcherEmitter.cpp - Matcher Emitter ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains code to generate C++ code for a matcher.
//
//===----------------------------------------------------------------------===//

#include "DAGISelMatcher.h"
#include "CodeGenDAGPatterns.h"
#include "llvm/TableGen/Record.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
using namespace llvm;

enum {
  CommentIndent = 30
};

// To reduce generated source code size.
static cl::opt<bool>
OmitComments("omit-comments", cl::desc("Do not generate comments"),
             cl::init(false));

namespace {
class MatcherTableEmitter {
  const CodeGenDAGPatterns &CGP;
  
  DenseMap<TreePattern *, unsigned> NodePredicateMap;
  std::vector<TreePredicateFn> NodePredicates;
  
  StringMap<unsigned> PatternPredicateMap;
  std::vector<std::string> PatternPredicates;

  DenseMap<const ComplexPattern*, unsigned> ComplexPatternMap;
  std::vector<const ComplexPattern*> ComplexPatterns;


  DenseMap<Record*, unsigned> NodeXFormMap;
  std::vector<Record*> NodeXForms;

public:
  MatcherTableEmitter(const CodeGenDAGPatterns &cgp)
    : CGP(cgp) {}

  unsigned EmitMatcherList(const Matcher *N, unsigned Indent,
                           unsigned StartIdx, formatted_raw_ostream &OS);

  void EmitPredicateFunctions(formatted_raw_ostream &OS);

  void EmitHistogram(const Matcher *N, formatted_raw_ostream &OS);
private:
  unsigned EmitMatcher(const Matcher *N, unsigned Indent, unsigned CurrentIdx,
                       formatted_raw_ostream &OS);

  unsigned getNodePredicate(TreePredicateFn Pred) {
    unsigned &Entry = NodePredicateMap[Pred.getOrigPatFragRecord()];
    if (Entry == 0) {
      NodePredicates.push_back(Pred);
      Entry = NodePredicates.size();
    }
    return Entry-1;
  }
  
  unsigned getPatternPredicate(StringRef PredName) {
    unsigned &Entry = PatternPredicateMap[PredName];
    if (Entry == 0) {
      PatternPredicates.push_back(PredName.str());
      Entry = PatternPredicates.size();
    }
    return Entry-1;
  }
  unsigned getComplexPat(const ComplexPattern &P) {
    unsigned &Entry = ComplexPatternMap[&P];
    if (Entry == 0) {
      ComplexPatterns.push_back(&P);
      Entry = ComplexPatterns.size();
    }
    return Entry-1;
  }

  unsigned getNodeXFormID(Record *Rec) {
    unsigned &Entry = NodeXFormMap[Rec];
    if (Entry == 0) {
      NodeXForms.push_back(Rec);
      Entry = NodeXForms.size();
    }
    return Entry-1;
  }

};
} // end anonymous namespace.

static unsigned GetVBRSize(unsigned Val) {
  if (Val <= 127) return 1;

  unsigned NumBytes = 0;
  while (Val >= 128) {
    Val >>= 7;
    ++NumBytes;
  }
  return NumBytes+1;
}

/// EmitVBRValue - Emit the specified value as a VBR, returning the number of
/// bytes emitted.
static uint64_t EmitVBRValue(uint64_t Val, raw_ostream &OS) {
  if (Val <= 127) {
    OS << Val << ", ";
    return 1;
  }

  uint64_t InVal = Val;
  unsigned NumBytes = 0;
  while (Val >= 128) {
    OS << (Val&127) << "|128,";
    Val >>= 7;
    ++NumBytes;
  }
  OS << Val;
  if (!OmitComments)
    OS << "/*" << InVal << "*/";
  OS << ", ";
  return NumBytes+1;
}

/// EmitMatcherOpcodes - Emit bytes for the specified matcher and return
/// the number of bytes emitted.
unsigned MatcherTableEmitter::
EmitMatcher(const Matcher *N, unsigned Indent, unsigned CurrentIdx,
            formatted_raw_ostream &OS) {
  OS.PadToColumn(Indent*2);

  switch (N->getKind()) {
  case Matcher::Scope: {
    const ScopeMatcher *SM = cast<ScopeMatcher>(N);
    assert(SM->getNext() == 0 && "Shouldn't have next after scope");

    unsigned StartIdx = CurrentIdx;

    // Emit all of the children.
    for (unsigned i = 0, e = SM->getNumChildren(); i != e; ++i) {
      if (i == 0) {
        OS << "OPC_Scope, ";
        ++CurrentIdx;
      } else  {
        if (!OmitComments) {
          OS << "/*" << CurrentIdx << "*/";
          OS.PadToColumn(Indent*2) << "/*Scope*/ ";
        } else
          OS.PadToColumn(Indent*2);
      }

      // We need to encode the child and the offset of the failure code before
      // emitting either of them.  Handle this by buffering the output into a
      // string while we get the size.  Unfortunately, the offset of the
      // children depends on the VBR size of the child, so for large children we
      // have to iterate a bit.
      SmallString<128> TmpBuf;
      unsigned ChildSize = 0;
      unsigned VBRSize = 0;
      do {
        VBRSize = GetVBRSize(ChildSize);

        TmpBuf.clear();
        raw_svector_ostream OS(TmpBuf);
        formatted_raw_ostream FOS(OS);
        ChildSize = EmitMatcherList(SM->getChild(i), Indent+1,
                                    CurrentIdx+VBRSize, FOS);
      } while (GetVBRSize(ChildSize) != VBRSize);

      assert(ChildSize != 0 && "Should not have a zero-sized child!");

      CurrentIdx += EmitVBRValue(ChildSize, OS);
      if (!OmitComments) {
        OS << "/*->" << CurrentIdx+ChildSize << "*/";

        if (i == 0)
          OS.PadToColumn(CommentIndent) << "// " << SM->getNumChildren()
            << " children in Scope";
      }

      OS << '\n' << TmpBuf.str();
      CurrentIdx += ChildSize;
    }

    // Emit a zero as a sentinel indicating end of 'Scope'.
    if (!OmitComments)
      OS << "/*" << CurrentIdx << "*/";
    OS.PadToColumn(Indent*2) << "0, ";
    if (!OmitComments)
      OS << "/*End of Scope*/";
    OS << '\n';
    return CurrentIdx - StartIdx + 1;
  }

  case Matcher::RecordNode:
    OS << "OPC_RecordNode,";
    if (!OmitComments)
      OS.PadToColumn(CommentIndent) << "// #"
        << cast<RecordMatcher>(N)->getResultNo() << " = "
        << cast<RecordMatcher>(N)->getWhatFor();
    OS << '\n';
    return 1;

  case Matcher::RecordChild:
    OS << "OPC_RecordChild" << cast<RecordChildMatcher>(N)->getChildNo()
       << ',';
    if (!OmitComments)
      OS.PadToColumn(CommentIndent) << "// #"
        << cast<RecordChildMatcher>(N)->getResultNo() << " = "
        << cast<RecordChildMatcher>(N)->getWhatFor();
    OS << '\n';
    return 1;

  case Matcher::RecordMemRef:
    OS << "OPC_RecordMemRef,\n";
    return 1;

  case Matcher::CaptureGlueInput:
    OS << "OPC_CaptureGlueInput,\n";
    return 1;

  case Matcher::MoveChild:
    OS << "OPC_MoveChild, " << cast<MoveChildMatcher>(N)->getChildNo() << ",\n";
    return 2;

  case Matcher::MoveParent:
    OS << "OPC_MoveParent,\n";
    return 1;

  case Matcher::CheckSame:
    OS << "OPC_CheckSame, "
       << cast<CheckSameMatcher>(N)->getMatchNumber() << ",\n";
    return 2;

  case Matcher::CheckPatternPredicate: {
    StringRef Pred =cast<CheckPatternPredicateMatcher>(N)->getPredicate();
    OS << "OPC_CheckPatternPredicate, " << getPatternPredicate(Pred) << ',';
    if (!OmitComments)
      OS.PadToColumn(CommentIndent) << "// " << Pred;
    OS << '\n';
    return 2;
  }
  case Matcher::CheckPredicate: {
    TreePredicateFn Pred = cast<CheckPredicateMatcher>(N)->getPredicate();
    OS << "OPC_CheckPredicate, " << getNodePredicate(Pred) << ',';
    if (!OmitComments)
      OS.PadToColumn(CommentIndent) << "// " << Pred.getFnName();
    OS << '\n';
    return 2;
  }

  case Matcher::CheckOpcode:
    OS << "OPC_CheckOpcode, TARGET_VAL("
       << cast<CheckOpcodeMatcher>(N)->getOpcode().getEnumName() << "),\n";
    return 3;

  case Matcher::SwitchOpcode:
  case Matcher::SwitchType: {
    unsigned StartIdx = CurrentIdx;

    unsigned NumCases;
    if (const SwitchOpcodeMatcher *SOM = dyn_cast<SwitchOpcodeMatcher>(N)) {
      OS << "OPC_SwitchOpcode ";
      NumCases = SOM->getNumCases();
    } else {
      OS << "OPC_SwitchType ";
      NumCases = cast<SwitchTypeMatcher>(N)->getNumCases();
    }

    if (!OmitComments)
      OS << "/*" << NumCases << " cases */";
    OS << ", ";
    ++CurrentIdx;

    // For each case we emit the size, then the opcode, then the matcher.
    for (unsigned i = 0, e = NumCases; i != e; ++i) {
      const Matcher *Child;
      unsigned IdxSize;
      if (const SwitchOpcodeMatcher *SOM = dyn_cast<SwitchOpcodeMatcher>(N)) {
        Child = SOM->getCaseMatcher(i);
        IdxSize = 2;  // size of opcode in table is 2 bytes.
      } else {
        Child = cast<SwitchTypeMatcher>(N)->getCaseMatcher(i);
        IdxSize = 1;  // size of type in table is 1 byte.
      }

      // We need to encode the opcode and the offset of the case code before
      // emitting the case code.  Handle this by buffering the output into a
      // string while we get the size.  Unfortunately, the offset of the
      // children depends on the VBR size of the child, so for large children we
      // have to iterate a bit.
      SmallString<128> TmpBuf;
      unsigned ChildSize = 0;
      unsigned VBRSize = 0;
      do {
        VBRSize = GetVBRSize(ChildSize);

        TmpBuf.clear();
        raw_svector_ostream OS(TmpBuf);
        formatted_raw_ostream FOS(OS);
        ChildSize = EmitMatcherList(Child, Indent+1, CurrentIdx+VBRSize+IdxSize,
                                    FOS);
      } while (GetVBRSize(ChildSize) != VBRSize);

      assert(ChildSize != 0 && "Should not have a zero-sized child!");

      if (i != 0) {
        OS.PadToColumn(Indent*2);
        if (!OmitComments)
        OS << (isa<SwitchOpcodeMatcher>(N) ?
                   "/*SwitchOpcode*/ " : "/*SwitchType*/ ");
      }

      // Emit the VBR.
      CurrentIdx += EmitVBRValue(ChildSize, OS);

      OS << ' ';
      if (const SwitchOpcodeMatcher *SOM = dyn_cast<SwitchOpcodeMatcher>(N))
        OS << "TARGET_VAL(" << SOM->getCaseOpcode(i).getEnumName() << "),";
      else
        OS << getEnumName(cast<SwitchTypeMatcher>(N)->getCaseType(i)) << ',';

      CurrentIdx += IdxSize;

      if (!OmitComments)
        OS << "// ->" << CurrentIdx+ChildSize;
      OS << '\n';
      OS << TmpBuf.str();
      CurrentIdx += ChildSize;
    }

    // Emit the final zero to terminate the switch.
    OS.PadToColumn(Indent*2) << "0, ";
    if (!OmitComments)
      OS << (isa<SwitchOpcodeMatcher>(N) ?
             "// EndSwitchOpcode" : "// EndSwitchType");

    OS << '\n';
    ++CurrentIdx;
    return CurrentIdx-StartIdx;
  }

 case Matcher::CheckType:
    assert(cast<CheckTypeMatcher>(N)->getResNo() == 0 &&
           "FIXME: Add support for CheckType of resno != 0");
    OS << "OPC_CheckType, "
       << getEnumName(cast<CheckTypeMatcher>(N)->getType()) << ",\n";
    return 2;

  case Matcher::CheckChildType:
    OS << "OPC_CheckChild"
       << cast<CheckChildTypeMatcher>(N)->getChildNo() << "Type, "
       << getEnumName(cast<CheckChildTypeMatcher>(N)->getType()) << ",\n";
    return 2;

  case Matcher::CheckInteger: {
    OS << "OPC_CheckInteger, ";
    unsigned Bytes=1+EmitVBRValue(cast<CheckIntegerMatcher>(N)->getValue(), OS);
    OS << '\n';
    return Bytes;
  }
  case Matcher::CheckCondCode:
    OS << "OPC_CheckCondCode, ISD::"
       << cast<CheckCondCodeMatcher>(N)->getCondCodeName() << ",\n";
    return 2;

  case Matcher::CheckValueType:
    OS << "OPC_CheckValueType, MVT::"
       << cast<CheckValueTypeMatcher>(N)->getTypeName() << ",\n";
    return 2;

  case Matcher::CheckComplexPat: {
    const CheckComplexPatMatcher *CCPM = cast<CheckComplexPatMatcher>(N);
    const ComplexPattern &Pattern = CCPM->getPattern();
    OS << "OPC_CheckComplexPat, /*CP*/" << getComplexPat(Pattern) << ", /*#*/"
       << CCPM->getMatchNumber() << ',';

    if (!OmitComments) {
      OS.PadToColumn(CommentIndent) << "// " << Pattern.getSelectFunc();
      OS << ":$" << CCPM->getName();
      for (unsigned i = 0, e = Pattern.getNumOperands(); i != e; ++i)
        OS << " #" << CCPM->getFirstResult()+i;

      if (Pattern.hasProperty(SDNPHasChain))
        OS << " + chain result";
    }
    OS << '\n';
    return 3;
  }

  case Matcher::CheckAndImm: {
    OS << "OPC_CheckAndImm, ";
    unsigned Bytes=1+EmitVBRValue(cast<CheckAndImmMatcher>(N)->getValue(), OS);
    OS << '\n';
    return Bytes;
  }

  case Matcher::CheckOrImm: {
    OS << "OPC_CheckOrImm, ";
    unsigned Bytes = 1+EmitVBRValue(cast<CheckOrImmMatcher>(N)->getValue(), OS);
    OS << '\n';
    return Bytes;
  }

  case Matcher::CheckFoldableChainNode:
    OS << "OPC_CheckFoldableChainNode,\n";
    return 1;

  case Matcher::EmitInteger: {
    int64_t Val = cast<EmitIntegerMatcher>(N)->getValue();
    OS << "OPC_EmitInteger, "
       << getEnumName(cast<EmitIntegerMatcher>(N)->getVT()) << ", ";
    unsigned Bytes = 2+EmitVBRValue(Val, OS);
    OS << '\n';
    return Bytes;
  }
  case Matcher::EmitStringInteger: {
    const std::string &Val = cast<EmitStringIntegerMatcher>(N)->getValue();
    // These should always fit into one byte.
    OS << "OPC_EmitInteger, "
      << getEnumName(cast<EmitStringIntegerMatcher>(N)->getVT()) << ", "
      << Val << ",\n";
    return 3;
  }

  case Matcher::EmitRegister: {
    const EmitRegisterMatcher *Matcher = cast<EmitRegisterMatcher>(N);
    const CodeGenRegister *Reg = Matcher->getReg();
    // If the enum value of the register is larger than one byte can handle,
    // use EmitRegister2.
    if (Reg && Reg->EnumValue > 255) {
      OS << "OPC_EmitRegister2, " << getEnumName(Matcher->getVT()) << ", ";
      OS << "TARGET_VAL(" << getQualifiedName(Reg->TheDef) << "),\n";
      return 4;
    } else {
      OS << "OPC_EmitRegister, " << getEnumName(Matcher->getVT()) << ", ";
      if (Reg) {
        OS << getQualifiedName(Reg->TheDef) << ",\n";
      } else {
        OS << "0 ";
        if (!OmitComments)
          OS << "/*zero_reg*/";
        OS << ",\n";
      }
      return 3;
    }
  }

  case Matcher::EmitConvertToTarget:
    OS << "OPC_EmitConvertToTarget, "
       << cast<EmitConvertToTargetMatcher>(N)->getSlot() << ",\n";
    return 2;

  case Matcher::EmitMergeInputChains: {
    const EmitMergeInputChainsMatcher *MN =
      cast<EmitMergeInputChainsMatcher>(N);

    // Handle the specialized forms OPC_EmitMergeInputChains1_0 and 1_1.
    if (MN->getNumNodes() == 1 && MN->getNode(0) < 2) {
      OS << "OPC_EmitMergeInputChains1_" << MN->getNode(0) << ",\n";
      return 1;
    }

    OS << "OPC_EmitMergeInputChains, " << MN->getNumNodes() << ", ";
    for (unsigned i = 0, e = MN->getNumNodes(); i != e; ++i)
      OS << MN->getNode(i) << ", ";
    OS << '\n';
    return 2+MN->getNumNodes();
  }
  case Matcher::EmitCopyToReg:
    OS << "OPC_EmitCopyToReg, "
       << cast<EmitCopyToRegMatcher>(N)->getSrcSlot() << ", "
       << getQualifiedName(cast<EmitCopyToRegMatcher>(N)->getDestPhysReg())
       << ",\n";
    return 3;
  case Matcher::EmitNodeXForm: {
    const EmitNodeXFormMatcher *XF = cast<EmitNodeXFormMatcher>(N);
    OS << "OPC_EmitNodeXForm, " << getNodeXFormID(XF->getNodeXForm()) << ", "
       << XF->getSlot() << ',';
    if (!OmitComments)
      OS.PadToColumn(CommentIndent) << "// "<<XF->getNodeXForm()->getName();
    OS <<'\n';
    return 3;
  }

  case Matcher::EmitNode:
  case Matcher::MorphNodeTo: {
    const EmitNodeMatcherCommon *EN = cast<EmitNodeMatcherCommon>(N);
    OS << (isa<EmitNodeMatcher>(EN) ? "OPC_EmitNode" : "OPC_MorphNodeTo");
    OS << ", TARGET_VAL(" << EN->getOpcodeName() << "), 0";

    if (EN->hasChain())   OS << "|OPFL_Chain";
    if (EN->hasInFlag())  OS << "|OPFL_GlueInput";
    if (EN->hasOutFlag()) OS << "|OPFL_GlueOutput";
    if (EN->hasMemRefs()) OS << "|OPFL_MemRefs";
    if (EN->getNumFixedArityOperands() != -1)
      OS << "|OPFL_Variadic" << EN->getNumFixedArityOperands();
    OS << ",\n";

    OS.PadToColumn(Indent*2+4) << EN->getNumVTs();
    if (!OmitComments)
      OS << "/*#VTs*/";
    OS << ", ";
    for (unsigned i = 0, e = EN->getNumVTs(); i != e; ++i)
      OS << getEnumName(EN->getVT(i)) << ", ";

    OS << EN->getNumOperands();
    if (!OmitComments)
      OS << "/*#Ops*/";
    OS << ", ";
    unsigned NumOperandBytes = 0;
    for (unsigned i = 0, e = EN->getNumOperands(); i != e; ++i)
      NumOperandBytes += EmitVBRValue(EN->getOperand(i), OS);

    if (!OmitComments) {
      // Print the result #'s for EmitNode.
      if (const EmitNodeMatcher *E = dyn_cast<EmitNodeMatcher>(EN)) {
        if (unsigned NumResults = EN->getNumVTs()) {
          OS.PadToColumn(CommentIndent) << "// Results =";
          unsigned First = E->getFirstResultSlot();
          for (unsigned i = 0; i != NumResults; ++i)
            OS << " #" << First+i;
        }
      }
      OS << '\n';

      if (const MorphNodeToMatcher *SNT = dyn_cast<MorphNodeToMatcher>(N)) {
        OS.PadToColumn(Indent*2) << "// Src: "
          << *SNT->getPattern().getSrcPattern() << " - Complexity = "
          << SNT->getPattern().getPatternComplexity(CGP) << '\n';
        OS.PadToColumn(Indent*2) << "// Dst: "
          << *SNT->getPattern().getDstPattern() << '\n';
      }
    } else
      OS << '\n';

    return 6+EN->getNumVTs()+NumOperandBytes;
  }
  case Matcher::MarkGlueResults: {
    const MarkGlueResultsMatcher *CFR = cast<MarkGlueResultsMatcher>(N);
    OS << "OPC_MarkGlueResults, " << CFR->getNumNodes() << ", ";
    unsigned NumOperandBytes = 0;
    for (unsigned i = 0, e = CFR->getNumNodes(); i != e; ++i)
      NumOperandBytes += EmitVBRValue(CFR->getNode(i), OS);
    OS << '\n';
    return 2+NumOperandBytes;
  }
  case Matcher::CompleteMatch: {
    const CompleteMatchMatcher *CM = cast<CompleteMatchMatcher>(N);
    OS << "OPC_CompleteMatch, " << CM->getNumResults() << ", ";
    unsigned NumResultBytes = 0;
    for (unsigned i = 0, e = CM->getNumResults(); i != e; ++i)
      NumResultBytes += EmitVBRValue(CM->getResult(i), OS);
    OS << '\n';
    if (!OmitComments) {
      OS.PadToColumn(Indent*2) << "// Src: "
        << *CM->getPattern().getSrcPattern() << " - Complexity = "
        << CM->getPattern().getPatternComplexity(CGP) << '\n';
      OS.PadToColumn(Indent*2) << "// Dst: "
        << *CM->getPattern().getDstPattern();
    }
    OS << '\n';
    return 2 + NumResultBytes;
  }
  }
  llvm_unreachable("Unreachable");
}

/// EmitMatcherList - Emit the bytes for the specified matcher subtree.
unsigned MatcherTableEmitter::
EmitMatcherList(const Matcher *N, unsigned Indent, unsigned CurrentIdx,
                formatted_raw_ostream &OS) {
  unsigned Size = 0;
  while (N) {
    if (!OmitComments)
      OS << "/*" << CurrentIdx << "*/";
    unsigned MatcherSize = EmitMatcher(N, Indent, CurrentIdx, OS);
    Size += MatcherSize;
    CurrentIdx += MatcherSize;

    // If there are other nodes in this list, iterate to them, otherwise we're
    // done.
    N = N->getNext();
  }
  return Size;
}

void MatcherTableEmitter::EmitPredicateFunctions(formatted_raw_ostream &OS) {
  // Emit pattern predicates.
  if (!PatternPredicates.empty()) {
    OS << "virtual bool CheckPatternPredicate(unsigned PredNo) const {\n";
    OS << "  switch (PredNo) {\n";
    OS << "  default: llvm_unreachable(\"Invalid predicate in table?\");\n";
    for (unsigned i = 0, e = PatternPredicates.size(); i != e; ++i)
      OS << "  case " << i << ": return "  << PatternPredicates[i] << ";\n";
    OS << "  }\n";
    OS << "}\n\n";
  }

  // Emit Node predicates.
  // FIXME: Annoyingly, these are stored by name, which we never even emit. Yay?
  StringMap<TreePattern*> PFsByName;

  for (CodeGenDAGPatterns::pf_iterator I = CGP.pf_begin(), E = CGP.pf_end();
       I != E; ++I)
    PFsByName[I->first->getName()] = I->second;

  if (!NodePredicates.empty()) {
    OS << "virtual bool CheckNodePredicate(SDNode *Node,\n";
    OS << "                                unsigned PredNo) const {\n";
    OS << "  switch (PredNo) {\n";
    OS << "  default: llvm_unreachable(\"Invalid predicate in table?\");\n";
    for (unsigned i = 0, e = NodePredicates.size(); i != e; ++i) {
      // Emit the predicate code corresponding to this pattern.
      TreePredicateFn PredFn = NodePredicates[i];
      
      assert(!PredFn.isAlwaysTrue() && "No code in this predicate");
      OS << "  case " << i << ": { // " << NodePredicates[i].getFnName() <<'\n';
      
      OS << PredFn.getCodeToRunOnSDNode() << "\n  }\n";
    }
    OS << "  }\n";
    OS << "}\n\n";
  }

  // Emit CompletePattern matchers.
  // FIXME: This should be const.
  if (!ComplexPatterns.empty()) {
    OS << "virtual bool CheckComplexPattern(SDNode *Root, SDNode *Parent,\n";
    OS << "                                 SDValue N, unsigned PatternNo,\n";
    OS << "         SmallVectorImpl<std::pair<SDValue, SDNode*> > &Result) {\n";
    OS << "  unsigned NextRes = Result.size();\n";
    OS << "  switch (PatternNo) {\n";
    OS << "  default: llvm_unreachable(\"Invalid pattern # in table?\");\n";
    for (unsigned i = 0, e = ComplexPatterns.size(); i != e; ++i) {
      const ComplexPattern &P = *ComplexPatterns[i];
      unsigned NumOps = P.getNumOperands();

      if (P.hasProperty(SDNPHasChain))
        ++NumOps;  // Get the chained node too.

      OS << "  case " << i << ":\n";
      OS << "    Result.resize(NextRes+" << NumOps << ");\n";
      OS << "    return "  << P.getSelectFunc();

      OS << "(";
      // If the complex pattern wants the root of the match, pass it in as the
      // first argument.
      if (P.hasProperty(SDNPWantRoot))
        OS << "Root, ";

      // If the complex pattern wants the parent of the operand being matched,
      // pass it in as the next argument.
      if (P.hasProperty(SDNPWantParent))
        OS << "Parent, ";

      OS << "N";
      for (unsigned i = 0; i != NumOps; ++i)
        OS << ", Result[NextRes+" << i << "].first";
      OS << ");\n";
    }
    OS << "  }\n";
    OS << "}\n\n";
  }


  // Emit SDNodeXForm handlers.
  // FIXME: This should be const.
  if (!NodeXForms.empty()) {
    OS << "virtual SDValue RunSDNodeXForm(SDValue V, unsigned XFormNo) {\n";
    OS << "  switch (XFormNo) {\n";
    OS << "  default: llvm_unreachable(\"Invalid xform # in table?\");\n";

    // FIXME: The node xform could take SDValue's instead of SDNode*'s.
    for (unsigned i = 0, e = NodeXForms.size(); i != e; ++i) {
      const CodeGenDAGPatterns::NodeXForm &Entry =
        CGP.getSDNodeTransform(NodeXForms[i]);

      Record *SDNode = Entry.first;
      const std::string &Code = Entry.second;

      OS << "  case " << i << ": {  ";
      if (!OmitComments)
        OS << "// " << NodeXForms[i]->getName();
      OS << '\n';

      std::string ClassName = CGP.getSDNodeInfo(SDNode).getSDClassName();
      if (ClassName == "SDNode")
        OS << "    SDNode *N = V.getNode();\n";
      else
        OS << "    " << ClassName << " *N = cast<" << ClassName
           << ">(V.getNode());\n";
      OS << Code << "\n  }\n";
    }
    OS << "  }\n";
    OS << "}\n\n";
  }
}

static void BuildHistogram(const Matcher *M, std::vector<unsigned> &OpcodeFreq){
  for (; M != 0; M = M->getNext()) {
    // Count this node.
    if (unsigned(M->getKind()) >= OpcodeFreq.size())
      OpcodeFreq.resize(M->getKind()+1);
    OpcodeFreq[M->getKind()]++;

    // Handle recursive nodes.
    if (const ScopeMatcher *SM = dyn_cast<ScopeMatcher>(M)) {
      for (unsigned i = 0, e = SM->getNumChildren(); i != e; ++i)
        BuildHistogram(SM->getChild(i), OpcodeFreq);
    } else if (const SwitchOpcodeMatcher *SOM =
                 dyn_cast<SwitchOpcodeMatcher>(M)) {
      for (unsigned i = 0, e = SOM->getNumCases(); i != e; ++i)
        BuildHistogram(SOM->getCaseMatcher(i), OpcodeFreq);
    } else if (const SwitchTypeMatcher *STM = dyn_cast<SwitchTypeMatcher>(M)) {
      for (unsigned i = 0, e = STM->getNumCases(); i != e; ++i)
        BuildHistogram(STM->getCaseMatcher(i), OpcodeFreq);
    }
  }
}

void MatcherTableEmitter::EmitHistogram(const Matcher *M,
                                        formatted_raw_ostream &OS) {
  if (OmitComments)
    return;

  std::vector<unsigned> OpcodeFreq;
  BuildHistogram(M, OpcodeFreq);

  OS << "  // Opcode Histogram:\n";
  for (unsigned i = 0, e = OpcodeFreq.size(); i != e; ++i) {
    OS << "  // #";
    switch ((Matcher::KindTy)i) {
    case Matcher::Scope: OS << "OPC_Scope"; break;
    case Matcher::RecordNode: OS << "OPC_RecordNode"; break;
    case Matcher::RecordChild: OS << "OPC_RecordChild"; break;
    case Matcher::RecordMemRef: OS << "OPC_RecordMemRef"; break;
    case Matcher::CaptureGlueInput: OS << "OPC_CaptureGlueInput"; break;
    case Matcher::MoveChild: OS << "OPC_MoveChild"; break;
    case Matcher::MoveParent: OS << "OPC_MoveParent"; break;
    case Matcher::CheckSame: OS << "OPC_CheckSame"; break;
    case Matcher::CheckPatternPredicate:
      OS << "OPC_CheckPatternPredicate"; break;
    case Matcher::CheckPredicate: OS << "OPC_CheckPredicate"; break;
    case Matcher::CheckOpcode: OS << "OPC_CheckOpcode"; break;
    case Matcher::SwitchOpcode: OS << "OPC_SwitchOpcode"; break;
    case Matcher::CheckType: OS << "OPC_CheckType"; break;
    case Matcher::SwitchType: OS << "OPC_SwitchType"; break;
    case Matcher::CheckChildType: OS << "OPC_CheckChildType"; break;
    case Matcher::CheckInteger: OS << "OPC_CheckInteger"; break;
    case Matcher::CheckCondCode: OS << "OPC_CheckCondCode"; break;
    case Matcher::CheckValueType: OS << "OPC_CheckValueType"; break;
    case Matcher::CheckComplexPat: OS << "OPC_CheckComplexPat"; break;
    case Matcher::CheckAndImm: OS << "OPC_CheckAndImm"; break;
    case Matcher::CheckOrImm: OS << "OPC_CheckOrImm"; break;
    case Matcher::CheckFoldableChainNode:
      OS << "OPC_CheckFoldableChainNode"; break;
    case Matcher::EmitInteger: OS << "OPC_EmitInteger"; break;
    case Matcher::EmitStringInteger: OS << "OPC_EmitStringInteger"; break;
    case Matcher::EmitRegister: OS << "OPC_EmitRegister"; break;
    case Matcher::EmitConvertToTarget: OS << "OPC_EmitConvertToTarget"; break;
    case Matcher::EmitMergeInputChains: OS << "OPC_EmitMergeInputChains"; break;
    case Matcher::EmitCopyToReg: OS << "OPC_EmitCopyToReg"; break;
    case Matcher::EmitNode: OS << "OPC_EmitNode"; break;
    case Matcher::MorphNodeTo: OS << "OPC_MorphNodeTo"; break;
    case Matcher::EmitNodeXForm: OS << "OPC_EmitNodeXForm"; break;
    case Matcher::MarkGlueResults: OS << "OPC_MarkGlueResults"; break;
    case Matcher::CompleteMatch: OS << "OPC_CompleteMatch"; break;
    }

    OS.PadToColumn(40) << " = " << OpcodeFreq[i] << '\n';
  }
  OS << '\n';
}


void llvm::EmitMatcherTable(const Matcher *TheMatcher,
                            const CodeGenDAGPatterns &CGP,
                            raw_ostream &O) {
  formatted_raw_ostream OS(O);

  OS << "// The main instruction selector code.\n";
  OS << "SDNode *SelectCode(SDNode *N) {\n";

  MatcherTableEmitter MatcherEmitter(CGP);

  OS << "  // Some target values are emitted as 2 bytes, TARGET_VAL handles\n";
  OS << "  // this.\n";
  OS << "  #define TARGET_VAL(X) X & 255, unsigned(X) >> 8\n";
  OS << "  static const unsigned char MatcherTable[] = {\n";
  unsigned TotalSize = MatcherEmitter.EmitMatcherList(TheMatcher, 5, 0, OS);
  OS << "    0\n  }; // Total Array size is " << (TotalSize+1) << " bytes\n\n";

  MatcherEmitter.EmitHistogram(TheMatcher, OS);

  OS << "  #undef TARGET_VAL\n";
  OS << "  return SelectCodeCommon(N, MatcherTable,sizeof(MatcherTable));\n}\n";
  OS << '\n';

  // Next up, emit the function for node and pattern predicates:
  MatcherEmitter.EmitPredicateFunctions(OS);
}
