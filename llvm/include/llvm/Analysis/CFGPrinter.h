//===-- CFGPrinter.h - CFG printer external interface -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines external functions that can be called to explicitly
// instantiate the CFG printer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CFGPRINTER_H
#define LLVM_ANALYSIS_CFGPRINTER_H

#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/GraphWriter.h"

namespace llvm {
template<>
struct DOTGraphTraits<const Function*> : public DefaultDOTGraphTraits {

  DOTGraphTraits (bool isSimple=false) : DefaultDOTGraphTraits(isSimple) {}

  static std::string getGraphName(const Function *F) {
    return "CFG for '" + F->getName().str() + "' function";
  }

  static std::string getSimpleNodeLabel(const BasicBlock *Node,
                                        const Function *) {
    if (!Node->getName().empty())
      return Node->getName().str();

    std::string Str;
    raw_string_ostream OS(Str);

    WriteAsOperand(OS, Node, false);
    return OS.str();
  }

  static std::string getCompleteNodeLabel(const BasicBlock *Node, 
                                          const Function *) {
    std::string Str;
    raw_string_ostream OS(Str);

    if (Node->getName().empty()) {
      WriteAsOperand(OS, Node, false);
      OS << ":";
    }

    OS << *Node;
    std::string OutStr = OS.str();
    if (OutStr[0] == '\n') OutStr.erase(OutStr.begin());

    // Process string output to make it nicer...
    for (unsigned i = 0; i != OutStr.length(); ++i)
      if (OutStr[i] == '\n') {                            // Left justify
        OutStr[i] = '\\';
        OutStr.insert(OutStr.begin()+i+1, 'l');
      } else if (OutStr[i] == ';') {                      // Delete comments!
        unsigned Idx = OutStr.find('\n', i+1);            // Find end of line
        OutStr.erase(OutStr.begin()+i, OutStr.begin()+Idx);
        --i;
      }

    return OutStr;
  }

  std::string getNodeLabel(const BasicBlock *Node,
                           const Function *Graph) {
    if (isSimple())
      return getSimpleNodeLabel(Node, Graph);
    else
      return getCompleteNodeLabel(Node, Graph);
  }

  static std::string getEdgeSourceLabel(const BasicBlock *Node,
                                        succ_const_iterator I) {
    // Label source of conditional branches with "T" or "F"
    if (const BranchInst *BI = dyn_cast<BranchInst>(Node->getTerminator()))
      if (BI->isConditional())
        return (I == succ_begin(Node)) ? "T" : "F";
    
    // Label source of switch edges with the associated value.
    if (const SwitchInst *SI = dyn_cast<SwitchInst>(Node->getTerminator())) {
      unsigned SuccNo = I.getSuccessorIndex();

      if (SuccNo == 0) return "def";
      
      std::string Str;
      raw_string_ostream OS(Str);
      SwitchInst::ConstCaseIt Case =
          SwitchInst::ConstCaseIt::fromSuccessorIndex(SI, SuccNo); 
      OS << Case.getCaseValue()->getValue();
      return OS.str();
    }    
    return "";
  }
};
} // End llvm namespace

namespace llvm {
  class FunctionPass;
  FunctionPass *createCFGPrinterPass ();
  FunctionPass *createCFGOnlyPrinterPass ();
} // End llvm namespace

#endif
