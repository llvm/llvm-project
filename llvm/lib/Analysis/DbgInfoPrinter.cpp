//===- DbgInfoPrinter.cpp - Print debug info in a human readable form ------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that prints instructions, and associated debug
// info:
// 
//   - source/line/col information
//   - original variable name
//   - original type name
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo.h"
#include "llvm/Function.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Metadata.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<bool>
PrintDirectory("print-fullpath",
               cl::desc("Print fullpath when printing debug info"),
               cl::Hidden);

namespace {
  class PrintDbgInfo : public FunctionPass {
    raw_ostream &Out;
    void printVariableDeclaration(const Value *V);
  public:
    static char ID; // Pass identification
    PrintDbgInfo() : FunctionPass(ID), Out(errs()) {
      initializePrintDbgInfoPass(*PassRegistry::getPassRegistry());
    }

    virtual bool runOnFunction(Function &F);
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };
  char PrintDbgInfo::ID = 0;
}

INITIALIZE_PASS(PrintDbgInfo, "print-dbginfo",
                "Print debug info in human readable form", false, false)

FunctionPass *llvm::createDbgInfoPrinterPass() { return new PrintDbgInfo(); }

/// Find the debug info descriptor corresponding to this global variable.
static Value *findDbgGlobalDeclare(GlobalVariable *V) {
  const Module *M = V->getParent();
  NamedMDNode *NMD = M->getNamedMetadata("llvm.dbg.gv");
  if (!NMD)
    return 0;

  for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i) {
    DIDescriptor DIG(cast<MDNode>(NMD->getOperand(i)));
    if (!DIG.isGlobalVariable())
      continue;
    if (DIGlobalVariable(DIG).getGlobal() == V)
      return DIG;
  }
  return 0;
}

/// Find the debug info descriptor corresponding to this function.
static Value *findDbgSubprogramDeclare(Function *V) {
  const Module *M = V->getParent();
  NamedMDNode *NMD = M->getNamedMetadata("llvm.dbg.sp");
  if (!NMD)
    return 0;

  for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i) {
    DIDescriptor DIG(cast<MDNode>(NMD->getOperand(i)));
    if (!DIG.isSubprogram())
      continue;
    if (DISubprogram(DIG).getFunction() == V)
      return DIG;
  }
  return 0;
}

/// Finds the llvm.dbg.declare intrinsic corresponding to this value if any.
/// It looks through pointer casts too.
static const DbgDeclareInst *findDbgDeclare(const Value *V) {
  V = V->stripPointerCasts();

  if (!isa<Instruction>(V) && !isa<Argument>(V))
    return 0;

  const Function *F = NULL;
  if (const Instruction *I = dyn_cast<Instruction>(V))
    F = I->getParent()->getParent();
  else if (const Argument *A = dyn_cast<Argument>(V))
    F = A->getParent();

  for (Function::const_iterator FI = F->begin(), FE = F->end(); FI != FE; ++FI)
    for (BasicBlock::const_iterator BI = (*FI).begin(), BE = (*FI).end();
         BI != BE; ++BI)
      if (const DbgDeclareInst *DDI = dyn_cast<DbgDeclareInst>(BI))
        if (DDI->getAddress() == V)
          return DDI;

  return 0;
}

static bool getLocationInfo(const Value *V, std::string &DisplayName,
                            std::string &Type, unsigned &LineNo,
                            std::string &File, std::string &Dir) {
  DICompileUnit Unit;
  DIType TypeD;

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(const_cast<Value*>(V))) {
    Value *DIGV = findDbgGlobalDeclare(GV);
    if (!DIGV) return false;
    DIGlobalVariable Var(cast<MDNode>(DIGV));

    StringRef D = Var.getDisplayName();
    if (!D.empty())
      DisplayName = D;
    LineNo = Var.getLineNumber();
    Unit = Var.getCompileUnit();
    TypeD = Var.getType();
  } else if (Function *F = dyn_cast<Function>(const_cast<Value*>(V))){
    Value *DIF = findDbgSubprogramDeclare(F);
    if (!DIF) return false;
    DISubprogram Var(cast<MDNode>(DIF));

    StringRef D = Var.getDisplayName();
    if (!D.empty())
      DisplayName = D;
    LineNo = Var.getLineNumber();
    Unit = Var.getCompileUnit();
    TypeD = Var.getType();
  } else {
    const DbgDeclareInst *DDI = findDbgDeclare(V);
    if (!DDI) return false;
    DIVariable Var(cast<MDNode>(DDI->getVariable()));

    StringRef D = Var.getName();
    if (!D.empty())
      DisplayName = D;
    LineNo = Var.getLineNumber();
    Unit = Var.getCompileUnit();
    TypeD = Var.getType();
  }

  StringRef T = TypeD.getName();
  if (!T.empty())
    Type = T;
  StringRef F = Unit.getFilename();
  if (!F.empty())
    File = F;
  StringRef D = Unit.getDirectory();
  if (!D.empty())
    Dir = D;
  return true;
}

void PrintDbgInfo::printVariableDeclaration(const Value *V) {
  std::string DisplayName, File, Directory, Type;
  unsigned LineNo = 0;

  if (!getLocationInfo(V, DisplayName, Type, LineNo, File, Directory))
    return;

  Out << "; ";
  WriteAsOperand(Out, V, false, 0);
  if (isa<Function>(V)) 
    Out << " is function " << DisplayName
        << " of type " << Type << " declared at ";
  else
    Out << " is variable " << DisplayName
        << " of type " << Type << " declared at ";

  if (PrintDirectory)
    Out << Directory << "/";

  Out << File << ":" << LineNo << "\n";
}

bool PrintDbgInfo::runOnFunction(Function &F) {
  if (F.isDeclaration())
    return false;

  Out << "function " << F.getName() << "\n\n";

  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    BasicBlock *BB = I;

    if (I != F.begin() && (pred_begin(BB) == pred_end(BB)))
      // Skip dead blocks.
      continue;

    Out << BB->getName();
    Out << ":";

    Out << "\n";

    for (BasicBlock::const_iterator i = BB->begin(), e = BB->end();
         i != e; ++i) {

        printVariableDeclaration(i);

        if (const User *U = dyn_cast<User>(i)) {
          for(unsigned i=0;i<U->getNumOperands();i++)
            printVariableDeclaration(U->getOperand(i));
        }
    }
  }
  return false;
}
