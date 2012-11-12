//===- MemDepPrinter.cpp - Printer for MemoryDependenceAnalysis -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/LLVMContext.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SetVector.h"
using namespace llvm;

namespace {
  struct MemDepPrinter : public FunctionPass {
    const Function *F;

    enum DepType {
      Clobber = 0,
      Def,
      NonFuncLocal,
      Unknown
    };

    static const char *const DepTypeStr[];

    typedef PointerIntPair<const Instruction *, 2, DepType> InstTypePair;
    typedef std::pair<InstTypePair, const BasicBlock *> Dep;
    typedef SmallSetVector<Dep, 4> DepSet;
    typedef DenseMap<const Instruction *, DepSet> DepSetMap;
    DepSetMap Deps;

    static char ID; // Pass identifcation, replacement for typeid
    MemDepPrinter() : FunctionPass(ID) {
      initializeMemDepPrinterPass(*PassRegistry::getPassRegistry());
    }

    virtual bool runOnFunction(Function &F);

    void print(raw_ostream &OS, const Module * = 0) const;

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredTransitive<AliasAnalysis>();
      AU.addRequiredTransitive<MemoryDependenceAnalysis>();
      AU.setPreservesAll();
    }

    virtual void releaseMemory() {
      Deps.clear();
      F = 0;
    }

  private:
    static InstTypePair getInstTypePair(MemDepResult dep) {
      if (dep.isClobber())
        return InstTypePair(dep.getInst(), Clobber);
      if (dep.isDef())
        return InstTypePair(dep.getInst(), Def);
      if (dep.isNonFuncLocal())
        return InstTypePair(dep.getInst(), NonFuncLocal);
      assert(dep.isUnknown() && "unexptected dependence type");
      return InstTypePair(dep.getInst(), Unknown);
    }
    static InstTypePair getInstTypePair(const Instruction* inst, DepType type) {
      return InstTypePair(inst, type);
    }
  };
}

char MemDepPrinter::ID = 0;
INITIALIZE_PASS_BEGIN(MemDepPrinter, "print-memdeps",
                      "Print MemDeps of function", false, true)
INITIALIZE_PASS_DEPENDENCY(MemoryDependenceAnalysis)
INITIALIZE_PASS_END(MemDepPrinter, "print-memdeps",
                      "Print MemDeps of function", false, true)

FunctionPass *llvm::createMemDepPrinter() {
  return new MemDepPrinter();
}

const char *const MemDepPrinter::DepTypeStr[]
  = {"Clobber", "Def", "NonFuncLocal", "Unknown"};

bool MemDepPrinter::runOnFunction(Function &F) {
  this->F = &F;
  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();
  MemoryDependenceAnalysis &MDA = getAnalysis<MemoryDependenceAnalysis>();

  // All this code uses non-const interfaces because MemDep is not
  // const-friendly, though nothing is actually modified.
  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
    Instruction *Inst = &*I;

    if (!Inst->mayReadFromMemory() && !Inst->mayWriteToMemory())
      continue;

    MemDepResult Res = MDA.getDependency(Inst);
    if (!Res.isNonLocal()) {
      Deps[Inst].insert(std::make_pair(getInstTypePair(Res),
                                       static_cast<BasicBlock *>(0)));
    } else if (CallSite CS = cast<Value>(Inst)) {
      const MemoryDependenceAnalysis::NonLocalDepInfo &NLDI =
        MDA.getNonLocalCallDependency(CS);

      DepSet &InstDeps = Deps[Inst];
      for (MemoryDependenceAnalysis::NonLocalDepInfo::const_iterator
           I = NLDI.begin(), E = NLDI.end(); I != E; ++I) {
        const MemDepResult &Res = I->getResult();
        InstDeps.insert(std::make_pair(getInstTypePair(Res), I->getBB()));
      }
    } else {
      SmallVector<NonLocalDepResult, 4> NLDI;
      if (LoadInst *LI = dyn_cast<LoadInst>(Inst)) {
        if (!LI->isUnordered()) {
          // FIXME: Handle atomic/volatile loads.
          Deps[Inst].insert(std::make_pair(getInstTypePair(0, Unknown),
                                           static_cast<BasicBlock *>(0)));
          continue;
        }
        AliasAnalysis::Location Loc = AA.getLocation(LI);
        MDA.getNonLocalPointerDependency(Loc, true, LI->getParent(), NLDI);
      } else if (StoreInst *SI = dyn_cast<StoreInst>(Inst)) {
        if (!SI->isUnordered()) {
          // FIXME: Handle atomic/volatile stores.
          Deps[Inst].insert(std::make_pair(getInstTypePair(0, Unknown),
                                           static_cast<BasicBlock *>(0)));
          continue;
        }
        AliasAnalysis::Location Loc = AA.getLocation(SI);
        MDA.getNonLocalPointerDependency(Loc, false, SI->getParent(), NLDI);
      } else if (VAArgInst *VI = dyn_cast<VAArgInst>(Inst)) {
        AliasAnalysis::Location Loc = AA.getLocation(VI);
        MDA.getNonLocalPointerDependency(Loc, false, VI->getParent(), NLDI);
      } else {
        llvm_unreachable("Unknown memory instruction!");
      }

      DepSet &InstDeps = Deps[Inst];
      for (SmallVectorImpl<NonLocalDepResult>::const_iterator
           I = NLDI.begin(), E = NLDI.end(); I != E; ++I) {
        const MemDepResult &Res = I->getResult();
        InstDeps.insert(std::make_pair(getInstTypePair(Res), I->getBB()));
      }
    }
  }

  return false;
}

void MemDepPrinter::print(raw_ostream &OS, const Module *M) const {
  for (const_inst_iterator I = inst_begin(*F), E = inst_end(*F); I != E; ++I) {
    const Instruction *Inst = &*I;

    DepSetMap::const_iterator DI = Deps.find(Inst);
    if (DI == Deps.end())
      continue;

    const DepSet &InstDeps = DI->second;

    for (DepSet::const_iterator I = InstDeps.begin(), E = InstDeps.end();
         I != E; ++I) {
      const Instruction *DepInst = I->first.getPointer();
      DepType type = I->first.getInt();
      const BasicBlock *DepBB = I->second;

      OS << "    ";
      OS << DepTypeStr[type];
      if (DepBB) {
        OS << " in block ";
        WriteAsOperand(OS, DepBB, /*PrintType=*/false, M);
      }
      if (DepInst) {
        OS << " from: ";
        DepInst->print(OS);
      }
      OS << "\n";
    }

    Inst->print(OS);
    OS << "\n\n";
  }
}
