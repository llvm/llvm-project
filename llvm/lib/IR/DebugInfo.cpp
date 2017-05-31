//===--- DebugInfo.cpp - Debug Information Helper Classes -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the helper classes used to build and interpret debug
// information in LLVM IR form.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugInfo.h"
#include "LLVMContextImpl.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GVMaterializer.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;
using namespace llvm::dwarf;

DISubprogram *llvm::getDISubprogram(const MDNode *Scope) {
  if (auto *LocalScope = dyn_cast_or_null<DILocalScope>(Scope))
    return LocalScope->getSubprogram();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// DebugInfoFinder implementations.
//===----------------------------------------------------------------------===//

void DebugInfoFinder::reset() {
  CUs.clear();
  SPs.clear();
  GVs.clear();
  TYs.clear();
  Scopes.clear();
  NodesSeen.clear();
}

void DebugInfoFinder::processModule(const Module &M) {
  for (auto *CU : M.debug_compile_units()) {
    addCompileUnit(CU);
    for (auto *DIG : CU->getGlobalVariables()) {
      if (addGlobalVariable(DIG)) {
        processScope(DIG->getScope());
        processType(DIG->getType().resolve());
      }
    }
    for (auto *ET : CU->getEnumTypes())
      processType(ET);
    for (auto *RT : CU->getRetainedTypes())
      if (auto *T = dyn_cast<DIType>(RT))
        processType(T);
      else
        processSubprogram(cast<DISubprogram>(RT));
    for (auto *Import : CU->getImportedEntities()) {
      auto *Entity = Import->getEntity().resolve();
      if (auto *T = dyn_cast<DIType>(Entity))
        processType(T);
      else if (auto *SP = dyn_cast<DISubprogram>(Entity))
        processSubprogram(SP);
      else if (auto *NS = dyn_cast<DINamespace>(Entity))
        processScope(NS->getScope());
      else if (auto *M = dyn_cast<DIModule>(Entity))
        processScope(M->getScope());
    }
  }
  for (auto &F : M.functions())
    if (auto *SP = cast_or_null<DISubprogram>(F.getSubprogram()))
      processSubprogram(SP);
}

void DebugInfoFinder::processLocation(const Module &M, const DILocation *Loc) {
  if (!Loc)
    return;
  processScope(Loc->getScope());
  processLocation(M, Loc->getInlinedAt());
}

void DebugInfoFinder::processType(DIType *DT) {
  if (!addType(DT))
    return;
  processScope(DT->getScope().resolve());
  if (auto *ST = dyn_cast<DISubroutineType>(DT)) {
    for (DITypeRef Ref : ST->getTypeArray())
      processType(Ref.resolve());
    return;
  }
  if (auto *DCT = dyn_cast<DICompositeType>(DT)) {
    processType(DCT->getBaseType().resolve());
    for (Metadata *D : DCT->getElements()) {
      if (auto *T = dyn_cast<DIType>(D))
        processType(T);
      else if (auto *SP = dyn_cast<DISubprogram>(D))
        processSubprogram(SP);
    }
    return;
  }
  if (auto *DDT = dyn_cast<DIDerivedType>(DT)) {
    processType(DDT->getBaseType().resolve());
  }
}

void DebugInfoFinder::processScope(DIScope *Scope) {
  if (!Scope)
    return;
  if (auto *Ty = dyn_cast<DIType>(Scope)) {
    processType(Ty);
    return;
  }
  if (auto *CU = dyn_cast<DICompileUnit>(Scope)) {
    addCompileUnit(CU);
    return;
  }
  if (auto *SP = dyn_cast<DISubprogram>(Scope)) {
    processSubprogram(SP);
    return;
  }
  if (!addScope(Scope))
    return;
  if (auto *LB = dyn_cast<DILexicalBlockBase>(Scope)) {
    processScope(LB->getScope());
  } else if (auto *NS = dyn_cast<DINamespace>(Scope)) {
    processScope(NS->getScope());
  } else if (auto *M = dyn_cast<DIModule>(Scope)) {
    processScope(M->getScope());
  }
}

void DebugInfoFinder::processSubprogram(DISubprogram *SP) {
  if (!addSubprogram(SP))
    return;
  processScope(SP->getScope().resolve());
  processType(SP->getType());
  for (auto *Element : SP->getTemplateParams()) {
    if (auto *TType = dyn_cast<DITemplateTypeParameter>(Element)) {
      processType(TType->getType().resolve());
    } else if (auto *TVal = dyn_cast<DITemplateValueParameter>(Element)) {
      processType(TVal->getType().resolve());
    }
  }
}

void DebugInfoFinder::processDeclare(const Module &M,
                                     const DbgDeclareInst *DDI) {
  auto *N = dyn_cast<MDNode>(DDI->getVariable());
  if (!N)
    return;

  auto *DV = dyn_cast<DILocalVariable>(N);
  if (!DV)
    return;

  if (!NodesSeen.insert(DV).second)
    return;
  processScope(DV->getScope());
  processType(DV->getType().resolve());
}

void DebugInfoFinder::processValue(const Module &M, const DbgValueInst *DVI) {
  auto *N = dyn_cast<MDNode>(DVI->getVariable());
  if (!N)
    return;

  auto *DV = dyn_cast<DILocalVariable>(N);
  if (!DV)
    return;

  if (!NodesSeen.insert(DV).second)
    return;
  processScope(DV->getScope());
  processType(DV->getType().resolve());
}

bool DebugInfoFinder::addType(DIType *DT) {
  if (!DT)
    return false;

  if (!NodesSeen.insert(DT).second)
    return false;

  TYs.push_back(const_cast<DIType *>(DT));
  return true;
}

bool DebugInfoFinder::addCompileUnit(DICompileUnit *CU) {
  if (!CU)
    return false;
  if (!NodesSeen.insert(CU).second)
    return false;

  CUs.push_back(CU);
  return true;
}

bool DebugInfoFinder::addGlobalVariable(DIGlobalVariable *DIG) {
  if (!DIG)
    return false;

  if (!NodesSeen.insert(DIG).second)
    return false;

  GVs.push_back(DIG);
  return true;
}

bool DebugInfoFinder::addSubprogram(DISubprogram *SP) {
  if (!SP)
    return false;

  if (!NodesSeen.insert(SP).second)
    return false;

  SPs.push_back(SP);
  return true;
}

bool DebugInfoFinder::addScope(DIScope *Scope) {
  if (!Scope)
    return false;
  // FIXME: Ocaml binding generates a scope with no content, we treat it
  // as null for now.
  if (Scope->getNumOperands() == 0)
    return false;
  if (!NodesSeen.insert(Scope).second)
    return false;
  Scopes.push_back(Scope);
  return true;
}

bool llvm::stripDebugInfo(Function &F) {
  bool Changed = false;
  if (F.getSubprogram()) {
    Changed = true;
    F.setSubprogram(nullptr);
  }

  for (BasicBlock &BB : F) {
    for (auto II = BB.begin(), End = BB.end(); II != End;) {
      Instruction &I = *II++; // We may delete the instruction, increment now.
      if (isa<DbgInfoIntrinsic>(&I)) {
        I.eraseFromParent();
        Changed = true;
        continue;
      }
      if (I.getDebugLoc()) {
        Changed = true;
        I.setDebugLoc(DebugLoc());
      }
    }
  }
  return Changed;
}

bool llvm::StripDebugInfo(Module &M) {
  bool Changed = false;

  for (Module::named_metadata_iterator NMI = M.named_metadata_begin(),
         NME = M.named_metadata_end(); NMI != NME;) {
    NamedMDNode *NMD = &*NMI;
    ++NMI;
    if (NMD->getName().startswith("llvm.dbg.")) {
      NMD->eraseFromParent();
      Changed = true;
    }
  }

  for (Function &F : M)
    Changed |= stripDebugInfo(F);

  if (GVMaterializer *Materializer = M.getMaterializer())
    Materializer->setStripDebugInfo();

  return Changed;
}

unsigned llvm::getDebugMetadataVersionFromModule(const Module &M) {
  if (auto *Val = mdconst::dyn_extract_or_null<ConstantInt>(
          M.getModuleFlag("Debug Info Version")))
    return Val->getZExtValue();
  return 0;
}
