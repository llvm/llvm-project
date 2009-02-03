//===- CloneModule.cpp - Clone an entire module ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CloneModule interface which makes a copy of an
// entire module.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/TypeSymbolTable.h"
#include "llvm/Constant.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
using namespace llvm;

/// CloneModule - Return an exact copy of the specified module.  This is not as
/// easy as it might seem because we have to worry about making copies of global
/// variables and functions, and making their (initializers and references,
/// respectively) refer to the right globals.
///
Module *llvm::CloneModule(const Module *M) {
  // Create the value map that maps things from the old module over to the new
  // module.
  DenseMap<const Value*, Value*> ValueMap;
  return CloneModule(M, ValueMap);
}

Module *llvm::CloneModule(const Module *M,
                          DenseMap<const Value*, Value*> &ValueMap) {
  // First off, we need to create the new module...
  Module *New = new Module(M->getModuleIdentifier());
  New->setDataLayout(M->getDataLayout());
  New->setTargetTriple(M->getTargetTriple());
  New->setModuleInlineAsm(M->getModuleInlineAsm());

  // Copy all of the type symbol table entries over.
  const TypeSymbolTable &TST = M->getTypeSymbolTable();
  for (TypeSymbolTable::const_iterator TI = TST.begin(), TE = TST.end(); 
       TI != TE; ++TI)
    New->addTypeName(TI->first, TI->second);
  
  // Copy all of the dependent libraries over.
  for (Module::lib_iterator I = M->lib_begin(), E = M->lib_end(); I != E; ++I)
    New->addLibrary(*I);

  // Loop over all of the global variables, making corresponding globals in the
  // new module.  Here we add them to the ValueMap and to the new Module.  We
  // don't worry about attributes or initializers, they will come later.
  //
  for (Module::const_global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I) {
    GlobalVariable *GV = new GlobalVariable(I->getType()->getElementType(),
                                            false,
                                            GlobalValue::ExternalLinkage, 0,
                                            I->getName(), New);
    GV->setAlignment(I->getAlignment());
    ValueMap[I] = GV;
  }

  // Loop over the functions in the module, making external functions as before
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I) {
    Function *NF =
      Function::Create(cast<FunctionType>(I->getType()->getElementType()),
                       GlobalValue::ExternalLinkage, I->getName(), New);
    NF->copyAttributesFrom(I);
    ValueMap[I] = NF;
  }

  // Loop over the aliases in the module
  for (Module::const_alias_iterator I = M->alias_begin(), E = M->alias_end();
       I != E; ++I)
    ValueMap[I] = new GlobalAlias(I->getType(), GlobalAlias::ExternalLinkage,
                                  I->getName(), NULL, New);
  
  // Now that all of the things that global variable initializer can refer to
  // have been created, loop through and copy the global variable referrers
  // over...  We also set the attributes on the global now.
  //
  for (Module::const_global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I) {
    GlobalVariable *GV = cast<GlobalVariable>(ValueMap[I]);
    if (I->hasInitializer())
      GV->setInitializer(cast<Constant>(MapValue(I->getInitializer(),
                                                 ValueMap)));
    GV->setLinkage(I->getLinkage());
    GV->setThreadLocal(I->isThreadLocal());
    GV->setConstant(I->isConstant());
  }

  // Similarly, copy over function bodies now...
  //
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I) {
    Function *F = cast<Function>(ValueMap[I]);
    if (!I->isDeclaration()) {
      Function::arg_iterator DestI = F->arg_begin();
      for (Function::const_arg_iterator J = I->arg_begin(); J != I->arg_end();
           ++J) {
        DestI->setName(J->getName());
        ValueMap[J] = DestI++;
      }

      std::vector<ReturnInst*> Returns;  // Ignore returns cloned...
      CloneFunctionInto(F, I, ValueMap, Returns);
    }

    F->setLinkage(I->getLinkage());
  }

  // And aliases
  for (Module::const_alias_iterator I = M->alias_begin(), E = M->alias_end();
       I != E; ++I) {
    GlobalAlias *GA = cast<GlobalAlias>(ValueMap[I]);
    GA->setLinkage(I->getLinkage());
    if (const Constant* C = I->getAliasee())
      GA->setAliasee(cast<Constant>(MapValue(C, ValueMap)));
  }
  
  return New;
}
