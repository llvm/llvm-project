//===- FindUsedTypes.cpp - Find all Types used by a module ----------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This pass is used to seek out all of the types in use by the program.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/FindUsedTypes.h"
#include "llvm/Assembly/CachedWriter.h"
#include "llvm/SymbolTable.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Support/InstIterator.h"

static RegisterAnalysis<FindUsedTypes>
X("printusedtypes", "Find Used Types");

// stub to help linkage
void FindUsedTypes::stub() {}

// IncorporateType - Incorporate one type and all of its subtypes into the
// collection of used types.
//
void FindUsedTypes::IncorporateType(const Type *Ty) {
  if (UsedTypes.count(Ty)) return;  // Already contain Ty.
                             
  // If ty doesn't already exist in the used types map, add it now.
  //
  UsedTypes.insert(Ty);
  
  // Make sure to add any types this type references now.
  //
  for (Type::subtype_iterator I = Ty->subtype_begin(), E = Ty->subtype_end();
       I != E; ++I)
    IncorporateType(*I);
}

void FindUsedTypes::IncorporateSymbolTable(const SymbolTable &ST) {
  SymbolTable::const_iterator TI = ST.find(Type::TypeTy);
  if (TI == ST.end()) return;  // No named types

  for (SymbolTable::type_const_iterator I = TI->second.begin(),
         E = TI->second.end(); I != E; ++I)
    IncorporateType(cast<Type>(I->second));
}

// run - This incorporates all types used by the specified module
//
bool FindUsedTypes::run(Module &m) {
  UsedTypes.clear();  // reset if run multiple times...

  IncorporateSymbolTable(m.getSymbolTable());

  // Loop over global variables, incorporating their types
  for (Module::const_giterator I = m.gbegin(), E = m.gend(); I != E; ++I)
    IncorporateType(I->getType());

  for (Module::iterator MI = m.begin(), ME = m.end(); MI != ME; ++MI) {
    IncorporateType(MI->getType());
    const Function &F = *MI;
    IncorporateSymbolTable(F.getSymbolTable());
  
    // Loop over all of the instructions in the function, adding their return
    // type as well as the types of their operands.
    //
    for (const_inst_iterator II = inst_begin(F), IE = inst_end(F);
         II != IE; ++II) {
      const Instruction *I = *II;
      const Type *Ty = I->getType();
    
      IncorporateType(Ty);  // Incorporate the type of the instruction
      for (User::const_op_iterator OI = I->op_begin(), OE = I->op_end();
           OI != OE; ++OI)
        if ((*OI)->getType() != Ty)         // Avoid set lookup in common case
          IncorporateType((*OI)->getType());// Insert inst operand types as well
    }
  }
 
  return false;
}

// Print the types found in the module.  If the optional Module parameter is
// passed in, then the types are printed symbolically if possible, using the
// symbol table from the module.
//
void FindUsedTypes::print(std::ostream &o, const Module *M) const {
  o << "Types in use by this module:\n";
  if (M) {
    CachedWriter CW(M, o);
    for (std::set<const Type *>::const_iterator I = UsedTypes.begin(),
           E = UsedTypes.end(); I != E; ++I)
      CW << "  " << *I << "\n";
  } else
    for (std::set<const Type *>::const_iterator I = UsedTypes.begin(),
           E = UsedTypes.end(); I != E; ++I)
      o << "  " << *I << "\n";
}
