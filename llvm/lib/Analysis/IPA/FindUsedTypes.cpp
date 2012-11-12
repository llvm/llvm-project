//===- FindUsedTypes.cpp - Find all Types used by a module ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is used to seek out all of the types in use by the program.  Note
// that this analysis explicitly does not include types only used by the symbol
// table.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/FindUsedTypes.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

char FindUsedTypes::ID = 0;
INITIALIZE_PASS(FindUsedTypes, "print-used-types",
                "Find Used Types", false, true)

// IncorporateType - Incorporate one type and all of its subtypes into the
// collection of used types.
//
void FindUsedTypes::IncorporateType(Type *Ty) {
  // If ty doesn't already exist in the used types map, add it now, otherwise
  // return.
  if (!UsedTypes.insert(Ty)) return;  // Already contain Ty.

  // Make sure to add any types this type references now.
  //
  for (Type::subtype_iterator I = Ty->subtype_begin(), E = Ty->subtype_end();
       I != E; ++I)
    IncorporateType(*I);
}

void FindUsedTypes::IncorporateValue(const Value *V) {
  IncorporateType(V->getType());

  // If this is a constant, it could be using other types...
  if (const Constant *C = dyn_cast<Constant>(V)) {
    if (!isa<GlobalValue>(C))
      for (User::const_op_iterator OI = C->op_begin(), OE = C->op_end();
           OI != OE; ++OI)
        IncorporateValue(*OI);
  }
}


// run - This incorporates all types used by the specified module
//
bool FindUsedTypes::runOnModule(Module &m) {
  UsedTypes.clear();  // reset if run multiple times...

  // Loop over global variables, incorporating their types
  for (Module::const_global_iterator I = m.global_begin(), E = m.global_end();
       I != E; ++I) {
    IncorporateType(I->getType());
    if (I->hasInitializer())
      IncorporateValue(I->getInitializer());
  }

  for (Module::iterator MI = m.begin(), ME = m.end(); MI != ME; ++MI) {
    IncorporateType(MI->getType());
    const Function &F = *MI;

    // Loop over all of the instructions in the function, adding their return
    // type as well as the types of their operands.
    //
    for (const_inst_iterator II = inst_begin(F), IE = inst_end(F);
         II != IE; ++II) {
      const Instruction &I = *II;

      IncorporateType(I.getType());  // Incorporate the type of the instruction
      for (User::const_op_iterator OI = I.op_begin(), OE = I.op_end();
           OI != OE; ++OI)
        IncorporateValue(*OI);  // Insert inst operand types as well
    }
  }

  return false;
}

// Print the types found in the module.  If the optional Module parameter is
// passed in, then the types are printed symbolically if possible, using the
// symbol table from the module.
//
void FindUsedTypes::print(raw_ostream &OS, const Module *M) const {
  OS << "Types in use by this module:\n";
  for (SetVector<Type *>::const_iterator I = UsedTypes.begin(),
       E = UsedTypes.end(); I != E; ++I) {
    OS << "   " << **I << '\n';
  }
}
