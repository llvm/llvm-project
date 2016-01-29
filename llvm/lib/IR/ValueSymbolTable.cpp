//===-- ValueSymbolTable.cpp - Implement the ValueSymbolTable class -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ValueSymbolTable class for the IR library.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "valuesymtab"

// Class destructor
ValueSymbolTable::~ValueSymbolTable() {
#ifndef NDEBUG   // Only do this in -g mode...
  for (iterator VI = vmap.begin(), VE = vmap.end(); VI != VE; ++VI)
    dbgs() << "Value still in symbol table! Type = '"
           << *VI->getValue()->getType() << "' Name = '"
           << VI->getKeyData() << "'\n";
  assert(vmap.empty() && "Values remain in symbol table!");
#endif
}

ValueName *ValueSymbolTable::makeUniqueName(Value *V,
                                            SmallString<256> &UniqueName) {
  unsigned BaseSize = UniqueName.size();
  while (1) {
    // Trim any suffix off and append the next number.
    UniqueName.resize(BaseSize);
    raw_svector_ostream S(UniqueName);
    if (isa<GlobalValue>(V))
      S << ".";
    S << ++LastUnique;

    // Try insert the vmap entry with this suffix.
    auto IterBool = vmap.insert(std::make_pair(UniqueName, V));
    if (IterBool.second)
      return &*IterBool.first;
  }
}

// Insert a value into the symbol table with the specified name...
//
void ValueSymbolTable::reinsertValue(Value* V) {
  assert(V->hasName() && "Can't insert nameless Value into symbol table");

  // Try inserting the name, assuming it won't conflict.
  if (vmap.insert(V->getValueName())) {
    //DEBUG(dbgs() << " Inserted value: " << V->getValueName() << ": " << *V << "\n");
    return;
  }
  
  // Otherwise, there is a naming conflict.  Rename this value.
  SmallString<256> UniqueName(V->getName().begin(), V->getName().end());

  // The name is too already used, just free it so we can allocate a new name.
  V->getValueName()->Destroy();

  ValueName *VN = makeUniqueName(V, UniqueName);
  V->setValueName(VN);
}

void ValueSymbolTable::removeValueName(ValueName *V) {
  //DEBUG(dbgs() << " Removing Value: " << V->getKeyData() << "\n");
  // Remove the value from the symbol table.
  vmap.remove(V);
}

/// createValueName - This method attempts to create a value name and insert
/// it into the symbol table with the specified name.  If it conflicts, it
/// auto-renames the name and returns that instead.
ValueName *ValueSymbolTable::createValueName(StringRef Name, Value *V) {
  // In the common case, the name is not already in the symbol table.
  auto IterBool = vmap.insert(std::make_pair(Name, V));
  if (IterBool.second) {
    //DEBUG(dbgs() << " Inserted value: " << Entry.getKeyData() << ": "
    //           << *V << "\n");
    return &*IterBool.first;
  }
  
  // Otherwise, there is a naming conflict.  Rename this value.
  SmallString<256> UniqueName(Name.begin(), Name.end());
  return makeUniqueName(V, UniqueName);
}


// dump - print out the symbol table
//
LLVM_DUMP_METHOD void ValueSymbolTable::dump() const {
  //DEBUG(dbgs() << "ValueSymbolTable:\n");
  for (const_iterator I = begin(), E = end(); I != E; ++I) {
    //DEBUG(dbgs() << "  '" << I->getKeyData() << "' = ");
    I->getValue()->dump();
    //DEBUG(dbgs() << "\n");
  }
}
