//===-- ValueSymbolTable.cpp - Implement the ValueSymbolTable class -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ValueSymbolTable class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "valuesymtab"
#include "llvm/GlobalValue.h"
#include "llvm/Type.h"
#include "llvm/ValueSymbolTable.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

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

// Insert a value into the symbol table with the specified name...
//
void ValueSymbolTable::reinsertValue(Value* V) {
  assert(V->hasName() && "Can't insert nameless Value into symbol table");

  // Try inserting the name, assuming it won't conflict.
  if (vmap.insert(V->Name)) {
    //DEBUG(dbgs() << " Inserted value: " << V->Name << ": " << *V << "\n");
    return;
  }
  
  // Otherwise, there is a naming conflict.  Rename this value.
  SmallString<256> UniqueName(V->getName().begin(), V->getName().end());

  // The name is too already used, just free it so we can allocate a new name.
  V->Name->Destroy();
  
  unsigned BaseSize = UniqueName.size();
  while (1) {
    // Trim any suffix off and append the next number.
    UniqueName.resize(BaseSize);
    raw_svector_ostream(UniqueName) << ++LastUnique;

    // Try insert the vmap entry with this suffix.
    ValueName &NewName = vmap.GetOrCreateValue(UniqueName);
    if (NewName.getValue() == 0) {
      // Newly inserted name.  Success!
      NewName.setValue(V);
      V->Name = &NewName;
     //DEBUG(dbgs() << " Inserted value: " << UniqueName << ": " << *V << "\n");
      return;
    }
  }
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
  ValueName &Entry = vmap.GetOrCreateValue(Name);
  if (Entry.getValue() == 0) {
    Entry.setValue(V);
    //DEBUG(dbgs() << " Inserted value: " << Entry.getKeyData() << ": "
    //           << *V << "\n");
    return &Entry;
  }
  
  // Otherwise, there is a naming conflict.  Rename this value.
  SmallString<256> UniqueName(Name.begin(), Name.end());
  
  while (1) {
    // Trim any suffix off and append the next number.
    UniqueName.resize(Name.size());
    raw_svector_ostream(UniqueName) << ++LastUnique;
    
    // Try insert the vmap entry with this suffix.
    ValueName &NewName = vmap.GetOrCreateValue(UniqueName);
    if (NewName.getValue() == 0) {
      // Newly inserted name.  Success!
      NewName.setValue(V);
     //DEBUG(dbgs() << " Inserted value: " << UniqueName << ": " << *V << "\n");
      return &NewName;
    }
  }
}


// dump - print out the symbol table
//
void ValueSymbolTable::dump() const {
  //DEBUG(dbgs() << "ValueSymbolTable:\n");
  for (const_iterator I = begin(), E = end(); I != E; ++I) {
    //DEBUG(dbgs() << "  '" << I->getKeyData() << "' = ");
    I->getValue()->dump();
    //DEBUG(dbgs() << "\n");
  }
}
