//===- lib/MC/MCModule.cpp - MCModule implementation ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAtom.h"
#include "llvm/MC/MCModule.h"

using namespace llvm;

MCAtom *MCModule::createAtom(MCAtom::AtomType Type,
                             uint64_t Begin, uint64_t End) {
  assert(Begin < End && "Creating MCAtom with endpoints reversed?");

  // Check for atoms already covering this range.
  IntervalMap<uint64_t, MCAtom*>::iterator I = OffsetMap.find(Begin);
  assert((!I.valid() || I.start() < End) && "Offset range already occupied!");

  // Create the new atom and add it to our maps.
  MCAtom *NewAtom = new MCAtom(Type, this, Begin, End);
  AtomAllocationTracker.insert(NewAtom);
  OffsetMap.insert(Begin, End, NewAtom);
  return NewAtom;
}

// remap - Update the interval mapping for an atom.
void MCModule::remap(MCAtom *Atom, uint64_t NewBegin, uint64_t NewEnd) {
  // Find and erase the old mapping.
  IntervalMap<uint64_t, MCAtom*>::iterator I = OffsetMap.find(Atom->Begin);
  assert(I.valid() && "Atom offset not found in module!");
  assert(*I == Atom && "Previous atom mapping was invalid!");
  I.erase();

  // Insert the new mapping.
  OffsetMap.insert(NewBegin, NewEnd, Atom);

  // Update the atom internal bounds.
  Atom->Begin = NewBegin;
  Atom->End = NewEnd;
}

