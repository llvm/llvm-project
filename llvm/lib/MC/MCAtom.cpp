//===- lib/MC/MCAtom.cpp - MCAtom implementation --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAtom.h"
#include "llvm/MC/MCModule.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

void MCAtom::addInst(const MCInst &I, uint64_t Address, unsigned Size) {
  assert(Type == TextAtom && "Trying to add MCInst to a non-text atom!");

  assert(Address < End+Size &&
         "Instruction not contiguous with end of atom!");
  if (Address > End)
    Parent->remap(this, Begin, End+Size);

  Text.push_back(std::make_pair(Address, I));
}

void MCAtom::addData(const MCData &D) {
  assert(Type == DataAtom && "Trying to add MCData to a non-data atom!");
  Parent->remap(this, Begin, End+1);

  Data.push_back(D);
}

MCAtom *MCAtom::split(uint64_t SplitPt) {
  assert((SplitPt > Begin && SplitPt <= End) &&
         "Splitting at point not contained in atom!");

  // Compute the new begin/end points.
  uint64_t LeftBegin = Begin;
  uint64_t LeftEnd = SplitPt - 1;
  uint64_t RightBegin = SplitPt;
  uint64_t RightEnd = End;

  // Remap this atom to become the lower of the two new ones.
  Parent->remap(this, LeftBegin, LeftEnd);

  // Create a new atom for the higher atom.
  MCAtom *RightAtom = Parent->createAtom(Type, RightBegin, RightEnd);

  // Split the contents of the original atom between it and the new one.  The
  // precise method depends on whether this is a data or a text atom.
  if (isDataAtom()) {
    std::vector<MCData>::iterator I = Data.begin() + (RightBegin - LeftBegin);

    assert(I != Data.end() && "Split point not found in range!");

    std::copy(I, Data.end(), RightAtom->Data.end());
    Data.erase(I, Data.end());
  } else if (isTextAtom()) {
    std::vector<std::pair<uint64_t, MCInst> >::iterator I = Text.begin();

    while (I != Text.end() && I->first < SplitPt) ++I;

    assert(I != Text.end() && "Split point not found in disassembly!");
    assert(I->first == SplitPt &&
           "Split point does not fall on instruction boundary!");

    std::copy(I, Text.end(), RightAtom->Text.end());
    Text.erase(I, Text.end());
  } else
    llvm_unreachable("Unknown atom type!");

  return RightAtom;
}

void MCAtom::truncate(uint64_t TruncPt) {
  assert((TruncPt >= Begin && TruncPt < End) &&
         "Truncation point not contained in atom!");

  Parent->remap(this, Begin, TruncPt);

  if (isDataAtom()) {
    Data.resize(TruncPt - Begin + 1);
  } else if (isTextAtom()) {
    std::vector<std::pair<uint64_t, MCInst> >::iterator I = Text.begin();

    while (I != Text.end() && I->first <= TruncPt) ++I;

    assert(I != Text.end() && "Truncation point not found in disassembly!");
    assert(I->first == TruncPt+1 &&
           "Truncation point does not fall on instruction boundary");

    Text.erase(I, Text.end());
  } else
    llvm_unreachable("Unknown atom type!");
}

