//===--------------------- Support.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements a few helper functions used by various backend
/// components.
///
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "llvm/MC/MCSchedule.h"

namespace mca {

using namespace llvm;

void computeProcResourceMasks(const MCSchedModel &SM,
                              SmallVectorImpl<uint64_t> &Masks) {
  unsigned ProcResourceID = 0;

  // Create a unique bitmask for every processor resource unit.
  // Skip resource at index 0, since it always references 'InvalidUnit'.
  Masks.resize(SM.getNumProcResourceKinds());
  for (unsigned I = 1, E = SM.getNumProcResourceKinds(); I < E; ++I) {
    const MCProcResourceDesc &Desc = *SM.getProcResource(I);
    if (Desc.SubUnitsIdxBegin)
      continue;
    Masks[I] = 1ULL << ProcResourceID;
    ProcResourceID++;
  }

  // Create a unique bitmask for every processor resource group.
  for (unsigned I = 1, E = SM.getNumProcResourceKinds(); I < E; ++I) {
    const MCProcResourceDesc &Desc = *SM.getProcResource(I);
    if (!Desc.SubUnitsIdxBegin)
      continue;
    Masks[I] = 1ULL << ProcResourceID;
    for (unsigned U = 0; U < Desc.NumUnits; ++U) {
      uint64_t OtherMask = Masks[Desc.SubUnitsIdxBegin[U]];
      Masks[I] |= OtherMask;
    }
    ProcResourceID++;
  }
}
} // namespace mca
