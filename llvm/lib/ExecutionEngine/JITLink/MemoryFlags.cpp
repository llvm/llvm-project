//===------------- MemoryFlags.cpp - Memory allocation flags --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/MemoryFlags.h"

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {

raw_ostream &operator<<(raw_ostream &OS, MemProt MP) {
  return OS << (((MP & MemProt::Read) != MemProt::None) ? 'R' : '-')
            << (((MP & MemProt::Write) != MemProt::None) ? 'W' : '-')
            << (((MP & MemProt::Exec) != MemProt::None) ? 'X' : '-');
}

raw_ostream &operator<<(raw_ostream &OS, MemDeallocPolicy MDP) {
  return OS << (MDP == MemDeallocPolicy::Standard ? "standard" : "finalize");
}

raw_ostream &operator<<(raw_ostream &OS, AllocGroup AG) {
  return OS << '(' << AG.getMemProt() << ", " << AG.getMemDeallocPolicy()
            << ')';
}

} // end namespace jitlink
} // end namespace llvm
