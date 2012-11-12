//===-- MCFunction.h ------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the data structures to hold a CFG reconstructed from
// machine code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECTDUMP_MCFUNCTION_H
#define LLVM_OBJECTDUMP_MCFUNCTION_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/MC/MCInst.h"
#include <map>

namespace llvm {

class MCDisassembler;
class MCInstrAnalysis;
class MemoryObject;
class raw_ostream;

/// MCDecodedInst - Small container to hold an MCInst and associated info like
/// address and size.
struct MCDecodedInst {
  uint64_t Address;
  uint64_t Size;
  MCInst Inst;

  MCDecodedInst() {}
  MCDecodedInst(uint64_t Address, uint64_t Size, MCInst Inst)
    : Address(Address), Size(Size), Inst(Inst) {}

  bool operator<(const MCDecodedInst &RHS) const {
    return Address < RHS.Address;
  }
};

/// MCBasicBlock - Consists of multiple MCDecodedInsts and a list of successing
/// MCBasicBlocks.
class MCBasicBlock {
  std::vector<MCDecodedInst> Insts;
  typedef DenseSet<uint64_t> SetTy;
  SetTy Succs;
public:
  ArrayRef<MCDecodedInst> getInsts() const { return Insts; }

  typedef SetTy::const_iterator succ_iterator;
  succ_iterator succ_begin() const { return Succs.begin(); }
  succ_iterator succ_end() const { return Succs.end(); }

  bool contains(uint64_t Addr) const { return Succs.count(Addr); }

  void addInst(const MCDecodedInst &Inst) { Insts.push_back(Inst); }
  void addSucc(uint64_t Addr) { Succs.insert(Addr); }

  bool operator<(const MCBasicBlock &RHS) const {
    return Insts.size() < RHS.Insts.size();
  }
};

/// MCFunction - Represents a named function in machine code, containing
/// multiple MCBasicBlocks.
class MCFunction {
  const StringRef Name;
  // Keep BBs sorted by address.
  typedef std::vector<std::pair<uint64_t, MCBasicBlock> > MapTy;
  MapTy Blocks;
public:
  MCFunction(StringRef Name) : Name(Name) {}

  // Create an MCFunction from a region of binary machine code.
  static MCFunction
  createFunctionFromMC(StringRef Name, const MCDisassembler *DisAsm,
                       const MemoryObject &Region, uint64_t Start, uint64_t End,
                       const MCInstrAnalysis *Ana, raw_ostream &DebugOut,
                       SmallVectorImpl<uint64_t> &Calls);

  typedef MapTy::const_iterator iterator;
  iterator begin() const { return Blocks.begin(); }
  iterator end() const { return Blocks.end(); }

  StringRef getName() const { return Name; }

  MCBasicBlock &addBlock(uint64_t Address, const MCBasicBlock &BB) {
    Blocks.push_back(std::make_pair(Address, BB));
    return Blocks.back().second;
  }
};

}

#endif
