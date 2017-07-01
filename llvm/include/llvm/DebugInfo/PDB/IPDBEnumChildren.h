//===- IPDBEnumChildren.h - base interface for child enumerator -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_IPDBENUMCHILDREN_H
#define LLVM_DEBUGINFO_PDB_IPDBENUMCHILDREN_H

#include <cstdint>
#include <memory>

namespace llvm {
namespace pdb {

template <typename ChildType> class IPDBEnumChildren {
public:
  using ChildTypePtr = std::unique_ptr<ChildType>;
  using MyType = IPDBEnumChildren<ChildType>;

  virtual ~IPDBEnumChildren() = default;

  virtual uint32_t getChildCount() const = 0;
  virtual ChildTypePtr getChildAtIndex(uint32_t Index) const = 0;
  virtual ChildTypePtr getNext() = 0;
  virtual void reset() = 0;
  virtual MyType *clone() const = 0;
};

} // end namespace pdb
} // end namespace llvm

#endif // LLVM_DEBUGINFO_PDB_IPDBENUMCHILDREN_H
