//===- StringTableBuilder.h - PDB String Table Builder ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file creates the "/names" stream.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_STRINGTABLEBUILDER_H
#define LLVM_DEBUGINFO_PDB_RAW_STRINGTABLEBUILDER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <vector>

namespace llvm {
class BinaryStreamWriter;

namespace pdb {

class StringTableBuilder {
public:
  // If string S does not exist in the string table, insert it.
  // Returns the ID for S.
  uint32_t insert(StringRef S);

  uint32_t finalize();
  Error commit(BinaryStreamWriter &Writer) const;

private:
  DenseMap<StringRef, uint32_t> Strings;
  uint32_t StringSize = 1;
};

} // end namespace pdb
} // end namespace llvm

#endif // LLVM_DEBUGINFO_PDB_RAW_STRINGTABLEBUILDER_H
