//===- MemoryTypeTableBuilder.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_MEMORYTYPETABLEBUILDER_H
#define LLVM_DEBUGINFO_CODEVIEW_MEMORYTYPETABLEBUILDER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/CodeView/TypeTableBuilder.h"
#include <vector>

namespace llvm {
namespace codeview {

class MemoryTypeTableBuilder : public TypeTableBuilder {
public:
  MemoryTypeTableBuilder() {}

  bool empty() const { return Records.empty(); }

  template <typename TFunc> void ForEachRecord(TFunc Func) {
    uint32_t Index = TypeIndex::FirstNonSimpleIndex;

    for (StringRef R : Records) {
      Func(TypeIndex(Index), R);
      ++Index;
    }
  }

protected:
  TypeIndex writeRecord(llvm::StringRef Data) override;

private:
  std::vector<StringRef> Records;
  BumpPtrAllocator RecordStorage;
  DenseMap<StringRef, TypeIndex> HashedRecords;
};

} // end namespace codeview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_MEMORYTYPETABLEBUILDER_H
