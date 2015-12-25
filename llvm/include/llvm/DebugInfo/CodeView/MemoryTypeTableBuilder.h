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

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/CodeView/TypeTableBuilder.h"
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace llvm {
namespace codeview {

class MemoryTypeTableBuilder : public TypeTableBuilder {
public:
  class Record {
  public:
    explicit Record(llvm::StringRef RData);

    const char *data() const { return Data.get(); }
    uint16_t size() const { return Size; }

  private:
    uint16_t Size;
    std::unique_ptr<char[]> Data;
  };

private:
  class RecordHash : std::unary_function<llvm::StringRef, size_t> {
  public:
    size_t operator()(llvm::StringRef Val) const {
      return static_cast<size_t>(llvm::hash_value(Val));
    }
  };

public:
  MemoryTypeTableBuilder() {}

  template <typename TFunc> void ForEachRecord(TFunc Func) {
    uint32_t Index = TypeIndex::FirstNonSimpleIndex;

    for (const std::unique_ptr<Record> &R : Records) {
      Func(TypeIndex(Index), R.get());
      ++Index;
    }
  }

private:
  virtual TypeIndex writeRecord(llvm::StringRef Data) override;

private:
  std::vector<std::unique_ptr<Record>> Records;
  std::unordered_map<llvm::StringRef, TypeIndex, RecordHash> HashedRecords;
};
}
}

#endif
