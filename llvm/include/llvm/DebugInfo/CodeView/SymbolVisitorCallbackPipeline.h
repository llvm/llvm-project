//===- SymbolVisitorCallbackPipeline.h ------------------------ *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_SYMBOLVISITORCALLBACKPIPELINE_H
#define LLVM_DEBUGINFO_CODEVIEW_SYMBOLVISITORCALLBACKPIPELINE_H

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/CodeView/SymbolVisitorCallbacks.h"

#include <vector>

namespace llvm {
namespace codeview {
class SymbolVisitorCallbackPipeline : public SymbolVisitorCallbacks {
public:
  SymbolVisitorCallbackPipeline() {}

  virtual Error visitUnknownSymbol(CVSymbol &Record) override {
    for (auto Visitor : Pipeline) {
      if (auto EC = Visitor->visitUnknownSymbol(Record))
        return EC;
    }
    return Error::success();
  }

  virtual Error visitSymbolBegin(CVSymbol &Record) override {
    for (auto Visitor : Pipeline) {
      if (auto EC = Visitor->visitSymbolBegin(Record))
        return EC;
    }
    return Error::success();
  }
  virtual Error visitSymbolEnd(CVSymbol &Record) override {
    for (auto Visitor : Pipeline) {
      if (auto EC = Visitor->visitSymbolEnd(Record))
        return EC;
    }
    return Error::success();
  }

  void addCallbackToPipeline(SymbolVisitorCallbacks &Callbacks) {
    Pipeline.push_back(&Callbacks);
  }

#define SYMBOL_RECORD(EnumName, EnumVal, Name)                                 \
  Error visitKnownRecord(CVSymbol &CVR, Name &Record) override {               \
    for (auto Visitor : Pipeline) {                                            \
      if (auto EC = Visitor->visitKnownRecord(CVR, Record))                    \
        return EC;                                                             \
    }                                                                          \
    return Error::success();                                                   \
  }
#define SYMBOL_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "llvm/DebugInfo/CodeView/CVSymbolTypes.def"

private:
  std::vector<SymbolVisitorCallbacks *> Pipeline;
};
}
}

#endif
