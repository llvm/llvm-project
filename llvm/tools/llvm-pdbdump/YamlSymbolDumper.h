//===- YamlSymbolDumper.h ------------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_YAMLSYMBOLDUMPER_H
#define LLVM_TOOLS_LLVMPDBDUMP_YAMLSYMBOLDUMPER_H

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/SymbolVisitorCallbacks.h"
#include "llvm/Support/YAMLTraits.h"

namespace llvm {
namespace pdb {
namespace yaml {
struct SerializationContext;
}
}
namespace codeview {
namespace yaml {
class YamlSymbolDumper : public SymbolVisitorCallbacks {
public:
  YamlSymbolDumper(llvm::yaml::IO &IO) : YamlIO(IO) {}

  virtual Error visitSymbolBegin(CVSymbol &Record) override;

#define SYMBOL_RECORD(EnumName, EnumVal, Name)                                 \
  Error visitKnownRecord(CVSymbol &CVR, Name &Record) override {               \
    visitKnownRecordImpl(#Name, CVR, Record);                                  \
    return Error::success();                                                   \
  }
#define SYMBOL_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "llvm/DebugInfo/CodeView/CVSymbolTypes.def"

private:
  template <typename T>
  void visitKnownRecordImpl(const char *Name, CVSymbol &Type, T &Record) {
    YamlIO.mapRequired(Name, Record);
  }

  llvm::yaml::IO &YamlIO;
};
}
}
}

namespace llvm {
namespace yaml {
template <> struct ScalarEnumerationTraits<codeview::SymbolKind> {
  static void enumeration(IO &io, codeview::SymbolKind &Value);
};

#define SYMBOL_RECORD(EnumName, EnumVal, Name)                                 \
  template <> struct MappingTraits<codeview::Name> {                           \
    static void mapping(IO &IO, codeview::Name &Obj);                          \
  };
#define SYMBOL_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "llvm/DebugInfo/CodeView/CVSymbolTypes.def"
}
}

#endif
