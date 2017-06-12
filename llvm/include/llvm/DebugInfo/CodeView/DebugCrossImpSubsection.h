//===- DebugCrossExSubsection.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_DEBUGCROSSIMPSUBSECTION_H
#define LLVM_DEBUGINFO_CODEVIEW_DEBUGCROSSIMPSUBSECTION_H

#include "llvm/ADT/StringMap.h"
#include "llvm/DebugInfo/CodeView/DebugSubsection.h"
#include "llvm/Support/BinaryStreamArray.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/Endian.h"

namespace llvm {
namespace codeview {

struct CrossModuleImportItem {
  const CrossModuleImport *Header = nullptr;
  llvm::FixedStreamArray<support::ulittle32_t> Imports;
};
}
}

namespace llvm {
template <> struct VarStreamArrayExtractor<codeview::CrossModuleImportItem> {
public:
  typedef void ContextType;

  Error operator()(BinaryStreamRef Stream, uint32_t &Len,
                   codeview::CrossModuleImportItem &Item);
};
}

namespace llvm {
namespace codeview {
class DebugStringTableSubsection;

class DebugCrossModuleImportsSubsectionRef final : public DebugSubsectionRef {
  typedef VarStreamArray<CrossModuleImportItem> ReferenceArray;
  typedef ReferenceArray::Iterator Iterator;

public:
  DebugCrossModuleImportsSubsectionRef()
      : DebugSubsectionRef(DebugSubsectionKind::CrossScopeImports) {}

  static bool classof(const DebugSubsectionRef *S) {
    return S->kind() == DebugSubsectionKind::CrossScopeImports;
  }

  Error initialize(BinaryStreamReader Reader);
  Error initialize(BinaryStreamRef Stream);

  Iterator begin() const { return References.begin(); }
  Iterator end() const { return References.end(); }

private:
  ReferenceArray References;
};

class DebugCrossModuleImportsSubsection final : public DebugSubsection {
public:
  explicit DebugCrossModuleImportsSubsection(
      DebugStringTableSubsection &Strings)
      : DebugSubsection(DebugSubsectionKind::CrossScopeImports),
        Strings(Strings) {}

  static bool classof(const DebugSubsection *S) {
    return S->kind() == DebugSubsectionKind::CrossScopeImports;
  }

  void addImport(StringRef Module, uint32_t ImportId);

  uint32_t calculateSerializedSize() const override;
  Error commit(BinaryStreamWriter &Writer) const override;

private:
  DebugStringTableSubsection &Strings;
  StringMap<std::vector<support::ulittle32_t>> Mappings;
};
}
}

#endif
