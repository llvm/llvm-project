//===- CodeViewYAMLSymbols.h - CodeView YAMLIO Symbol implementation ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines classes for handling the YAML representation of CodeView
// Debug Info.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECTYAML_CODEVIEWYAMLSYMBOLS_H
#define LLVM_OBJECTYAML_CODEVIEWYAMLSYMBOLS_H

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/ObjectYAML/YAML.h"

namespace llvm {
namespace CodeViewYAML {
namespace detail {
struct SymbolRecordBase;
}

struct SymbolRecord {
  std::shared_ptr<detail::SymbolRecordBase> Symbol;

  codeview::CVSymbol
  toCodeViewSymbol(BumpPtrAllocator &Allocator,
                   codeview::CodeViewContainer Container) const;
  static Expected<SymbolRecord> fromCodeViewSymbol(codeview::CVSymbol Symbol);
};

} // namespace CodeViewYAML
} // namespace llvm

LLVM_YAML_DECLARE_MAPPING_TRAITS(CodeViewYAML::SymbolRecord)
LLVM_YAML_IS_SEQUENCE_VECTOR(CodeViewYAML::SymbolRecord)

#endif
