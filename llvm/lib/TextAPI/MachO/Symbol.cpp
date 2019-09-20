//===- Symbol.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the Symbol.
//
//===----------------------------------------------------------------------===//

#include "llvm/TextAPI/MachO/Symbol.h"
#include <string>

namespace llvm {
namespace MachO {

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void Symbol::dump(raw_ostream &OS) const {
  std::string Result;
  if (isUndefined())
    Result += "(undef) ";
  if (isWeakDefined())
    Result += "(weak-def) ";
  if (isWeakReferenced())
    Result += "(weak-ref) ";
  if (isThreadLocalValue())
    Result += "(tlv) ";
  switch (Kind) {
  case SymbolKind::GlobalSymbol:
    Result += Name.str();
    break;
  case SymbolKind::ObjectiveCClass:
    Result += "(ObjC Class) " + Name.str();
    break;
  case SymbolKind::ObjectiveCClassEHType:
    Result += "(ObjC Class EH) " + Name.str();
    break;
  case SymbolKind::ObjectiveCInstanceVariable:
    Result += "(ObjC IVar) " + Name.str();
    break;
  }
  OS << Result;
}
#endif

Symbol::const_filtered_target_range
Symbol::targets(ArchitectureSet Architectures) const {
  std::function<bool(const Target &)> FN =
      [Architectures](const Target &Target) {
        return Architectures.has(Target.Arch);
      };
  return make_filter_range(Targets, FN);
}

} // end namespace MachO.
} // end namespace llvm.
