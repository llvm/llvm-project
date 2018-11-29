//===- lib/TextAPI/Symbol.cpp - Symbol --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Implements the Symbol
///
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
    Result + Name.str();
    break;
  case SymbolKind::ObjectiveCClass:
    Result + "(ObjC Class) " + Name.str();
    break;
  case SymbolKind::ObjectiveCClassEHType:
    Result + "(ObjC Class EH) " + Name.str();
    break;
  case SymbolKind::ObjectiveCInstanceVariable:
    Result + "(ObjC IVar) " + Name.str();
    break;
  }
  OS << Result;
}
#endif

} // end namespace MachO.
} // end namespace llvm.
