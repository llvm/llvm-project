//===-- CompactTypeDumpVisitor.h - CodeView type info dumper ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_COMPACTTYPEDUMPVISITOR_H
#define LLVM_DEBUGINFO_CODEVIEW_COMPACTTYPEDUMPVISITOR_H

#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"

namespace llvm {
class ScopedPrinter;
namespace codeview {
class TypeDatabase;
}

namespace pdb {

/// Dumper for CodeView type streams found in COFF object files and PDB files.
/// Dumps records on a single line, and ignores member records.
class CompactTypeDumpVisitor : public codeview::TypeVisitorCallbacks {
public:
  CompactTypeDumpVisitor(codeview::TypeDatabase &TypeDB, ScopedPrinter *W);
  CompactTypeDumpVisitor(codeview::TypeDatabase &TypeDB,
                         codeview::TypeIndex FirstTI, ScopedPrinter *W);

  /// Paired begin/end actions for all types. Receives all record data,
  /// including the fixed-length record prefix.
  Error visitTypeBegin(codeview::CVType &Record) override;
  Error visitTypeEnd(codeview::CVType &Record) override;

private:
  ScopedPrinter *W;

  codeview::TypeIndex TI;
  uint32_t Offset;
  codeview::TypeDatabase &TypeDB;
};

} // end namespace pdb
} // end namespace llvm

#endif
