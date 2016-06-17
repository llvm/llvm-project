//===- CVTypeVisitor.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_CVTYPEVISITOR_H
#define LLVM_DEBUGINFO_CODEVIEW_CVTYPEVISITOR_H

#include "llvm/DebugInfo/CodeView/CVRecord.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace codeview {

class CVTypeVisitor {
public:
  explicit CVTypeVisitor(TypeVisitorCallbacks &Callbacks);

  Error visitTypeRecord(const CVRecord<TypeLeafKind> &Record);

  /// Visits the type records in Data. Sets the error flag on parse failures.
  Error visitTypeStream(const CVTypeArray &Types);

  Error skipPadding(ArrayRef<uint8_t> &Data);

  /// Visits individual member records of a field list record. Member records do
  /// not describe their own length, and need special handling.
  Error visitFieldList(const CVRecord<TypeLeafKind> &Record);

private:
  /// The interface to the class that gets notified of each visitation.
  TypeVisitorCallbacks &Callbacks;
};

} // end namespace codeview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_CVTYPEVISITOR_H
