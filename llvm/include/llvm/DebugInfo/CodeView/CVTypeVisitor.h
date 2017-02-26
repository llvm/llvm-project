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

#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/DebugInfo/CodeView/CVRecord.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/TypeServerHandler.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace codeview {

class CVTypeVisitor {
public:
  explicit CVTypeVisitor(TypeVisitorCallbacks &Callbacks);

  void addTypeServerHandler(TypeServerHandler &Handler);

  Error visitTypeRecord(CVType &Record);
  Error visitMemberRecord(CVMemberRecord &Record);

  /// Visits the type records in Data. Sets the error flag on parse failures.
  Error visitTypeStream(const CVTypeArray &Types);
  Error visitTypeStream(CVTypeRange Types);

  Error visitFieldListMemberStream(ArrayRef<uint8_t> FieldList);
  Error visitFieldListMemberStream(msf::StreamReader Reader);

private:
  /// The interface to the class that gets notified of each visitation.
  TypeVisitorCallbacks &Callbacks;

  TinyPtrVector<TypeServerHandler *> Handlers;
};

} // end namespace codeview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_CVTYPEVISITOR_H
