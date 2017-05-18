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

  Error visitTypeRecord(CVType &Record, TypeIndex Index);
  Error visitTypeRecord(CVType &Record);
  Error visitMemberRecord(CVMemberRecord Record);

  /// Visits the type records in Data. Sets the error flag on parse failures.
  Error visitTypeStream(const CVTypeArray &Types);
  Error visitTypeStream(CVTypeRange Types);

  Error visitFieldListMemberStream(ArrayRef<uint8_t> FieldList);
  Error visitFieldListMemberStream(BinaryStreamReader Reader);

private:
  Expected<bool> handleTypeServer(CVType &Record);
  Error finishVisitation(CVType &Record);

  /// The interface to the class that gets notified of each visitation.
  TypeVisitorCallbacks &Callbacks;

  TinyPtrVector<TypeServerHandler *> Handlers;
};

enum VisitorDataSource {
  VDS_BytesPresent, // The record bytes are passed into the the visitation
                    // function.  The algorithm should first deserialize them
                    // before passing them on through the pipeline.
  VDS_BytesExternal // The record bytes are not present, and it is the
                    // responsibility of the visitor callback interface to
                    // supply the bytes.
};

Error visitTypeRecord(CVType &Record, TypeIndex Index,
                      TypeVisitorCallbacks &Callbacks,
                      VisitorDataSource Source = VDS_BytesPresent,
                      TypeServerHandler *TS = nullptr);
Error visitTypeRecord(CVType &Record, TypeVisitorCallbacks &Callbacks,
                      VisitorDataSource Source = VDS_BytesPresent,
                      TypeServerHandler *TS = nullptr);

Error visitMemberRecord(CVMemberRecord Record, TypeVisitorCallbacks &Callbacks,
                        VisitorDataSource Source = VDS_BytesPresent);
Error visitMemberRecord(TypeLeafKind Kind, ArrayRef<uint8_t> Record,
                        TypeVisitorCallbacks &Callbacks);

Error visitMemberRecordStream(ArrayRef<uint8_t> FieldList,
                              TypeVisitorCallbacks &Callbacks);

Error visitTypeStream(const CVTypeArray &Types, TypeVisitorCallbacks &Callbacks,
                      TypeServerHandler *TS = nullptr);
Error visitTypeStream(CVTypeRange Types, TypeVisitorCallbacks &Callbacks,
                      TypeServerHandler *TS = nullptr);

} // end namespace codeview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_CVTYPEVISITOR_H
