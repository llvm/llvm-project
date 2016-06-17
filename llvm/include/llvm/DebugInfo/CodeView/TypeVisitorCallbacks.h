//===- TypeVisitorCallbacks.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_TYPEVISITORCALLBACKS_H
#define LLVM_DEBUGINFO_CODEVIEW_TYPEVISITORCALLBACKS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace codeview {
class TypeVisitorCallbacks {
  friend class CVTypeVisitor;

public:
  virtual ~TypeVisitorCallbacks() {}

  /// Action to take on unknown types. By default, they are ignored.
  virtual Error visitUnknownType(const CVRecord<TypeLeafKind> &Record) {
    return Error::success();
  }
  virtual Error visitUnknownMember(const CVRecord<TypeLeafKind> &Record) {
    return Error::success();
  }

  /// Paired begin/end actions for all types. Receives all record data,
  /// including the fixed-length record prefix.
  virtual Error visitTypeBegin(const CVRecord<TypeLeafKind> &Record) {
    return Error::success();
  }
  virtual Error visitTypeEnd(const CVRecord<TypeLeafKind> &Record) {
    return Error::success();
  }

  virtual Error visitFieldListBegin(const CVRecord<TypeLeafKind> &Record) {
    return Error::success();
  }

  virtual Error visitFieldListEnd(const CVRecord<TypeLeafKind> &Record) {
    return Error::success();
  }

#define TYPE_RECORD(EnumName, EnumVal, Name)                                   \
  virtual Error visit##Name(Name##Record &Record) { return Error::success(); }
#define MEMBER_RECORD(EnumName, EnumVal, Name)                                 \
  TYPE_RECORD(EnumName, EnumVal, Name)
#define TYPE_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#define MEMBER_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "TypeRecords.def"
};
}
}

#endif