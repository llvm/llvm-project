//===- MethodListRecordBuilder.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_METHODLISTRECORDBUILDER_H
#define LLVM_DEBUGINFO_CODEVIEW_METHODLISTRECORDBUILDER_H

#include "llvm/DebugInfo/CodeView/ListRecordBuilder.h"

namespace llvm {
namespace codeview {

class MethodInfo;

class MethodListRecordBuilder : public ListRecordBuilder {
private:
  MethodListRecordBuilder(const MethodListRecordBuilder &) = delete;
  MethodListRecordBuilder &operator=(const MethodListRecordBuilder &) = delete;

public:
  MethodListRecordBuilder();

  void writeMethod(MemberAccess Access, MethodKind Kind, MethodOptions Options,
                   TypeIndex Type, int32_t VTableSlotOffset);
  void writeMethod(const MethodInfo &Method);
};
}
}

#endif
