//===-- DILAST.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/ValueObject/DILAST.h"
#include "llvm/Support/ErrorHandling.h"

namespace lldb_private::dil {

llvm::Expected<lldb::ValueObjectSP> ErrorNode::Accept(Visitor *v) const {
  llvm_unreachable("Attempting to Visit a DIL ErrorNode.");
}

llvm::Expected<lldb::ValueObjectSP> IdentifierNode::Accept(Visitor *v) const {
  return v->Visit(this);
}

llvm::Expected<lldb::ValueObjectSP> MemberOfNode::Accept(Visitor *v) const {
  return v->Visit(this);
}

llvm::Expected<lldb::ValueObjectSP> UnaryOpNode::Accept(Visitor *v) const {
  return v->Visit(this);
}

} // namespace lldb_private::dil
