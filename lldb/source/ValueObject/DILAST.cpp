//===-- DILAST.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/ValueObject/DILAST.h"

namespace lldb_private {

namespace dil {

void ErrorNode::Accept(Visitor *v) const { v->Visit(this); }

void IdentifierNode::Accept(Visitor *v) const { v->Visit(this); }

} // namespace dil

} // namespace lldb_private
