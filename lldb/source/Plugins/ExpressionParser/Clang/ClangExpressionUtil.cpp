//===-- ClangExpressionUtil.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangExpressionUtil.h"

#include "lldb/Target/StackFrame.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/ValueObject/ValueObject.h"

namespace lldb_private {
namespace ClangExpressionUtil {
lldb::ValueObjectSP GetLambdaValueObject(StackFrame *frame) {
  assert(frame);

  if (auto this_val_sp = frame->FindVariable(ConstString("this")))
    if (this_val_sp->GetChildMemberWithName("this"))
      return this_val_sp;

  return nullptr;
}
} // namespace ClangExpressionUtil
} // namespace lldb_private
