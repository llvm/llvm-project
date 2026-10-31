//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Type.h"

using namespace lldb_private::clike;

// LLVM RTTI.
char Type::ID = 0;

bool Type::HasFields(llvm::function_ref<void(Type *)> complete) {
  Type *t = Desugar();
  complete(t);
  if (t->GetNumFields() != 0)
    return true;
  // A record without a definition (e.g., due to -flimit-debug-info) is
  // treated as having fields so that it isn't hidden as an empty base class.
  if (!t->IsComplete())
    return true;
  for (uint32_t i = 0, n = t->GetNumBaseClasses(); i < n; ++i) {
    const BaseClass *base = t->GetBaseClassAtIndex(i);
    if (base && base->GetType().Get().HasFields(complete))
      return true;
  }
  return false;
}
