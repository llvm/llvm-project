//===- CASNodeSchema.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/CASNodeSchema.h"

using namespace llvm;
using namespace llvm::cas;

char NodeSchema::ID = 0;
void NodeSchema::anchor() {}

NodeSchema *SchemaPool::getSchemaForRoot(cas::ObjectHandle Node) const {
  for (auto &Schema : Schemas)
    if (Schema->isRootNode(Node))
      return Schema.get();
  return nullptr;
}
