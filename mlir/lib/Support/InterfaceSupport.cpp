//===- InterfaceSupport.cpp - MLIR Interface Support Classes --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines several support classes for defining interfaces.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/InterfaceSupport.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "interfaces"

using namespace mlir;

void detail::InterfaceMap::insert(
    ArrayRef<std::pair<TypeID, void *>> elements) {
  // Insert directly into the right position to keep the interfaces sorted.
  for (auto &element : elements) {
    TypeID id = element.first;
    auto *it = llvm::lower_bound(interfaces, id, [](const auto &it, TypeID id) {
      return compare(it.first, id);
    });
    if (it != interfaces.end() && it->first == id) {
      LLVM_DEBUG(llvm::dbgs() << "Ignoring repeated interface registration");
      continue;
    }
    interfaces.insert(it, element);
  }
}
