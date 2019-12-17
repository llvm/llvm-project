//===- GIMatchDagEdge.cpp - An edge describing a def/use lookup -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GIMatchDagEdge.h"
#include "GIMatchDagInstr.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

LLVM_DUMP_METHOD void GIMatchDagEdge::print(raw_ostream &OS) const {
  OS << getFromMI()->getName() << "[" << getFromMO()->getName() << "] --["
     << Name << "]--> " << getToMI()->getName() << "[" << getToMO()->getName()
     << "]";
}

void GIMatchDagEdge::reverse() {
  std::swap(FromMI, ToMI);
  std::swap(FromMO, ToMO);
}

