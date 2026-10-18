//===-- AGXSelectionDAGInfo.cpp - AGX SelectionDAG info -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AGXSelectionDAGInfo.h"

#define GET_SDNODE_DESC
#include "AGXGenSDNodeInfo.inc"

using namespace llvm;

AGXSelectionDAGInfo::AGXSelectionDAGInfo()
    : SelectionDAGGenTargetInfo(AGXGenSDNodeInfo) {}
