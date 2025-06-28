//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SparcSelectionDAGInfo.h"

#define GET_SDNODE_DESC
#include "SparcGenSDNodeInfo.inc"

using namespace llvm;

SparcSelectionDAGInfo::SparcSelectionDAGInfo()
    : SelectionDAGGenTargetInfo(SparcGenSDNodeInfo) {}

SparcSelectionDAGInfo::~SparcSelectionDAGInfo() = default;
