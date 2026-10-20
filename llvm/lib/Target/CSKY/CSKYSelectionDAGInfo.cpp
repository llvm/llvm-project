//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CSKYSelectionDAGInfo.h"

#define GET_SDNODE_DESC
#include "CSKYGenSDNodeInfo.inc"

using namespace llvm;

CSKYSelectionDAGInfo::CSKYSelectionDAGInfo()
    : SelectionDAGGenTargetInfo(CSKYGenSDNodeInfo) {}

CSKYSelectionDAGInfo::~CSKYSelectionDAGInfo() = default;
