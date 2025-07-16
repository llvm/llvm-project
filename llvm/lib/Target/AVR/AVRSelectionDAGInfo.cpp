//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AVRSelectionDAGInfo.h"

#define GET_SDNODE_DESC
#include "AVRGenSDNodeInfo.inc"

using namespace llvm;

AVRSelectionDAGInfo::AVRSelectionDAGInfo()
    : SelectionDAGGenTargetInfo(AVRGenSDNodeInfo) {}

AVRSelectionDAGInfo::~AVRSelectionDAGInfo() = default;
