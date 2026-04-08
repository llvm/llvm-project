//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MSP430SelectionDAGInfo.h"

#define GET_SDNODE_DESC
#include "MSP430GenSDNodeInfo.inc"

using namespace llvm;

MSP430SelectionDAGInfo::MSP430SelectionDAGInfo()
    : SelectionDAGGenTargetInfo(MSP430GenSDNodeInfo) {}

MSP430SelectionDAGInfo::~MSP430SelectionDAGInfo() = default;
