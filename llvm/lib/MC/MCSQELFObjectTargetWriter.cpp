//===-- MCSQELFObjectTargetWriter.cpp - SQELF Target Writer Subclass
//--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSQELFObjectWriter.h"

using namespace llvm;

MCSQELFObjectTargetWriter::MCSQELFObjectTargetWriter(bool Is64Bit_,
                                                     uint8_t OSABI_,
                                                     uint16_t EMachine_,
                                                     uint8_t ABIVersion_)
    : OSABI(OSABI_), ABIVersion(ABIVersion_), EMachine(EMachine_),
      Is64Bit(Is64Bit_) {}

MCSQELFObjectTargetWriter::~MCSQELFObjectTargetWriter() = default;
