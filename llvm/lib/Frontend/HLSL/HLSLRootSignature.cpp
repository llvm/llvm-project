//===- HLSLRootSignature.cpp - HLSL Root Signature helper objects ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helpers for working with HLSL Root Signatures.
///
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/HLSL/HLSLRootSignature.h"
#include "llvm/ADT/bit.h"

namespace llvm {
namespace hlsl {
namespace rootsig {

static void dumpRegType(raw_ostream& OS, RegisterType Type) {
  switch (Type) {
  case RegisterType::BReg:
    OS << "b";
    break;
  case RegisterType::TReg:
    OS << "t";
    break;
  case RegisterType::UReg:
    OS << "u";
    break;
  case RegisterType::SReg:
    OS << "s";
    break;
  }
}

void Register::dump(raw_ostream &OS) const {
  dumpRegType(OS, ViewType);
  OS << Number;
}

static void dumpClauseType(raw_ostream& OS, ClauseType Type) {
  switch (Type) {
  case ClauseType::CBuffer:
    OS << "CBV";
    break;
  case ClauseType::SRV:
    OS << "SRV";
    break;
  case ClauseType::UAV:
    OS << "UAV";
    break;
  case ClauseType::Sampler:
    OS << "Sampler";
    break;
  }
}

static void dumpDescriptorRangeFlag(raw_ostream &OS, unsigned Bit) {
  switch (static_cast<DescriptorRangeFlags>(Bit)) {
    case DescriptorRangeFlags::DescriptorsVolatile:
      OS << "DescriptorsVolatile";
      break;
    case DescriptorRangeFlags::DataVolatile:
      OS << "DataVolatile";
      break;
    case DescriptorRangeFlags::DataStaticWhileSetAtExecute:
      OS << "DataStaticWhileSetAtExecute";
      break;
    case DescriptorRangeFlags::DataStatic:
      OS << "DataStatic";
      break;
    case DescriptorRangeFlags::DescriptorsStaticKeepingBufferBoundsChecks:
      OS << "DescriptorsStaticKeepingBufferBoundsChecks";
      break;
    default:
      OS << "invalid: " << Bit;
      break;
  }
}

static void dumpDescriptorRangeFlags(raw_ostream &OS, DescriptorRangeFlags Flags) {
  bool FlagSet = false;
  unsigned Remaining = llvm::to_underlying(Flags);
  while (Remaining) {
    unsigned Bit = 1u << llvm::countr_zero(Remaining);
    if (Remaining & Bit) {
      if (FlagSet)
        OS << " | ";
      dumpDescriptorRangeFlag(OS, Bit);
      FlagSet = true;
    }
    Remaining &= ~Bit;
  }
  if (!FlagSet)
    OS << "None";
}

void DescriptorTableClause::dump(raw_ostream &OS) const {
  dumpClauseType(OS, Type);
  OS << "(";
  Reg.dump(OS);
  OS << ", numDescriptors = " << NumDescriptors;
  OS << ", space = " << Space;
  OS << ", offset = ";
  if (Offset == DescriptorTableOffsetAppend)
    OS << "DESCRIPTOR_TABLE_OFFSET_APPEND";
  else
    OS << Offset;
  OS << ", flags = ";
  dumpDescriptorRangeFlags(OS, Flags);
  OS << ")";
}

} // namespace rootsig
} // namespace hlsl
} // namespace llvm
