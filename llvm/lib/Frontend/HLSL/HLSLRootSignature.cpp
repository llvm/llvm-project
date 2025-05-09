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

static raw_ostream &operator<<(raw_ostream &OS, const Register &Reg) {
  switch (Reg.ViewType) {
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
  OS << Reg.Number;
  return OS;
}

static raw_ostream &operator<<(raw_ostream &OS,
                               const ShaderVisibility &Visibility) {
  switch (Visibility) {
  case ShaderVisibility::All:
    OS << "All";
    break;
  case ShaderVisibility::Vertex:
    OS << "Vertex";
    break;
  case ShaderVisibility::Hull:
    OS << "Hull";
    break;
  case ShaderVisibility::Domain:
    OS << "Domain";
    break;
  case ShaderVisibility::Geometry:
    OS << "Geometry";
    break;
  case ShaderVisibility::Pixel:
    OS << "Pixel";
    break;
  case ShaderVisibility::Amplification:
    OS << "Amplification";
    break;
  case ShaderVisibility::Mesh:
    OS << "Mesh";
    break;
  }

  return OS;
}

void DescriptorTable::dump(raw_ostream &OS) const {
  OS << "DescriptorTable(numClauses = " << NumClauses
     << ", visibility = " << Visibility << ")";
}

static raw_ostream &operator<<(raw_ostream &OS, const ClauseType &Type) {
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

  return OS;
}

static raw_ostream &operator<<(raw_ostream &OS,
                               const DescriptorRangeFlags &Flags) {
  bool FlagSet = false;
  unsigned Remaining = llvm::to_underlying(Flags);
  while (Remaining) {
    unsigned Bit = 1u << llvm::countr_zero(Remaining);
    if (Remaining & Bit) {
      if (FlagSet)
        OS << " | ";

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

      FlagSet = true;
    }
    Remaining &= ~Bit;
  }

  if (!FlagSet)
    OS << "None";

  return OS;
}

void DescriptorTableClause::dump(raw_ostream &OS) const {
  OS << Type << "(" << Reg << ", numDescriptors = " << NumDescriptors
     << ", space = " << Space << ", offset = ";
  if (Offset == DescriptorTableOffsetAppend)
    OS << "DescriptorTableOffsetAppend";
  else
    OS << Offset;
  OS << ", flags = " << Flags << ")";
}

// Helper callable so that we can use the overloaded notation of std::visit
namespace {
struct ElementDumper {
  raw_ostream &OS;
  template <typename T> void operator()(const T &Element) const {
    Element.dump(OS);
  }
};
} // namespace

void dumpRootElements(raw_ostream &OS, ArrayRef<RootElement> Elements) {
  ElementDumper Dumper{OS};
  OS << "RootElements{";
  bool First = true;
  for (const RootElement &Element : Elements) {
    if (!First)
      OS << ",";
    OS << " ";
    First = false;
    std::visit(Dumper, Element);
  }
  OS << "}";
}

} // namespace rootsig
} // namespace hlsl
} // namespace llvm
