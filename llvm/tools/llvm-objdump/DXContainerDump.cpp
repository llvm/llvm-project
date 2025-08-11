//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the DXContainer-specific dumper for llvm-objdump.
///
//===----------------------------------------------------------------------===//

#include "llvm-objdump.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/Object/DXContainer.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace llvm;
using namespace llvm::object;

static llvm::SmallString<4> maskToString(uint8_t Mask) {
  llvm::SmallString<4> Result("    ");
  if (Mask & 1)
    Result[0] = 'x';
  if (Mask & 2)
    Result[1] = 'y';
  if (Mask & 4)
    Result[2] = 'z';
  if (Mask & 8)
    Result[3] = 'w';
  return Result;
}

static void printColumnHeader(raw_ostream &OS, size_t Length) {
  for (size_t I = 0; I < Length; ++I)
    OS << "-";
}

static void printColumnHeaders(raw_ostream &OS, ArrayRef<size_t> Lengths) {
  for (auto L : Lengths) {
    printColumnHeader(OS, L);
    OS << " ";
  }
  OS << "\n";
}

static size_t digitsForNumber(size_t N) {
  return static_cast<size_t>(log10(static_cast<double>(N))) + 1;
}

namespace {
class DXContainerDumper : public objdump::Dumper {
  const DXContainerObjectFile &Obj;

public:
  DXContainerDumper(const DXContainerObjectFile &O)
      : objdump::Dumper(O), Obj(O) {}

  void printPrivateHeaders() override;
  void printSignature(const DirectX::Signature &S);
};

void DXContainerDumper::printSignature(const DirectX::Signature &S) {
  // DXC prints a table like this as part of the shader disassembly:
  //; Name                 Index   Mask Register SysValue  Format   Used
  //; -------------------- ----- ------ -------- -------- ------- ------
  //; NORMAL                   0   xyz         0     NONE   float   xyz
  //; TEXCOORD                 0   xy          1     NONE   float   xy

  // DXC's implementation doesn't scale columns entirely completely for the
  // provided input, so this implementation is a bit more complicated in
  // formatting logic to scale with the size of the printed text.

  // DXC gives names 21 characters for some unknown reason, I arbitrarily chose
  // to start at 24 so that we're not going shorter but are using a round
  // number.
  size_t LongestName = 24;
  size_t LongestSV = 10;
  size_t LongestIndex = strlen("Index");
  size_t LongestRegister = strlen("Register");
  size_t LongestFormat = strlen("Format");
  const size_t MaskWidth = 5;
  // Compute the column widths. Skip calculating the "Mask" and "Used" columns
  // since they both have widths of 4.
  for (auto El : S) {
    LongestName = std::max(LongestName, S.getName(El.NameOffset).size());
    LongestSV = std::max(
        LongestSV,
        enumToStringRef(El.SystemValue, dxbc::getD3DSystemValues()).size());
    LongestIndex = std::max(LongestIndex, digitsForNumber(El.Index));
    LongestRegister = std::max(LongestRegister, digitsForNumber(El.Register));
    LongestFormat = std::max(
        LongestFormat,
        enumToStringRef(El.CompType, dxbc::getSigComponentTypes()).size());
  }

  // Print Column headers.
  OS << "; ";
  OS << left_justify("Name", LongestName) << " ";
  OS << right_justify("Index", LongestIndex) << " ";
  OS << right_justify("Mask", MaskWidth) << " ";
  OS << right_justify("Register", LongestRegister) << " ";
  OS << right_justify("SysValue", LongestSV) << " ";
  OS << right_justify("Format", LongestFormat) << " ";
  OS << right_justify("Used", MaskWidth) << "\n";
  OS << "; ";
  printColumnHeaders(OS, {LongestName, LongestIndex, MaskWidth, LongestRegister,
                          LongestSV, LongestFormat, MaskWidth});

  for (auto El : S) {
    OS << "; " << left_justify(S.getName(El.NameOffset), LongestName) << " ";
    OS << right_justify(std::to_string(El.Index), LongestIndex) << " ";
    OS << right_justify(maskToString(El.Mask), MaskWidth) << " ";
    OS << right_justify(std::to_string(El.Register), LongestRegister) << " ";
    OS << right_justify(
              enumToStringRef(El.SystemValue, dxbc::getD3DSystemValues()),
              LongestSV)
       << " ";
    OS << right_justify(
              enumToStringRef(El.CompType, dxbc::getSigComponentTypes()),
              LongestFormat)
       << " ";
    OS << right_justify(maskToString(El.ExclusiveMask), MaskWidth) << "\n";
  }
}

void DXContainerDumper::printPrivateHeaders() {
  const DXContainer &C =
      cast<object::DXContainerObjectFile>(Obj).getDXContainer();

  if (!C.getInputSignature().isEmpty()) {
    OS << "; Input signature:\n;\n";
    printSignature(C.getInputSignature());
    OS << ";\n";
  }

  if (!C.getOutputSignature().isEmpty()) {
    OS << "; Output signature:\n;\n";
    printSignature(C.getOutputSignature());
    OS << ";\n";
  }

  if (!C.getPatchConstantSignature().isEmpty()) {
    OS << "; Patch Constant signature:\n;\n";
    printSignature(C.getPatchConstantSignature());
    OS << ";\n";
  }
}
} // namespace

std::unique_ptr<objdump::Dumper> llvm::objdump::createDXContainerDumper(
    const object::DXContainerObjectFile &Obj) {
  return std::make_unique<DXContainerDumper>(Obj);
}
