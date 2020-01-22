//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Automatically generated file, do not edit!
//===----------------------------------------------------------------------===//


#ifndef LLVM_LIB_TARGET_HEXAGON_HEXAGONDEPARCH_H
#define LLVM_LIB_TARGET_HEXAGON_HEXAGONDEPARCH_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <map>

namespace llvm {
namespace Hexagon {
enum class ArchEnum { NoArch, Generic, V5, V55, V60, V62, V65, V66, V67 };

static constexpr unsigned ArchValsNumArray[] = {5, 55, 60, 62, 65, 66, 67};
static constexpr ArrayRef<unsigned> ArchValsNum(ArchValsNumArray);

static constexpr StringLiteral ArchValsTextArray[] = { "v5", "v55", "v60", "v62", "v65", "v66", "v67" };
static constexpr ArrayRef<StringLiteral> ArchValsText(ArchValsTextArray);

static constexpr StringLiteral CpuValsTextArray[] = { "hexagonv5", "hexagonv55", "hexagonv60", "hexagonv62", "hexagonv65", "hexagonv66", "hexagonv67", "hexagonv67t" };
static constexpr ArrayRef<StringLiteral> CpuValsText(CpuValsTextArray);

static constexpr StringLiteral CpuNickTextArray[] = { "v5", "v55", "v60", "v62", "v65", "v66", "v67", "v67t" };
static constexpr ArrayRef<StringLiteral> CpuNickText(CpuNickTextArray);

static const std::map<std::string, ArchEnum> CpuTable{
    {"generic", Hexagon::ArchEnum::V60},
    {"hexagonv5", Hexagon::ArchEnum::V5},
    {"hexagonv55", Hexagon::ArchEnum::V55},
    {"hexagonv60", Hexagon::ArchEnum::V60},
    {"hexagonv62", Hexagon::ArchEnum::V62},
    {"hexagonv65", Hexagon::ArchEnum::V65},
    {"hexagonv66", Hexagon::ArchEnum::V66},
    {"hexagonv67", Hexagon::ArchEnum::V67},
    {"hexagonv67t", Hexagon::ArchEnum::V67},
};
} // namespace Hexagon
} // namespace llvm;
#endif  // LLVM_LIB_TARGET_HEXAGON_HEXAGONDEPARCH_H
