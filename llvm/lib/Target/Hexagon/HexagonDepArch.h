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
#include "llvm/BinaryFormat/ELF.h"

#include <map>
#include <string>

namespace llvm {
namespace Hexagon {
enum class ArchEnum { NoArch, Generic, V5, V55, V60, V62, V65, V66, V67, V68 };

static constexpr unsigned ArchValsNumArray[] = {5, 55, 60, 62, 65, 66, 67, 68};
static constexpr ArrayRef<unsigned> ArchValsNum(ArchValsNumArray);

static constexpr StringLiteral ArchValsTextArray[] = { "v5", "v55", "v60", "v62", "v65", "v66", "v67", "v68" };
static constexpr ArrayRef<StringLiteral> ArchValsText(ArchValsTextArray);

static constexpr StringLiteral CpuValsTextArray[] = { "hexagonv5", "hexagonv55", "hexagonv60", "hexagonv62", "hexagonv65", "hexagonv66", "hexagonv67", "hexagonv67t", "hexagonv68" };
static constexpr ArrayRef<StringLiteral> CpuValsText(CpuValsTextArray);

static constexpr StringLiteral CpuNickTextArray[] = { "v5", "v55", "v60", "v62", "v65", "v66", "v67", "v67t", "v68" };
static constexpr ArrayRef<StringLiteral> CpuNickText(CpuNickTextArray);

static const std::map<std::string, ArchEnum> CpuTable{
  {"generic", Hexagon::ArchEnum::V5},
  {"hexagonv5", Hexagon::ArchEnum::V5},
  {"hexagonv55", Hexagon::ArchEnum::V55},
  {"hexagonv60", Hexagon::ArchEnum::V60},
  {"hexagonv62", Hexagon::ArchEnum::V62},
  {"hexagonv65", Hexagon::ArchEnum::V65},
  {"hexagonv66", Hexagon::ArchEnum::V66},
  {"hexagonv67", Hexagon::ArchEnum::V67},
  {"hexagonv67t", Hexagon::ArchEnum::V67},
  {"hexagonv68", Hexagon::ArchEnum::V68},
};

static const std::map<std::string, unsigned> ElfFlagsByCpuStr = {
  {"generic", llvm::ELF::EF_HEXAGON_MACH_V5},
  {"hexagonv5", llvm::ELF::EF_HEXAGON_MACH_V5},
  {"hexagonv55", llvm::ELF::EF_HEXAGON_MACH_V55},
  {"hexagonv60", llvm::ELF::EF_HEXAGON_MACH_V60},
  {"hexagonv62", llvm::ELF::EF_HEXAGON_MACH_V62},
  {"hexagonv65", llvm::ELF::EF_HEXAGON_MACH_V65},
  {"hexagonv66", llvm::ELF::EF_HEXAGON_MACH_V66},
  {"hexagonv67", llvm::ELF::EF_HEXAGON_MACH_V67},
  {"hexagonv67t", llvm::ELF::EF_HEXAGON_MACH_V67T},
  {"hexagonv68", llvm::ELF::EF_HEXAGON_MACH_V68},
};
static const std::map<unsigned, std::string> ElfArchByMachFlags = {
  {llvm::ELF::EF_HEXAGON_MACH_V5, "V5"},
  {llvm::ELF::EF_HEXAGON_MACH_V55, "V55"},
  {llvm::ELF::EF_HEXAGON_MACH_V60, "V60"},
  {llvm::ELF::EF_HEXAGON_MACH_V62, "V62"},
  {llvm::ELF::EF_HEXAGON_MACH_V65, "V65"},
  {llvm::ELF::EF_HEXAGON_MACH_V66, "V66"},
  {llvm::ELF::EF_HEXAGON_MACH_V67, "V67"},
  {llvm::ELF::EF_HEXAGON_MACH_V67T, "V67T"},
  {llvm::ELF::EF_HEXAGON_MACH_V68, "V68"},
};
static const std::map<unsigned, std::string> ElfCpuByMachFlags = {
  {llvm::ELF::EF_HEXAGON_MACH_V5, "hexagonv5"},
  {llvm::ELF::EF_HEXAGON_MACH_V55, "hexagonv55"},
  {llvm::ELF::EF_HEXAGON_MACH_V60, "hexagonv60"},
  {llvm::ELF::EF_HEXAGON_MACH_V62, "hexagonv62"},
  {llvm::ELF::EF_HEXAGON_MACH_V65, "hexagonv65"},
  {llvm::ELF::EF_HEXAGON_MACH_V66, "hexagonv66"},
  {llvm::ELF::EF_HEXAGON_MACH_V67, "hexagonv67"},
  {llvm::ELF::EF_HEXAGON_MACH_V67T, "hexagonv67t"},
  {llvm::ELF::EF_HEXAGON_MACH_V68, "hexagonv68"},
};

} // namespace Hexagon
} // namespace llvm;

#endif // LLVM_LIB_TARGET_HEXAGON_HEXAGONDEPARCH_H
