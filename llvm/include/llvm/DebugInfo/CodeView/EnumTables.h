//===- EnumTables.h - Enum to string conversion tables ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_ENUMTABLES_H
#define LLVM_DEBUGINFO_CODEVIEW_ENUMTABLES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/Support/Compiler.h"
#include <cstdint>

namespace llvm {
template <typename T> struct EnumEntry;
namespace codeview {

LLVM_ABI ArrayRef<EnumEntry<SymbolKind>> getSymbolTypeNames();
LLVM_ABI ArrayRef<EnumEntry<TypeLeafKind>> getTypeLeafNames();
LLVM_ABI ArrayRef<EnumEntry<uint16_t>> getRegisterNames(CPUType Cpu);
LLVM_ABI ArrayRef<EnumEntry<uint32_t>> getPublicSymFlagNames();
LLVM_ABI ArrayRef<EnumEntry<uint8_t>> getProcSymFlagNames();
LLVM_ABI ArrayRef<EnumEntry<uint16_t>> getLocalFlagNames();
LLVM_ABI ArrayRef<EnumEntry<uint8_t>> getFrameCookieKindNames();
LLVM_ABI ArrayRef<EnumEntry<SourceLanguage>> getSourceLanguageNames();
LLVM_ABI ArrayRef<EnumEntry<uint32_t>> getCompileSym2FlagNames();
LLVM_ABI ArrayRef<EnumEntry<uint32_t>> getCompileSym3FlagNames();
LLVM_ABI ArrayRef<EnumEntry<uint32_t>> getFileChecksumNames();
LLVM_ABI ArrayRef<EnumEntry<unsigned>> getCPUTypeNames();
LLVM_ABI ArrayRef<EnumEntry<uint32_t>> getFrameProcSymFlagNames();
LLVM_ABI ArrayRef<EnumEntry<uint16_t>> getExportSymFlagNames();
LLVM_ABI ArrayRef<EnumEntry<uint32_t>> getModuleSubstreamKindNames();
LLVM_ABI ArrayRef<EnumEntry<uint8_t>> getThunkOrdinalNames();
LLVM_ABI ArrayRef<EnumEntry<uint16_t>> getTrampolineNames();
LLVM_ABI ArrayRef<EnumEntry<COFF::SectionCharacteristics>>
getImageSectionCharacteristicNames();
LLVM_ABI ArrayRef<EnumEntry<uint16_t>> getClassOptionNames();
LLVM_ABI ArrayRef<EnumEntry<uint8_t>> getMemberAccessNames();
LLVM_ABI ArrayRef<EnumEntry<uint16_t>> getMethodOptionNames();
LLVM_ABI ArrayRef<EnumEntry<uint16_t>> getMemberKindNames();
LLVM_ABI ArrayRef<EnumEntry<uint8_t>> getPtrKindNames();
LLVM_ABI ArrayRef<EnumEntry<uint8_t>> getPtrModeNames();
LLVM_ABI ArrayRef<EnumEntry<uint16_t>> getPtrMemberRepNames();
LLVM_ABI ArrayRef<EnumEntry<uint16_t>> getTypeModifierNames();
LLVM_ABI ArrayRef<EnumEntry<uint8_t>> getCallingConventions();
LLVM_ABI ArrayRef<EnumEntry<uint8_t>> getFunctionOptionEnum();
LLVM_ABI ArrayRef<EnumEntry<uint16_t>> getLabelTypeEnum();
LLVM_ABI ArrayRef<EnumEntry<uint16_t>> getJumpTableEntrySizeNames();

} // end namespace codeview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_ENUMTABLES_H
