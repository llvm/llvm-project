//===- EnumTables.h - Enum to string conversion tables ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_ENUMTABLES_H
#define LLVM_DEBUGINFO_CODEVIEW_ENUMTABLES_H

#include "llvm/BinaryFormat/COFF.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/Support/Compiler.h"
#include <cstdint>

namespace llvm {
template <typename, unsigned> class EnumStrings;
namespace codeview {

LLVM_ABI EnumStrings<SymbolKind, 1> getSymbolTypeNames();
LLVM_ABI EnumStrings<TypeLeafKind, 1> getTypeLeafNames();
LLVM_ABI EnumStrings<uint16_t, 1> getRegisterNames(CPUType Cpu);
LLVM_ABI EnumStrings<uint32_t, 1> getPublicSymFlagNames();
LLVM_ABI EnumStrings<uint8_t, 1> getProcSymFlagNames();
LLVM_ABI EnumStrings<uint16_t, 1> getLocalFlagNames();
LLVM_ABI EnumStrings<uint8_t, 1> getFrameCookieKindNames();
LLVM_ABI EnumStrings<SourceLanguage, 1> getSourceLanguageNames();
LLVM_ABI EnumStrings<uint32_t, 1> getCompileSym2FlagNames();
LLVM_ABI EnumStrings<uint32_t, 1> getCompileSym3FlagNames();
LLVM_ABI EnumStrings<uint32_t, 1> getFileChecksumNames();
LLVM_ABI EnumStrings<unsigned, 1> getCPUTypeNames();
LLVM_ABI EnumStrings<uint32_t, 1> getFrameProcSymFlagNames();
LLVM_ABI EnumStrings<uint16_t, 1> getExportSymFlagNames();
LLVM_ABI EnumStrings<uint32_t, 1> getModuleSubstreamKindNames();
LLVM_ABI EnumStrings<uint8_t, 1> getThunkOrdinalNames();
LLVM_ABI EnumStrings<uint16_t, 1> getTrampolineNames();
LLVM_ABI EnumStrings<COFF::SectionCharacteristics, 1>
getImageSectionCharacteristicNames();
LLVM_ABI EnumStrings<uint16_t, 1> getClassOptionNames();
LLVM_ABI EnumStrings<uint8_t, 1> getMemberAccessNames();
LLVM_ABI EnumStrings<uint16_t, 1> getMethodOptionNames();
LLVM_ABI EnumStrings<uint16_t, 1> getMemberKindNames();
LLVM_ABI EnumStrings<uint8_t, 1> getPtrKindNames();
LLVM_ABI EnumStrings<uint8_t, 1> getPtrModeNames();
LLVM_ABI EnumStrings<uint16_t, 1> getPtrMemberRepNames();
LLVM_ABI EnumStrings<uint16_t, 1> getTypeModifierNames();
LLVM_ABI EnumStrings<uint8_t, 1> getCallingConventions();
LLVM_ABI EnumStrings<uint8_t, 1> getFunctionOptionEnum();
LLVM_ABI EnumStrings<uint16_t, 1> getLabelTypeEnum();
LLVM_ABI EnumStrings<uint16_t, 1> getJumpTableEntrySizeNames();

} // end namespace codeview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_ENUMTABLES_H
