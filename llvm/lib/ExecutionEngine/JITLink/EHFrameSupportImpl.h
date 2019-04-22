//===------- EHFrameSupportImpl.h - JITLink eh-frame utils ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// EHFrame registration support for JITLink.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_EXECUTIONENGINE_JITLINK_EHFRAMESUPPORTIMPL_H
#define LLVM_LIB_EXECUTIONENGINE_JITLINK_EHFRAMESUPPORTIMPL_H

#include "llvm/ExecutionEngine/JITLink/EHFrameSupport.h"

#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/Support/BinaryStreamReader.h"

namespace llvm {
namespace jitlink {

/// A generic parser for eh-frame sections.
///
/// Adds atoms representing CIE and FDE entries, using the given FDE-to-CIE and
/// FDEToTarget relocation kinds.
class EHFrameParser {
public:
  EHFrameParser(AtomGraph &G, Section &EHFrameSection, StringRef EHFrameContent,
                JITTargetAddress EHFrameAddress, Edge::Kind FDEToCIERelocKind,
                Edge::Kind FDEToTargetRelocKind);
  Error atomize();

private:
  struct AugmentationInfo {
    bool AugmentationDataPresent = false;
    bool EHDataFieldPresent = false;
    uint8_t Fields[4] = {0x0, 0x0, 0x0, 0x0};
  };

  Expected<AugmentationInfo> parseAugmentationString();
  Expected<JITTargetAddress> readAbsolutePointer();
  Error processCIE();
  Error processFDE(JITTargetAddress CIEPointerAddress, uint32_t CIEPointer);

  AtomGraph &G;
  Section &EHFrameSection;
  StringRef EHFrameContent;
  JITTargetAddress EHFrameAddress;
  BinaryStreamReader EHFrameReader;
  DefinedAtom *CurRecordAtom = nullptr;
  bool LSDAFieldPresent = false;
  Edge::Kind FDEToCIERelocKind;
  Edge::Kind FDEToTargetRelocKind;
};

Error addEHFrame(AtomGraph &G, Section &EHFrameSection,
                 StringRef EHFrameContent, JITTargetAddress EHFrameAddress,
                 Edge::Kind FDEToCIERelocKind, Edge::Kind FDEToTargetRelocKind);

} // end namespace jitlink
} // end namespace llvm

#endif // LLVM_LIB_EXECUTIONENGINE_JITLINK_EHFRAMESUPPORTIMPL_H
