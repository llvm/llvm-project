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

/// A generic binary parser for eh-frame sections.
///
/// Adds blocks and symbols representing CIE and FDE entries to a JITLink graph.
///
/// This parser assumes that the user has already verified that the EH-frame's
/// address range does not overlap any other section/symbol, so that generated
/// CIE/FDE records do not overlap other sections/symbols.
class EHFrameBinaryParser {
public:
  EHFrameBinaryParser(JITTargetAddress EHFrameAddress, StringRef EHFrameContent,
                      unsigned PointerSize, support::endianness Endianness);
  virtual ~EHFrameBinaryParser() {}

  Error addToGraph();

private:
  virtual void anchor();
  virtual Symbol *getSymbolAtAddress(JITTargetAddress Addr) = 0;
  virtual Symbol &createCIERecord(JITTargetAddress RecordAddr,
                                  StringRef RecordContent) = 0;
  virtual Expected<Symbol &>
  createFDERecord(JITTargetAddress RecordAddr, StringRef RecordContent,
                  Symbol &CIE, size_t CIEOffset, Symbol &Func,
                  size_t FuncOffset, Symbol *LSDA, size_t LSDAOffset) = 0;

  struct AugmentationInfo {
    bool AugmentationDataPresent = false;
    bool EHDataFieldPresent = false;
    uint8_t Fields[4] = {0x0, 0x0, 0x0, 0x0};
  };

  Expected<AugmentationInfo> parseAugmentationString();
  Expected<JITTargetAddress> readAbsolutePointer();
  Error processCIE(size_t RecordOffset, size_t RecordLength);
  Error processFDE(size_t RecordOffset, size_t RecordLength,
                   JITTargetAddress CIEPointerOffset, uint32_t CIEPointer);

  struct CIEInformation {
    CIEInformation() = default;
    CIEInformation(Symbol &CIESymbol) : CIESymbol(&CIESymbol) {}
    Symbol *CIESymbol = nullptr;
    bool FDEsHaveLSDAField = false;
  };

  JITTargetAddress EHFrameAddress;
  StringRef EHFrameContent;
  unsigned PointerSize;
  BinaryStreamReader EHFrameReader;
  DenseMap<JITTargetAddress, CIEInformation> CIEInfos;
};

} // end namespace jitlink
} // end namespace llvm

#endif // LLVM_LIB_EXECUTIONENGINE_JITLINK_EHFRAMESUPPORTIMPL_H
