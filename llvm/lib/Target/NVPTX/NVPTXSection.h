//===- NVPTXSection.h - NVPTX-specific section representation -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the NVPTXSection class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_NVPTXSECTION_H
#define LLVM_NVPTXSECTION_H

#include "llvm/MC/MCSection.h"
#include "llvm/GlobalVariable.h"
#include <vector>

namespace llvm {
/// NVPTXSection - Represents a section in PTX
/// PTX does not have sections. We create this class in order to use
/// the ASMPrint interface.
///
class NVPTXSection : public MCSection {

public:
  NVPTXSection(SectionVariant V, SectionKind K) : MCSection(V, K) {}
  ~NVPTXSection() {}

  /// Override this as NVPTX has its own way of printing switching
  /// to a section.
  virtual void PrintSwitchToSection(const MCAsmInfo &MAI,
                                    raw_ostream &OS) const {}

  /// Base address of PTX sections is zero.
  virtual bool isBaseAddressKnownZero() const { return true; }
  virtual bool UseCodeAlign() const { return false; }
  virtual bool isVirtualSection() const { return false; }
};

} // end namespace llvm

#endif
