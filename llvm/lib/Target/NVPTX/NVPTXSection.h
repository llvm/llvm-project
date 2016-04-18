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

#ifndef LLVM_LIB_TARGET_NVPTX_NVPTXSECTION_H
#define LLVM_LIB_TARGET_NVPTX_NVPTXSECTION_H

#include "llvm/IR/GlobalVariable.h"
#include "llvm/MC/MCSection.h"

namespace llvm {
/// Represents a section in PTX PTX does not have sections. We create this class
/// in order to use the ASMPrint interface.
///
class NVPTXSection final : public MCSection {
  virtual void anchor();
public:
  NVPTXSection(SectionVariant V, SectionKind K) : MCSection(V, K, nullptr) {}
  ~NVPTXSection() {}

  /// Override this as NVPTX has its own way of printing switching
  /// to a section.
  void PrintSwitchToSection(const MCAsmInfo &MAI,
                            raw_ostream &OS,
                            const MCExpr *Subsection) const override {}

  /// Base address of PTX sections is zero.
  bool UseCodeAlign() const override { return false; }
  bool isVirtualSection() const override { return false; }
};

} // end namespace llvm

#endif
