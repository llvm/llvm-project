//===---- MCATestBase.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Test fixture common to all MCA tests.
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_TOOLS_LLVMMCA_MCATESTBASE_H
#define LLVM_UNITTESTS_TOOLS_LLVMMCA_MCATESTBASE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/MCA/Context.h"

#include "gtest/gtest.h"

namespace llvm {
namespace json {
class Object;
} // end namespace json

namespace mca {
class View;

class MCATestBase : public ::testing::Test {
protected:
  // Note: Subclass ctors are expected to perform target-specific
  // initializations.
  MCATestBase(StringRef TripleStr, StringRef CPUName, StringRef MAttr = "")
      : TheTriple(TripleStr), CPUName(CPUName), MAttr(MAttr) {}

  /// Factory function to create a Target.
  virtual const Target *getLLVMTarget() const;

  /// Factory function to create a MCTargetOptions instance. Returns an
  /// empty one by default.
  virtual MCTargetOptions getMCTargetOptions() { return MCTargetOptions(); }

  const Target *TheTarget;
  const Triple TheTriple;
  StringRef CPUName;
  StringRef MAttr;

  // MC components.
  std::unique_ptr<MCSubtargetInfo> STI;
  std::unique_ptr<MCRegisterInfo> MRI;
  std::unique_ptr<MCAsmInfo> MAI;
  std::unique_ptr<MCObjectFileInfo> MOFI;
  std::unique_ptr<MCContext> Ctx;
  std::unique_ptr<MCInstrInfo> MCII;
  std::unique_ptr<MCInstrAnalysis> MCIA;
  std::unique_ptr<MCInstPrinter> IP;

  static mca::PipelineOptions getDefaultPipelineOptions();

  void SetUp() override;

  /// Utility function to run MCA with (nearly) the same configuration as the
  /// `llvm-mca` tool to verify result correctness.
  /// This function only displays on SummaryView by default.
  virtual Error runBaselineMCA(json::Object &Result, ArrayRef<MCInst> Insts,
                               ArrayRef<mca::View *> Views = std::nullopt,
                               const mca::PipelineOptions *PO = nullptr);
};

} // end namespace mca
} // end namespace llvm
#endif
