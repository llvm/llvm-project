//===-- AssemblerUtils.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Assembler.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace exegesis {

class MachineFunctionGeneratorBaseTest : public ::testing::Test {
protected:
  MachineFunctionGeneratorBaseTest(const std::string &TT,
                                   const std::string &CpuName)
      : TT(TT), CpuName(CpuName),
        CanExecute(llvm::Triple(TT).getArch() ==
                   llvm::Triple(llvm::sys::getProcessTriple()).getArch()) {
    if (!CanExecute) {
      llvm::outs() << "Skipping execution, host:"
                   << llvm::sys::getProcessTriple() << ", target:" << TT
                   << "\n";
    }
  }

  template <class... Bs> inline void Check(llvm::MCInst MCInst, Bs... Bytes) {
    CheckWithSetup(nullptr, {}, MCInst, Bytes...);
  }

  template <class... Bs>
  inline void CheckWithSetup(const ExegesisTarget *ET,
                             llvm::ArrayRef<unsigned> RegsToDef,
                             llvm::MCInst MCInst, Bs... Bytes) {
    ExecutableFunction Function =
        (MCInst.getOpcode() == 0) ? assembleToFunction(ET, RegsToDef, {})
                                  : assembleToFunction(ET, RegsToDef, {MCInst});
    ASSERT_THAT(Function.getFunctionBytes().str(),
                testing::ElementsAre(Bytes...));
    if (CanExecute)
      Function();
  }

private:
  std::unique_ptr<llvm::LLVMTargetMachine> createTargetMachine() {
    std::string Error;
    const llvm::Target *TheTarget =
        llvm::TargetRegistry::lookupTarget(TT, Error);
    EXPECT_TRUE(TheTarget) << Error << " " << TT;
    const llvm::TargetOptions Options;
    llvm::TargetMachine *TM = TheTarget->createTargetMachine(
        TT, CpuName, "", Options, llvm::Reloc::Model::Static);
    EXPECT_TRUE(TM) << TT << " " << CpuName;
    return std::unique_ptr<llvm::LLVMTargetMachine>(
        static_cast<llvm::LLVMTargetMachine *>(TM));
  }

  ExecutableFunction
  assembleToFunction(const ExegesisTarget *ET,
                     llvm::ArrayRef<unsigned> RegsToDef,
                     llvm::ArrayRef<llvm::MCInst> Instructions) {
    llvm::SmallString<256> Buffer;
    llvm::raw_svector_ostream AsmStream(Buffer);
    assembleToStream(ET, createTargetMachine(), RegsToDef, Instructions,
                     AsmStream);
    return ExecutableFunction(createTargetMachine(),
                              getObjectFromBuffer(AsmStream.str()));
  }

  const std::string TT;
  const std::string CpuName;
  const bool CanExecute;
};

} // namespace exegesis
