//===-- AssemblerUtils.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_TOOLS_LLVMEXEGESIS_ASSEMBLERUTILS_H
#define LLVM_UNITTESTS_TOOLS_LLVMEXEGESIS_ASSEMBLERUTILS_H

#include "Assembler.h"
#include "BenchmarkRunner.h"
#include "Target.h"
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

namespace llvm {
namespace exegesis {

class MachineFunctionGeneratorBaseTest : public ::testing::Test {
protected:
  MachineFunctionGeneratorBaseTest(const std::string &TT,
                                   const std::string &CpuName)
      : TT(TT), CpuName(CpuName),
        CanExecute(llvm::Triple(TT).getArch() ==
                   llvm::Triple(llvm::sys::getProcessTriple()).getArch()),
        ET(ExegesisTarget::lookup(llvm::Triple(TT))) {
    assert(ET);
    if (!CanExecute) {
      llvm::outs() << "Skipping execution, host:"
                   << llvm::sys::getProcessTriple() << ", target:" << TT
                   << "\n";
    }
  }

  template <class... Bs>
  inline void Check(llvm::ArrayRef<RegisterValue> RegisterInitialValues,
                    llvm::MCInst MCInst, Bs... Bytes) {
    ExecutableFunction Function =
        (MCInst.getOpcode() == 0)
            ? assembleToFunction(RegisterInitialValues, {})
            : assembleToFunction(RegisterInitialValues, {MCInst});
    ASSERT_THAT(Function.getFunctionBytes().str(),
                testing::ElementsAre(Bytes...));
    if (CanExecute) {
      BenchmarkRunner::ScratchSpace Scratch;
      Function(Scratch.ptr());
    }
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
  assembleToFunction(llvm::ArrayRef<RegisterValue> RegisterInitialValues,
                     llvm::ArrayRef<llvm::MCInst> Instructions) {
    llvm::SmallString<256> Buffer;
    llvm::raw_svector_ostream AsmStream(Buffer);
    assembleToStream(*ET, createTargetMachine(), /*LiveIns=*/{},
                     RegisterInitialValues, Instructions, AsmStream);
    return ExecutableFunction(createTargetMachine(),
                              getObjectFromBuffer(AsmStream.str()));
  }

  const std::string TT;
  const std::string CpuName;
  const bool CanExecute;
  const ExegesisTarget *const ET;
};

} // namespace exegesis
} // namespace llvm

#endif
