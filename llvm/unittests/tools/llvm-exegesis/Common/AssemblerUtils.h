//===-- AssemblerUtils.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace llvm {
namespace exegesis {

class MachineFunctionGeneratorBaseTest : public ::testing::Test {
protected:
  MachineFunctionGeneratorBaseTest(const std::string &TT,
                                   const std::string &CpuName)
      : TT(TT), CpuName(CpuName),
        CanExecute(Triple(TT).getArch() ==
                   Triple(sys::getProcessTriple()).getArch()),
        ET(ExegesisTarget::lookup(Triple(TT))) {
    assert(ET);
    if (!CanExecute) {
      outs() << "Skipping execution, host:" << sys::getProcessTriple()
             << ", target:" << TT << "\n";
    }
  }

  template <class... Bs>
  inline void Check(ArrayRef<RegisterValue> RegisterInitialValues, MCInst Inst,
                    Bs... Bytes) {
    ExecutableFunction Function =
        (Inst.getOpcode() == 0)
            ? assembleToFunction(RegisterInitialValues, [](FunctionFiller &) {})
            : assembleToFunction(RegisterInitialValues,
                                 [Inst](FunctionFiller &Filler) {
                                   Filler.getEntry().addInstruction(Inst);
                                 });
    ASSERT_THAT(Function.getFunctionBytes().str(),
                testing::ElementsAre(Bytes...));
    if (CanExecute) {
      BenchmarkRunner::ScratchSpace Scratch;
      Function(Scratch.ptr());
    }
  }

private:
  std::unique_ptr<TargetMachine> createTargetMachine() {
    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);
    EXPECT_TRUE(TheTarget) << Error << " " << TT;
    const TargetOptions Options;
    TargetMachine *TM = TheTarget->createTargetMachine(TT, CpuName, "", Options,
                                                       Reloc::Model::Static);
    EXPECT_TRUE(TM) << TT << " " << CpuName;
    return std::unique_ptr<TargetMachine>(TM);
  }

  ExecutableFunction
  assembleToFunction(ArrayRef<RegisterValue> RegisterInitialValues,
                     FillFunction Fill) {
    SmallString<256> Buffer;
    raw_svector_ostream AsmStream(Buffer);
    BenchmarkKey Key;
    Key.RegisterInitialValues = RegisterInitialValues;
    EXPECT_FALSE(assembleToStream(*ET, createTargetMachine(), /*LiveIns=*/{},
                                  Fill, AsmStream, Key, false));
    Expected<ExecutableFunction> ExecFunc = ExecutableFunction::create(
        createTargetMachine(), getObjectFromBuffer(AsmStream.str()));

    // We can't use ASSERT_THAT_EXPECTED here as it doesn't work inside of
    // non-void functions.
    EXPECT_TRUE(detail::TakeExpected(ExecFunc).Success());
    return std::move(*ExecFunc);
  }

  const std::string TT;
  const std::string CpuName;
  const bool CanExecute;
  const ExegesisTarget *const ET;
};

} // namespace exegesis
} // namespace llvm

#endif
