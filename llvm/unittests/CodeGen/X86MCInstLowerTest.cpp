//===- llvm/unittest/CodeGen/X86MCInstLowerTest.cpp
//-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestAsmPrinter.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

namespace llvm {

class X86MCInstLowerTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    InitializeAllTargetMCs();
    InitializeAllTargetInfos();
    InitializeAllTargets();
    InitializeAllAsmPrinters();
  }

  // Function to setup codegen pipeline and returns the AsmPrinter.
  AsmPrinter *addPassesToEmitFile(llvm::legacy::PassManagerBase &PM,
                                  llvm::raw_pwrite_stream &Out,
                                  llvm::CodeGenFileType FileType,
                                  llvm::MachineModuleInfoWrapperPass *MMIWP) {
    TargetPassConfig *PassConfig = TM->createPassConfig(PM);

    PassConfig->setDisableVerify(true);
    PM.add(PassConfig);
    PM.add(MMIWP);

    if (PassConfig->addISelPasses())
      return nullptr;

    PassConfig->addMachinePasses();
    PassConfig->setInitialized();

    MC.reset(new MCContext(TM->getTargetTriple(), TM->getMCAsmInfo(),
                           TM->getMCRegisterInfo(), TM->getMCSubtargetInfo()));
    MC->setObjectFileInfo(TM->getObjFileLowering());
    TM->getObjFileLowering()->Initialize(*MC, *TM);
    MC->setObjectFileInfo(TM->getObjFileLowering());

    // Use a new MCContext for AsmPrinter for testing.
    // AsmPrinter.OutContext will be different from
    // MachineFunction's MCContext in MMIWP.
    Expected<std::unique_ptr<MCStreamer>> MCStreamerOrErr =
        TM->createMCStreamer(Out, nullptr, FileType, *MC);

    if (auto Err = MCStreamerOrErr.takeError())
      return nullptr;

    AsmPrinter *Printer =
        TM->getTarget().createAsmPrinter(*TM, std::move(*MCStreamerOrErr));

    if (!Printer)
      return nullptr;

    PM.add(Printer);

    return Printer;
  }

  void SetUp() override {
    // Module to compile.
    const char *FooStr = R""""(
        @G = external global i32

        define i32 @foo() {
          %1 = load i32, ptr @G; load the global variable
          %2 = call i32 @f()
          %3 = mul i32 %1, %2
          ret i32 %3
        }

        declare i32 @f() #0
      )"""";
    StringRef AssemblyF(FooStr);

    // Get target triple for X86_64
    Triple TargetTriple("x86_64--");
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);
    // Skip the test if target is not built.
    if (!T)
      GTEST_SKIP();

    // Get TargetMachine.
    // Use Reloc::Model::PIC_ and CodeModel::Model::Large
    // to get GOT during codegen as MO_ExternalSymbol.
    TargetOptions Options;
    TM = std::unique_ptr<TargetMachine>(T->createTargetMachine(
        TargetTriple, "", "", Options, Reloc::Model::PIC_,
        CodeModel::Model::Large, CodeGenOptLevel::Default));
    if (!TM)
      GTEST_SKIP();

    SMDiagnostic SMError;

    // Parse the module.
    M = parseAssemblyString(AssemblyF, SMError, Context);
    if (!M)
      report_fatal_error(SMError.getMessage());
    M->setDataLayout(TM->createDataLayout());

    // Get llvm::Function from M
    Foo = M->getFunction("foo");
    if (!Foo)
      report_fatal_error("foo?");

    // Prepare the MCContext for codegen M.
    // MachineFunction for Foo will have this MCContext.
    MCFoo.reset(new MCContext(TargetTriple, TM->getMCAsmInfo(),
                              TM->getMCRegisterInfo(),
                              TM->getMCSubtargetInfo()));
    MCFoo->setObjectFileInfo(TM->getObjFileLowering());
    TM->getObjFileLowering()->Initialize(*MCFoo, *TM);
    MCFoo->setObjectFileInfo(TM->getObjFileLowering());
  }

  LLVMContext Context;
  std::unique_ptr<TargetMachine> TM;
  std::unique_ptr<Module> M;

  std::unique_ptr<MCContext> MC;
  std::unique_ptr<MCContext> MCFoo;

  Function *Foo;
  std::unique_ptr<MachineFunction> MFFoo;
};

TEST_F(X86MCInstLowerTest, moExternalSymbol_MCSYMBOL) {

  MachineModuleInfoWrapperPass *MMIWP =
      new MachineModuleInfoWrapperPass(TM.get(), &*MCFoo);

  SmallString<1024> Buf;
  llvm::raw_svector_ostream OS(Buf);
  legacy::PassManager PassMgrF;

  AsmPrinter *Printer =
      addPassesToEmitFile(PassMgrF, OS, CodeGenFileType::AssemblyFile, MMIWP);
  PassMgrF.run(*M);

  // Check GOT MCSymbol is from Printer.OutContext.
  MCSymbol *GOTPrinterPtr =
      Printer->OutContext.lookupSymbol("_GLOBAL_OFFSET_TABLE_");

  // Check GOT MCSymbol is NOT from MachineFunction's MCContext.
  MCSymbol *GOTMFCtxPtr =
      MMIWP->getMMI().getMachineFunction(*Foo)->getContext().lookupSymbol(
          "_GLOBAL_OFFSET_TABLE_");

  EXPECT_NE(GOTPrinterPtr, nullptr);
  EXPECT_EQ(GOTMFCtxPtr, nullptr);
}

} // end namespace llvm
