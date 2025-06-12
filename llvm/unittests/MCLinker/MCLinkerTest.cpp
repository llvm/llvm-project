//===- llvm/unittest/Linker/LinkModulesTest.cpp - IRBuilder tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/MCLinker/MCLinker.h"
#include "llvm/MCLinker/MCPipeline.h"
#include "llvm/ModuleSplitter/ModuleSplitter.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class MCLinkerTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    InitializeAllTargetMCs();
    InitializeAllTargetInfos();
    InitializeAllTargets();
    InitializeAllAsmPrinters();
  }

  // Get TargetMachine.
  std::unique_ptr<TargetMachine> getTargetMachine() {
    // Get target triple for X86_64
    Triple TargetTriple("x86_64--");
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);
    if (!T)
      return nullptr;

    TargetOptions Options;
    return std::unique_ptr<TargetMachine>(T->createTargetMachine(
        TargetTriple, "", "", Options, Reloc::Model::PIC_, {},
        CodeGenOptLevel::Default));
  }

  std::unique_ptr<MCContext> getMCContext(TargetMachine &TM) {
    Triple TargetTriple("x86_64--");
    std::unique_ptr<MCContext> Ctx(
        new MCContext(TargetTriple, TM.getMCAsmInfo(), TM.getMCRegisterInfo(),
                      TM.getMCSubtargetInfo()));

    Ctx->setObjectFileInfo(TM.getObjFileLowering());
    TM.getObjFileLowering()->Initialize(*Ctx, TM);
    Ctx->setObjectFileInfo(TM.getObjFileLowering());
    return Ctx;
  }

  MachineModuleInfoWrapperPass *getMMIWP(TargetMachine &TM,
                                         MCContext &ExternMC) {
    return new MachineModuleInfoWrapperPass(&TM, &ExternMC);
  }

  void SetUp() override {
    // Module to compile.
    const char *FooStr = R""""(
      define void @foo() {
        call void @baz()
        call void @boo()
        ret void
      }

      define internal void @baz() {
        ret void
      }

      define void @bar() {
        call void @baz()
        call void @boo()
        ret void
      }

      define internal void @boo() {
        ret void
      }
    )"""";
    StringRef AssemblyF(FooStr);

    TM = getTargetMachine();

    if (!TM)
      GTEST_SKIP();

    // Parse the module.
    Expected<bool> MResult = M.create(
        [&](llvm::LLVMContext &Context) -> Expected<std::unique_ptr<Module>> {
          SMDiagnostic SMError;
          std::unique_ptr<Module> M =
              parseAssemblyString(AssemblyF, SMError, Context);
          if (!M) {
            return make_error<StringError>("could not load LLVM file",
                                           inconvertibleErrorCode());
          }
          return M;
        });

    ASSERT_FALSE((!MResult));

    M->setDataLayout(TM->createDataLayout());
  }

  LLVMModuleAndContext M;
  std::unique_ptr<TargetMachine> TM;
};

TEST_F(MCLinkerTest, SplitModuleCompilerMCLink) {

  SymbolAndMCInfo SMCInfo;
  bool Failed = false;

  auto OutputLambda =
      [&](llvm::unique_function<LLVMModuleAndContext()> ProduceModule,
          std::optional<int64_t> Idx, unsigned NumFunctionsBase) mutable {
        LLVMModuleAndContext SubModule = ProduceModule();
        std::unique_ptr<TargetMachine> TM = getTargetMachine();
        std::unique_ptr<MCContext> MCCtx = getMCContext(*TM);
        MachineModuleInfoWrapperPass *MMIWP = getMMIWP(*TM, *MCCtx);

        // Create codegen pipeline.
        legacy::PassManager PassMgr;
        mclinker::addPassesToEmitMC(*TM, PassMgr, true, MMIWP,
                                    NumFunctionsBase);
        // Generate code.
        if (!PassMgr.run(*SubModule))
          Failed = true;

        // Put codegen result back.
        SMCInfo.McInfos.emplace_back(std::make_unique<MCInfo>(
            std::make_unique<MachineModuleInfo>(std::move(MMIWP->getMMI())),
            std::move(SubModule), std::move(TM), std::move(MCCtx), Idx));
      };

  // Split the module into per-function submodules and run codegen pipeline for
  // each submodule but stop before AsmPrint.
  splitPerFunction(std::move(M), OutputLambda, SMCInfo.SymbolLinkageTypes, 0);

  // Create and run MCLinker.
  std::unique_ptr<TargetMachine> TMMCLink = getTargetMachine();
  SmallVector<SymbolAndMCInfo *> SMCInfos{&SMCInfo};
  llvm::StringMap<llvm::GlobalValue::LinkageTypes> SymbolLinkageTypes;

  MCLinker Linker(SMCInfos, *TMMCLink, SymbolLinkageTypes);
  Expected<std::unique_ptr<WritableMemoryBuffer>> LinkResult =
      Linker.linkAndPrint("SplitModuleCompilerMCLink",
                          llvm::CodeGenFileType::ObjectFile, true);

  // Check MCLinker is successful.
  ASSERT_FALSE((!LinkResult));

  // Check the binary object output.
  Expected<std::unique_ptr<llvm::object::Binary>> Binary =
      llvm::object::createBinary((*LinkResult)->getMemBufferRef());

  ASSERT_FALSE((!Binary));

  llvm::object::ObjectFile *Obj =
      dyn_cast<llvm::object::ObjectFile>((*Binary).get());
  ASSERT_TRUE(Obj != nullptr);

  if (!Obj->isELF())
    GTEST_SKIP();

  auto *ELFO = dyn_cast<llvm::object::ELFObjectFileBase>(Obj);

  if (!ELFO)
    GTEST_SKIP();

  for (auto Sym : ELFO->symbols()) {
    Expected<StringRef> Name = Sym.getName();
    if (!Name)
      GTEST_SKIP();

    if (*Name == "foo") {
      // foo is global
      EXPECT_TRUE(Sym.getBinding() == 1);
    } else if (*Name == "bar") {
      // bar is global
      EXPECT_TRUE(Sym.getBinding() == 1);
    } else if (*Name == "baz") {
      // baz is internal
      EXPECT_TRUE(Sym.getBinding() == 0);
    } else if (*Name == "boo") {
      // boo is internal
      EXPECT_TRUE(Sym.getBinding() == 0);
    }
  }
}

} // end anonymous namespace
