//===- VirtualFileSystemTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Module.h"
#include "llvm/LTO/LTOBackend.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Transforms/Utils/AssignGUID.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(LTOVirtualFileSystemTest, ImportUsesLogicalModuleIdentifier) {
  if (InitializeNativeTarget())
    GTEST_SKIP() << "Native target not built";

  LLVMContext Context;
  Context.enableDebugTypeODRUniquing();
  SMDiagnostic Err;
  auto Source = parseAssemblyString(R"(
    @local = internal global i32 42
    define i32 @callee() {
      %v = load i32, ptr @local
      ret i32 %v
    }
  )",
                                    Err, Context);
  ASSERT_TRUE(Source);
  Source->setTargetTriple(Triple(sys::getDefaultTargetTriple()));
  AssignGUIDPass::runOnModule(*Source);
  ProfileSummaryInfo PSI(*Source);
  ModuleSummaryIndex Summary = buildModuleSummaryIndex(*Source, nullptr, &PSI);
  SmallString<0> Bitcode;
  raw_svector_ostream OS(Bitcode);
  WriteBitcodeToFile(*Source, OS, /*ShouldPreserveUseListOrder=*/false,
                     &Summary, /*GenerateHash=*/true);

  StringRef Path = "callee.bc";
  ModuleSummaryIndex CombinedIndex(/*HaveGVs=*/false);
  ASSERT_THAT_ERROR(
      readModuleSummaryIndex(MemoryBufferRef(Bitcode, Path), CombinedIndex),
      Succeeded());
  CombinedIndex.addModule("main.bc");
  FunctionImporter::ImportIDTable ImportIDs;
  FunctionImporter::ImportMapTy Imports(ImportIDs);
  Imports.addDefinition(Path, Source->getFunction("callee")->getGUID());
  Imports.addDefinition(Path, Source->getNamedGlobal("local")->getGUID());
  std::string PromotedName = ModuleSummaryIndex::getGlobalNameForLocal(
      "local", CombinedIndex.getModuleHash(Path));

  // Neither the empty identifier nor an unrelated backing filename may
  // replace the module path used in the combined index.
  for (StringRef BufferName : {StringRef(), StringRef("backing.bc")}) {
    SCOPED_TRACE(BufferName.str());
    auto FS = makeIntrusiveRefCnt<vfs::InMemoryFileSystem>();
    ASSERT_TRUE(FS->addFile(
        Path, 0, MemoryBuffer::getMemBufferCopy(Bitcode, BufferName)));
    Module Dest("main.bc", Context);
    Dest.setTargetTriple(Source->getTargetTriple());

    lto::Config Conf;
    Conf.FS = FS;
    bool Imported = false;
    Conf.PostImportModuleHook = [&](unsigned, const Module &M) {
      Imported = true;
      const Function *Callee = M.getFunction("callee");
      EXPECT_TRUE(Callee && !Callee->isDeclaration());
      const GlobalVariable *Local = M.getNamedGlobal(PromotedName);
      EXPECT_TRUE(Local && Local->hasInitializer());
      if (Local && Local->hasInitializer())
        EXPECT_EQ(cast<ConstantInt>(Local->getInitializer())->getZExtValue(),
                  42u);
      return false;
    };
    ASSERT_THAT_ERROR(lto::thinBackend(Conf, 0, nullptr, Dest, CombinedIndex,
                                       Imports, GVSummaryMapTy(),
                                       /*ModuleMap=*/nullptr,
                                       /*CodeGenOnly=*/false, {}),
                      Succeeded());
    EXPECT_TRUE(Imported);
  }
}

TEST(LTOVirtualFileSystemTest, CustomPipelineUsesVirtualSampleProfile) {
  if (InitializeNativeTarget())
    GTEST_SKIP() << "Native target not built";

  LLVMContext Context;
  std::string Diagnostics;
  Context.setDiagnosticHandlerCallBack(
      [](const DiagnosticInfo *DI, void *Opaque) {
        raw_string_ostream OS(*static_cast<std::string *>(Opaque));
        DiagnosticPrinterRawOStream Printer(OS);
        DI->print(Printer);
      },
      &Diagnostics);
  SMDiagnostic Err;
  auto M = parseAssemblyString("define void @f() { ret void }", Err, Context);
  ASSERT_TRUE(M);
  M->setTargetTriple(Triple(sys::getDefaultTargetTriple()));

  // The explicitly named pass uses -sample-profile-file, independently of
  // Config::SampleProfile, which configures the default LTO pipeline.
  auto *ProfileOption = static_cast<cl::opt<std::string> *>(
      cl::getRegisteredOptions().lookup("sample-profile-file"));
  ASSERT_NE(ProfileOption, nullptr);
  SaveAndRestore<std::string> RestoreProfile(ProfileOption->getValue(),
                                             "virtual-profile.prof");
  auto FS = makeIntrusiveRefCnt<vfs::InMemoryFileSystem>();
  ASSERT_TRUE(FS->addFile("virtual-profile.prof", 0,
                          MemoryBuffer::getMemBufferCopy("f:0:0\n")));
  lto::Config Conf;
  Conf.FS = FS;
  Conf.OptPipeline = "sample-profile";
  Conf.PostOptModuleHook = [](unsigned, const Module &) { return false; };
  ModuleSummaryIndex Index(/*HaveGVs=*/false);
  ASSERT_THAT_ERROR(lto::backend(Conf, nullptr, 1, *M, Index, {}), Succeeded());
  EXPECT_TRUE(Diagnostics.empty()) << Diagnostics;
  EXPECT_NE(M->getProfileSummary(/*IsCS=*/false), nullptr);
}

} // namespace
