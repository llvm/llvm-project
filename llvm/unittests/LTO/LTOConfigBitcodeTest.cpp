//===- LTOConfigBitcodeTest.cpp - LTO config bitcode tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/LTO/LTOConfigBitcode.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/LTO/TargetOptionsBitcode.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#include <limits>

using namespace llvm;
using namespace llvm::lto;

TEST(TargetOptionsBitcodeTest, RoundTripThroughBitcode) {
  TargetOptions Input;
  Input.BinutilsVersion = {2, 41};
  Input.FunctionSections = true;
  Input.DataSections = true;
  Input.GlobalISelAbort = GlobalISelAbortMode::DisableWithDiag;
  Input.BBSections = BasicBlockSection::List;
  Input.BBSectionsFuncListBuf = MemoryBuffer::getMemBufferCopy(
      "v1\nf foo\nc 0 1\n", "basic-block-sections.profile");
  Input.EnableDefaultMachineVerifier = false;
  Input.StackUsageFile = "output.su";
  Input.ExceptionModel = ExceptionHandling::Wasm;
  Input.MCOptions.ABIName = "test-abi";
  Input.MCOptions.OutputAsmVariant = 1;
  Input.MCOptions.IASSearchPaths = {"include/one", "include/two"};
  Input.MCOptions.InstPrinterOptions = {"no-aliases"};
  Input.MCOptions.LargeEHEncoding = true;

  LLVMContext WriteCtx;
  Module M("target-options", WriteCtx);
  ASSERT_THAT_ERROR(encodeTargetOptionsToModule(M, Input), Succeeded());
  EXPECT_TRUE(hasEncodedTargetOptions(M));

  SmallString<0> Storage;
  raw_svector_ostream OS(Storage);
  WriteBitcodeToFile(M, OS);

  LLVMContext ReadCtx;
  Expected<std::unique_ptr<Module>> Parsed = parseBitcodeFile(
      MemoryBufferRef(StringRef(Storage.data(), Storage.size()), "options.bc"),
      ReadCtx);
  ASSERT_THAT_EXPECTED(Parsed, Succeeded());
  Expected<TargetOptions> Output = decodeTargetOptionsFromModule(**Parsed);
  ASSERT_THAT_EXPECTED(Output, Succeeded());

  EXPECT_EQ(Output->BinutilsVersion, Input.BinutilsVersion);
  EXPECT_TRUE(Output->FunctionSections);
  EXPECT_TRUE(Output->DataSections);
  EXPECT_EQ(Output->GlobalISelAbort, GlobalISelAbortMode::DisableWithDiag);
  EXPECT_EQ(Output->BBSections, BasicBlockSection::List);
  ASSERT_TRUE(Output->BBSectionsFuncListBuf);
  EXPECT_EQ(Output->BBSectionsFuncListBuf->getBufferIdentifier(),
            "basic-block-sections.profile");
  EXPECT_EQ(Output->BBSectionsFuncListBuf->getBuffer(), "v1\nf foo\nc 0 1\n");
  EXPECT_FALSE(Output->EnableDefaultMachineVerifier);
  EXPECT_EQ(Output->StackUsageFile, "output.su");
  EXPECT_EQ(Output->ExceptionModel, ExceptionHandling::Wasm);
  EXPECT_EQ(Output->MCOptions.ABIName, "test-abi");
  EXPECT_EQ(Output->MCOptions.OutputAsmVariant, 1u);
  EXPECT_EQ(Output->MCOptions.IASSearchPaths, Input.MCOptions.IASSearchPaths);
  EXPECT_EQ(Output->MCOptions.InstPrinterOptions,
            Input.MCOptions.InstPrinterOptions);
  EXPECT_TRUE(Output->MCOptions.LargeEHEncoding);
}

TEST(LTOConfigBitcodeTest, RoundTripThroughFile) {
  Config Input;
  Input.CPU = "generic";
  Input.MAttrs = {"+crc", "+simd"};
  Input.MllvmArgs = {"-inline-threshold=42"};
  Input.PassPluginFilenames = {"plugin.so"};
  Input.RelocModel = std::nullopt;
  Input.CodeModel = CodeModel::Large;
  Input.CGOptLevel = CodeGenOptLevel::Aggressive;
  Input.OptLevel = 3;
  Input.Dtlto = true;
  Input.RemarksHotnessThreshold = std::numeric_limits<uint64_t>::max();
  Input.ThinLTOModulesToCompile = {"one.bc", "two.bc"};
  Input.PTO.LoopInterchange = true;
  Input.Options.FunctionSections = true;
  Input.Options.MCOptions.IASSearchPaths = {"sdk/include"};

  SmallString<128> Path;
  ASSERT_FALSE(sys::fs::createTemporaryFile("lto-config", "bc", Path));
  FileRemover Cleanup(Path);

  ASSERT_THAT_ERROR(writeLTOConfigToFile(Path, Input), Succeeded());
  Expected<Config> Output = readLTOConfigFromFile(Path);
  ASSERT_THAT_EXPECTED(Output, Succeeded());

  EXPECT_EQ(Output->CPU, "generic");
  EXPECT_EQ(Output->MAttrs, Input.MAttrs);
  EXPECT_EQ(Output->MllvmArgs, Input.MllvmArgs);
  EXPECT_EQ(Output->PassPluginFilenames, Input.PassPluginFilenames);
  EXPECT_EQ(Output->RelocModel, std::nullopt);
  EXPECT_EQ(Output->CodeModel, CodeModel::Large);
  EXPECT_EQ(Output->CGOptLevel, CodeGenOptLevel::Aggressive);
  EXPECT_EQ(Output->OptLevel, 3u);
  EXPECT_TRUE(Output->Dtlto);
  EXPECT_EQ(Output->RemarksHotnessThreshold,
            std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(Output->ThinLTOModulesToCompile, Input.ThinLTOModulesToCompile);
  EXPECT_TRUE(Output->PTO.LoopInterchange);
  EXPECT_EQ(Output->PTO.InlinerThreshold, Input.PTO.InlinerThreshold);
  EXPECT_TRUE(Output->Options.FunctionSections);
  EXPECT_EQ(Output->Options.MCOptions.IASSearchPaths,
            Input.Options.MCOptions.IASSearchPaths);
}

TEST(LTOConfigBitcodeTest, RoundTripThroughThinLTOSummaryIndex) {
  ModuleSummaryIndex InputIndex(/*HaveGVs=*/false);
  Config InputConfig;
  InputConfig.CPU = "summary-cpu";
  InputConfig.OptLevel = 3;
  InputConfig.Options.DataSections = true;

  SmallString<0> Storage;
  raw_svector_ostream OS(Storage);
  ASSERT_THAT_ERROR(writeIndexWithLTOConfigToFile(InputIndex, InputConfig, OS),
                    Succeeded());

  MemoryBufferRef Buffer(StringRef(Storage.data(), Storage.size()),
                         "summary.thinlto.bc");

  // Existing summary-index readers must accept the embedded metadata.
  ModuleSummaryIndex OutputIndex(/*HaveGVs=*/false);
  ASSERT_THAT_ERROR(readModuleSummaryIndex(Buffer, OutputIndex), Succeeded());

  Expected<Config> OutputConfig = readLTOConfigFromSummaryIndex(Buffer);
  ASSERT_THAT_EXPECTED(OutputConfig, Succeeded());
  EXPECT_EQ(OutputConfig->CPU, "summary-cpu");
  EXPECT_EQ(OutputConfig->OptLevel, 3u);
  EXPECT_TRUE(OutputConfig->Options.DataSections);
}

TEST(LTOConfigBitcodeTest, ThinLTOSummaryIndexWithoutConfig) {
  ModuleSummaryIndex Index(/*HaveGVs=*/false);
  SmallString<0> Storage;
  raw_svector_ostream OS(Storage);
  writeIndexToFile(Index, OS);

  MemoryBufferRef Buffer(StringRef(Storage.data(), Storage.size()),
                         "summary.thinlto.bc");
  Expected<std::optional<Config>> OutputConfig =
      readLTOConfigFromSummaryIndexIfPresent(Buffer);
  ASSERT_THAT_EXPECTED(OutputConfig, Succeeded());
  EXPECT_FALSE(OutputConfig->has_value());
}
