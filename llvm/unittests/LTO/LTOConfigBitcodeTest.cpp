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
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/LTO/TargetOptionsBitcode.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/JSON.h"
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
  Input.MCOptions.BinutilsVersion = {2, 41};
  Input.FunctionSections = true;
  Input.DataSections = true;
  Input.GlobalISelAbort = GlobalISelAbortMode::DisableWithDiag;
  Input.BBSections = BasicBlockSection::List;
  Input.BBSectionsFuncListBuf = MemoryBuffer::getMemBufferCopy(
      "v1\nf foo\nc 0 1\n", "basic-block-sections.profile");
  Input.EnableDefaultMachineVerifier = false;
  Input.StackUsageFile = "output.su";
  Input.FunctionSplitting = FunctionSplittingMode::All;
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

  EXPECT_EQ(Output->MCOptions.BinutilsVersion, Input.MCOptions.BinutilsVersion);
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
  EXPECT_EQ(Output->FunctionSplitting, FunctionSplittingMode::All);
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
  InputConfig.PTO.MergeFunctions = false;
  InputConfig.PTO.MaxDevirtIterations = 0;
  InputConfig.RemarksHotnessThreshold = std::nullopt;
  StringRef Profile("profile\xFF\0", 9);
  InputConfig.Options.BBSectionsFuncListBuf =
      MemoryBuffer::getMemBufferCopy(Profile, "profile");
  InputIndex.setLTOConfigJSON(serializeLTOConfig(InputConfig));

  SmallString<0> Storage;
  raw_svector_ostream OS(Storage);
  writeIndexToFile(InputIndex, OS);

  MemoryBufferRef Buffer(StringRef(Storage.data(), Storage.size()),
                         "summary.thinlto.bc");

  ModuleSummaryIndex OutputIndex(/*HaveGVs=*/false);
  ASSERT_THAT_ERROR(readModuleSummaryIndex(Buffer, OutputIndex), Succeeded());
  EXPECT_EQ(OutputIndex.getLTOConfigJSON(), InputIndex.getLTOConfigJSON());
  auto JSON = json::parse(OutputIndex.getLTOConfigJSON());
  ASSERT_THAT_EXPECTED(JSON, Succeeded());
  ASSERT_TRUE(JSON->getAsObject());
  EXPECT_EQ(JSON->getAsObject()->getString("CPU"), "summary-cpu");

  Expected<Config> OutputConfig = readLTOConfigFromSummaryIndex(Buffer);
  ASSERT_THAT_EXPECTED(OutputConfig, Succeeded());
  EXPECT_EQ(OutputConfig->CPU, "summary-cpu");
  EXPECT_EQ(OutputConfig->OptLevel, 3u);
  EXPECT_TRUE(OutputConfig->Options.DataSections);
  EXPECT_EQ(OutputConfig->PTO.MergeFunctions, std::optional(false));
  EXPECT_FALSE(OutputConfig->PTO.ForgetAllSCEVInLoopUnroll);
  EXPECT_EQ(OutputConfig->PTO.MaxDevirtIterations, std::optional(0u));
  EXPECT_EQ(OutputConfig->RemarksHotnessThreshold, std::nullopt);
  ASSERT_TRUE(OutputConfig->Options.BBSectionsFuncListBuf);
  EXPECT_EQ(OutputConfig->Options.BBSectionsFuncListBuf->getBuffer(), Profile);
}

TEST(LTOConfigBitcodeTest, ReadsLegacySummaryIndex) {
  LLVMContext Ctx;
  Module LegacyConfig("legacy", Ctx);
  auto *CPU = MDNode::get(
      Ctx, {MDString::get(Ctx, "CPU"), MDString::get(Ctx, "legacy-cpu")});
  auto *Version =
      ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), 1));
  LegacyConfig.getOrInsertNamedMetadata(LTOConfigMetadataName)
      ->addOperand(MDNode::get(Ctx, {Version, CPU}));

  ModuleSummaryIndex Index(/*HaveGVs=*/false);
  SmallString<0> Storage;
  raw_svector_ostream OS(Storage);
  WriteBitcodeToFile(LegacyConfig, OS, /*ShouldPreserveUseListOrder=*/false,
                     &Index);

  auto Output = readLTOConfigFromSummaryIndex(
      MemoryBufferRef(Storage, "legacy.thinlto.bc"));
  ASSERT_THAT_EXPECTED(Output, Succeeded());
  EXPECT_EQ(Output->CPU, "legacy-cpu");
}

TEST(LTOConfigBitcodeTest, InvalidJSONInSummaryIndex) {
  ModuleSummaryIndex Index(/*HaveGVs=*/false);
  Index.setLTOConfigJSON("{");
  SmallString<0> Storage;
  raw_svector_ostream OS(Storage);
  writeIndexToFile(Index, OS);

  EXPECT_THAT_EXPECTED(readLTOConfigFromSummaryIndexIfPresent(
                           MemoryBufferRef(Storage, "invalid.thinlto.bc")),
                       Failed());
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
