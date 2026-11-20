//===- LTOConfigBitcodeTest.cpp - LTO config bitcode tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/LTO/LTOConfigBitcode.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#include <limits>

using namespace llvm;
using namespace llvm::lto;

static Expected<std::optional<Config>> readIndexConfig(MemoryBufferRef Buffer) {
  auto Index = getModuleSummaryIndex(Buffer);
  if (!Index)
    return Index.takeError();
  return readLTOConfigFromSummaryIndexIfPresent(**Index);
}

TEST(LTOConfigBitcodeTest, RoundTripConfigThroughSummaryIndex) {
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

  ModuleSummaryIndex Index(/*HaveGVs=*/false);
  Index.setLTOConfigJSON(serializeLTOConfig(Input));
  SmallString<0> Storage;
  raw_svector_ostream OS(Storage);
  writeIndexToFile(Index, OS);

  auto ParsedConfig =
      readIndexConfig(MemoryBufferRef(Storage, "config.thinlto.bc"));
  ASSERT_THAT_EXPECTED(ParsedConfig, Succeeded());
  ASSERT_TRUE(ParsedConfig->has_value());
  const Config *Output = &**ParsedConfig;

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

TEST(LTOConfigBitcodeTest, RoundTripNestedOptionsThroughSummaryIndex) {
  ModuleSummaryIndex InputIndex(/*HaveGVs=*/false);
  Config InputConfig;
  InputConfig.CPU = "summary-cpu";
  InputConfig.OptLevel = 3;
  InputConfig.Options.DataSections = true;
  InputConfig.Options.TLSSize = 255;
  InputConfig.Options.MCOptions.MCNoExecStack = true;
  InputConfig.Options.MCOptions.BinutilsVersion = {2, 41};
  InputConfig.Options.MCOptions.OutputAsmVariant = 0;
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

  auto ParsedConfig = readLTOConfigFromSummaryIndexIfPresent(OutputIndex);
  ASSERT_THAT_EXPECTED(ParsedConfig, Succeeded());
  ASSERT_TRUE(ParsedConfig->has_value());
  const Config *OutputConfig = &**ParsedConfig;
  EXPECT_EQ(OutputConfig->CPU, "summary-cpu");
  EXPECT_EQ(OutputConfig->OptLevel, 3u);
  EXPECT_TRUE(OutputConfig->Options.DataSections);
  EXPECT_EQ(OutputConfig->Options.TLSSize, 255u);
  EXPECT_TRUE(OutputConfig->Options.MCOptions.MCNoExecStack);
  EXPECT_EQ(OutputConfig->Options.MCOptions.BinutilsVersion,
            InputConfig.Options.MCOptions.BinutilsVersion);
  EXPECT_EQ(OutputConfig->Options.MCOptions.OutputAsmVariant,
            std::optional(0u));
  EXPECT_EQ(OutputConfig->PTO.MergeFunctions, std::optional(false));
  EXPECT_FALSE(OutputConfig->PTO.ForgetAllSCEVInLoopUnroll);
  EXPECT_EQ(OutputConfig->PTO.MaxDevirtIterations, std::optional(0u));
  EXPECT_EQ(OutputConfig->RemarksHotnessThreshold, std::nullopt);
  ASSERT_TRUE(OutputConfig->Options.BBSectionsFuncListBuf);
  EXPECT_EQ(OutputConfig->Options.BBSectionsFuncListBuf->getBuffer(), Profile);
}

TEST(LTOConfigBitcodeTest, InvalidJSONInSummaryIndex) {
  ModuleSummaryIndex Index(/*HaveGVs=*/false);
  Index.setLTOConfigJSON("{");
  SmallString<0> Storage;
  raw_svector_ostream OS(Storage);
  writeIndexToFile(Index, OS);

  EXPECT_THAT_EXPECTED(
      readIndexConfig(MemoryBufferRef(Storage, "invalid.thinlto.bc")),
      Failed());
}

TEST(LTOConfigBitcodeTest, ThinLTOSummaryIndexWithoutConfig) {
  ModuleSummaryIndex Index(/*HaveGVs=*/false);
  SmallString<0> Storage;
  raw_svector_ostream OS(Storage);
  writeIndexToFile(Index, OS);

  MemoryBufferRef Buffer(StringRef(Storage.data(), Storage.size()),
                         "summary.thinlto.bc");
  auto OutputConfig = readIndexConfig(Buffer);
  ASSERT_THAT_EXPECTED(OutputConfig, Succeeded());
  EXPECT_FALSE(OutputConfig->has_value());
}
