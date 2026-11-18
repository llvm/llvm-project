//===- MCPseudoProbeTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCPseudoProbe.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

GUIDProbeFunctionMap makeMap() {
  GUIDProbeFunctionMap Map;
  Map.emplace_back(10, 100, "foo");
  Map.emplace_back(20, 200, "bar");
  Map.emplace_back(30, 300, "baz");
  Map.shrink_to_fit();
  return Map;
}

TEST(GUIDProbeFunctionMapTest, FindPresent) {
  GUIDProbeFunctionMap Map = makeMap();
  for (uint64_t GUID : {10, 20, 30}) {
    auto It = Map.find(GUID);
    ASSERT_NE(It, Map.end());
    EXPECT_EQ(It->FuncGUID, GUID);
  }
  EXPECT_EQ(Map.find(20)->FuncName, "bar");
}

TEST(GUIDProbeFunctionMapTest, FindAbsent) {
  GUIDProbeFunctionMap Map = makeMap();
  EXPECT_EQ(Map.find(5), Map.end());
  EXPECT_EQ(Map.find(15), Map.end());
}

TEST(GUIDProbeFunctionMapTest, FindGreaterThanAll) {
  GUIDProbeFunctionMap Map = makeMap();
  EXPECT_EQ(Map.find(31), Map.end());
  EXPECT_EQ(Map.find(UINT64_MAX), Map.end());
}

TEST(GUIDProbeFunctionMapTest, FindInEmpty) {
  GUIDProbeFunctionMap Map;
  EXPECT_EQ(Map.find(0), Map.end());
  EXPECT_EQ(Map.find(42), Map.end());
}

// The callback of buildAddress2ProbeMap sees every probe in the section,
// including the probes of a function that the GUID filter skips and of its
// inlinees, while only the probes of the selected functions are decoded.
TEST(MCPseudoProbeDecoderTest, CallbackSeesSkippedFunctions) {
  // Function records as described in MCPseudoProbe.h: GUID, number of probes,
  // number of inlinees, then for each probe its index, a byte with the probe
  // type (0x80: an address delta follows) and the address delta. An inlinee
  // record starts with the index of its inline site.
  const uint8_t Section[] = {
      // GUID 1: a block probe.
      1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0x80, 4,
      // GUID 2: a block probe and an inlinee at site 2.
      2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0x80, 4,
      // Inlinee GUID 3: an indirect call probe with index 2.
      2, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0x81, 4};
  MCPseudoProbeDecoder Decoder;
  MCPseudoProbeDecoder::Uint64Set GuidFilter;
  GuidFilter.insert(1);
  MCPseudoProbeDecoder::Uint64Map FuncStartAddrs;
  std::vector<std::tuple<uint64_t, uint32_t, PseudoProbeType>> Seen;
  ASSERT_TRUE(Decoder.buildAddress2ProbeMap(
      Section, sizeof(Section), GuidFilter, FuncStartAddrs,
      [&](uint64_t Guid, uint32_t Index, PseudoProbeType Type) {
        Seen.emplace_back(Guid, Index, Type);
      }));
  EXPECT_EQ(Decoder.getAddress2ProbesMap().size(), 1u);
  std::vector<std::tuple<uint64_t, uint32_t, PseudoProbeType>> Expected = {
      {1, 1, PseudoProbeType::Block},
      {2, 1, PseudoProbeType::Block},
      {3, 2, PseudoProbeType::IndirectCall}};
  EXPECT_EQ(Seen, Expected);
}

} // namespace
