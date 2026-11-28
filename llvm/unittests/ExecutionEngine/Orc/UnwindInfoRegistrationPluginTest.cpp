//===-- UnwindInfoRegistrationPluginTest.cpp - Unwind range unit tests
//-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/UnwindInfoRegistrationPlugin.h"
#include "OrcTestCommon.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::orc;
using namespace llvm::orc::shared;

TEST(UnwindInfoRegistrationPluginTest,
     CoalescesDuplicateAndAdjacentCodeRanges) {
  ExecutionSession ES{std::make_unique<UnsupportedExecutorProcessControl>()};
  auto &JD = ES.createBareJITDylib("main");
  LinkGraph G("shared-block", ES.getSymbolStringPool(),
              Triple("arm64-apple-darwin"), 8, SubtargetFeatures(),
              getGenericEdgeKindName);
  auto &Text = G.createSection("__TEXT,__text", MemProt::Read | MemProt::Exec);
  char Content[8] = {};
  auto &First = G.createContentBlock(Text, Content, ExecutorAddr(0x1000), 4, 0);
  auto &Adjacent =
      G.createContentBlock(Text, Content, ExecutorAddr(0x1008), 4, 0);
  auto &Separate =
      G.createContentBlock(Text, Content, ExecutorAddr(0x2000), 4, 0);
  Symbol *Functions[] = {
      &G.addDefinedSymbol(First, 0, "first", 4, Linkage::Strong, Scope::Default,
                          true, true),
      &G.addDefinedSymbol(First, 4, "second", 4, Linkage::Strong,
                          Scope::Default, true, true),
      &G.addDefinedSymbol(Adjacent, 0, "third", 8, Linkage::Strong,
                          Scope::Default, true, true),
      &G.addDefinedSymbol(Separate, 0, "fourth", 8, Linkage::Strong,
                          Scope::Default, true, true)};
  auto &Frames = G.createSection("__TEXT,__eh_frame", MemProt::Read);
  auto &FrameBlock =
      G.createContentBlock(Frames, Content, ExecutorAddr(0x3000), 8, 0);
  for (auto *F : Functions)
    FrameBlock.addEdge(Edge::KeepAlive, 0, *F, 0);
  auto &Unwind = G.createSection("__TEXT,__unwind_info", MemProt::Read);
  auto &UnwindBlock =
      G.createContentBlock(Unwind, Content, ExecutorAddr(0x4000), 8, 0);
  UnwindBlock.addEdge(Edge::KeepAlive, 0, *Functions[0], 0);
  UnwindBlock.addEdge(Edge::KeepAlive, 0, *Functions[1], 0);
  G.addAbsoluteSymbol("__jitlink$libunwind_dso_base", ExecutorAddr(0x5000), 0,
                      Linkage::Strong, Scope::Local, false);

  UnwindInfoRegistrationPlugin Plugin(ES, ExecutorAddr(1), ExecutorAddr(2));
  auto Anchor = ES.intern("anchor");
  cantFail(JD.define(std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap{{Anchor, JITSymbolFlags::Exported}},
      [&](std::unique_ptr<MaterializationResponsibility> MR) {
        PassConfiguration Config;
        Plugin.modifyPassConfig(*MR, G, Config);
        for (auto &Pass : Config.PostFixupPasses)
          cantFail(Pass(G));
        cantFail(MR->notifyResolved(
            {{Anchor, ExecutorSymbolDef(ExecutorAddr(0x1000),
                                        JITSymbolFlags::Exported)}}));
        cantFail(MR->notifyEmitted({}));
      })));
  cantFail(ES.lookup(&JD, "anchor"));
  cantFail(ES.endSession());

  ASSERT_EQ(G.allocActions().size(), 1U);
  auto &Actions = G.allocActions().front();
  auto Args = Actions.Finalize.getArgData();
  SPSInputBuffer Input(Args.data(), Args.size());
  std::vector<ExecutorAddrRange> Registered, Deregistered;
  ExecutorAddr DSOBase;
  ExecutorAddrRange FrameRange, UnwindRange;
  using RegisterArgs =
      SPSArgList<SPSSequence<SPSExecutorAddrRange>, SPSExecutorAddr,
                 SPSExecutorAddrRange, SPSExecutorAddrRange>;
  ASSERT_TRUE(RegisterArgs::deserialize(Input, Registered, DSOBase, FrameRange,
                                        UnwindRange));
  auto DeallocArgs = Actions.Dealloc.getArgData();
  SPSInputBuffer DeallocInput(DeallocArgs.data(), DeallocArgs.size());
  ASSERT_TRUE(SPSArgList<SPSSequence<SPSExecutorAddrRange>>::deserialize(
      DeallocInput, Deregistered));
  ASSERT_EQ(Registered.size(), 2U);
  EXPECT_EQ(Registered[0], ExecutorAddrRange(ExecutorAddr(0x1000), 16));
  EXPECT_EQ(Registered[1], ExecutorAddrRange(ExecutorAddr(0x2000), 8));
  EXPECT_EQ(Deregistered, Registered);
  EXPECT_EQ(DSOBase, ExecutorAddr(0x5000));
  EXPECT_EQ(FrameRange, ExecutorAddrRange(ExecutorAddr(0x3000), 8));
  EXPECT_EQ(UnwindRange, ExecutorAddrRange(ExecutorAddr(0x4000), 8));
}
