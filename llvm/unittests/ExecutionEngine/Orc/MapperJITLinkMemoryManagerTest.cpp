//===-------------- MapperJITLinkMemoryManagerTest.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"

#include "llvm/ExecutionEngine/Orc/MapperJITLinkMemoryManager.h"

#include "llvm/Testing/Support/Error.h"

#include <vector>

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::orc;
using namespace llvm::orc::shared;

namespace {

TEST(MapperJITLinkMemoryManagerTest, InProcess) {
  auto MemMgr = cantFail(
      MapperJITLinkMemoryManager::CreateWithMapper<InProcessMemoryMapper>());

  StringRef Hello = "hello";
  auto SSA = jitlink::SimpleSegmentAlloc::Create(
      *MemMgr, nullptr, {{jitlink::MemProt::Read, {Hello.size(), Align(1)}}});
  EXPECT_THAT_EXPECTED(SSA, Succeeded());
  auto SegInfo = SSA->getSegInfo(jitlink::MemProt::Read);
  memcpy(SegInfo.WorkingMem.data(), Hello.data(), Hello.size());

  auto FA = SSA->finalize();
  EXPECT_THAT_EXPECTED(FA, Succeeded());

  ExecutorAddr TargetAddr(SegInfo.Addr);

  const char *TargetMem = TargetAddr.toPtr<const char *>();
  StringRef TargetHello(TargetMem, Hello.size());
  EXPECT_EQ(Hello, TargetHello);

  auto Err2 = MemMgr->deallocate(std::move(*FA));
  EXPECT_THAT_ERROR(std::move(Err2), Succeeded());
}

} // namespace
