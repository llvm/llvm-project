//===- MockGRPCServer.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/thread.h"
#include <condition_variable>
#include <memory>
#include <mutex>
#include <utility>

#include "CASTestConfig.h"

#if LLVM_CAS_ENABLE_REMOTE_CACHE

#include "llvm/RemoteCachingService/RemoteCacheServer.h"

using namespace llvm;
using namespace llvm::cas;

namespace {

class MockEnvImpl final : public unittest::cas::MockEnv {
public:
  MockEnvImpl(StringRef Socket, StringRef Temp,
              std::unique_ptr<ObjectStore> CAS,
              std::unique_ptr<ActionCache> Cache);
  ~MockEnvImpl();

private:
  remote::RemoteCacheServer Server;
  std::unique_ptr<thread> Thread;
};

} // namespace

MockEnvImpl::MockEnvImpl(StringRef Socket, StringRef Temp,
                         std::unique_ptr<ObjectStore> CAS,
                         std::unique_ptr<ActionCache> Cache)
    : Server(Socket, Temp, std::move(CAS), std::move(Cache)) {
  Server.Start();
  auto ServerFunc = [&]() { Server.Listen(); };
  Thread = std::make_unique<thread>(ServerFunc);
}

MockEnvImpl::~MockEnvImpl() {
  Server.Shutdown();
  Thread->join();
}

std::unique_ptr<llvm::unittest::cas::MockEnv> createGRPCEnv(StringRef Socket,
                                                            StringRef TempDir) {
  auto CAS = createInMemoryCAS();
  auto Cache = createInMemoryActionCache();

  return std::make_unique<MockEnvImpl>(Socket, TempDir, std::move(CAS),
                                       std::move(Cache));
}

#endif /* LLVM_CAS_ENABLE_REMOTE_CACHE */
