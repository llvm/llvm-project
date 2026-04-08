//===- RemoteCachingService.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/RemoteCachingService/RemoteCachingService.h"
#include "llvm/Config/llvm-config.h"

using namespace llvm;

static Expected<std::pair<std::shared_ptr<cas::ObjectStore>,
                          std::shared_ptr<cas::ActionCache>>>
createGRPCRelayDBs(const llvm::Twine &Path) {
  std::shared_ptr<cas::ObjectStore> CAS;
  std::shared_ptr<cas::ActionCache> AC;
  SmallString<128> Buffer;
  Path.toVector(Buffer);
  if (Error E = cas::createGRPCRelayCAS(Buffer).moveInto(CAS))
    return std::move(E);
  if (Error E = cas::createGRPCActionCache(Buffer).moveInto(AC))
    return std::move(E);
  return std::make_pair(std::move(CAS), std::move(AC));
}

cas::RegisterGRPCCAS::RegisterGRPCCAS() {
  cas::registerCASURLScheme("grpc://", createGRPCRelayDBs);
}
