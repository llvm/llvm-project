//===- NullService.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/RemoteCachingService/RemoteCachingService.h"
#include "llvm/RemoteCachingService/Client.h"

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::cas::remote;

void AsyncCallerContext::anchor() {}
void AsyncQueueBase::anchor() {}

void KeyValueDBClient::GetValueAsyncQueue::anchor() {}
void KeyValueDBClient::PutValueAsyncQueue::anchor() {}
void KeyValueDBClient::anchor() {}

void CASDBClient::LoadAsyncQueue::anchor() {}
void CASDBClient::SaveAsyncQueue::anchor() {}
void CASDBClient::GetAsyncQueue::anchor() {}
void CASDBClient::PutAsyncQueue::anchor() {}
void CASDBClient::anchor() {}

static Error unsupportedError() {
  return createStringError(
      inconvertibleErrorCode(),
      "RemoteCachingService (LLVM_CAS_ENABLE_REMOTE_CACHE) is disabled");
}

Expected<std::unique_ptr<ObjectStore>>
cas::createGRPCRelayCAS(const Twine &Path) {
  return unsupportedError();
}

Expected<std::unique_ptr<cas::ActionCache>>
cas::createGRPCActionCache(StringRef Path) {
  return unsupportedError();
}

Expected<ClientServices>
remote::createCompilationCachingRemoteClient(StringRef SocketPath) {
  return unsupportedError();
}

Expected<std::unique_ptr<CASDBClient>>
remote::createRemoteCASDBClient(StringRef SocketPath) {
  return unsupportedError();
}

Expected<std::unique_ptr<KeyValueDBClient>>
remote::createRemoteKeyValueClient(StringRef SocketPath) {
  return unsupportedError();
}
