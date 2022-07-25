//===- llvm/RemoteCachingService/RemoteCachingService.h ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_REMOTECACHINGSERVICE_REMOTECACHING_SERVICE_H
#define LLVM_REMOTECACHINGSERVICE_REMOTECACHING_SERVICE_H

#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/ObjectStore.h"

namespace llvm::cas {
/// Create GRPC ObjectStore from a path.
Expected<std::unique_ptr<ObjectStore>> createGRPCRelayCAS(const Twine &Path);

/// Create GRPC ActionCache from a path.
Expected<std::unique_ptr<ActionCache>> createGRPCActionCache(StringRef Path);

// Register GRPC CAS.
class RegisterGRPCCAS {
public:
  RegisterGRPCCAS();
};

} // namespace llvm::cas

#endif
