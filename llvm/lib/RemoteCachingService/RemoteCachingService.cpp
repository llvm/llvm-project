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

cas::RegisterGRPCCAS::RegisterGRPCCAS() {
  cas::registerCASURLScheme("grpc://", &cas::createGRPCRelayCAS);
}
