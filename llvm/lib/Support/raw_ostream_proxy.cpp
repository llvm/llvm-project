//===- raw_ostream_proxy.cpp - Implement the raw_ostream proxies ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/raw_ostream_proxy.h"

using namespace llvm;

void raw_ostream_proxy::anchor() {}

void raw_pwrite_stream_proxy::anchor() {}
