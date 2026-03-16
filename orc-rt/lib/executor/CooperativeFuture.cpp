//===- CooperativeFuture.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the implementation of APIs in the orc-rt/CooperativeFuture.h header.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/CooperativeFuture.h"

namespace orc_rt {

CooperativeFutureTaskRunner::~CooperativeFutureTaskRunner() = default;

} // namespace orc_rt
