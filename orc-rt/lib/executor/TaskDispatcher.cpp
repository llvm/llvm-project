//===- TaskDispatch.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the implementation of APIs in the orc-rt/TaskDispatch.h header.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/TaskDispatcher.h"

namespace orc_rt {

Task::~Task() = default;
TaskDispatcher::~TaskDispatcher() = default;

} // namespace orc_rt
