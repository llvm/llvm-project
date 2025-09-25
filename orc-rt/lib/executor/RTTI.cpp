//===- RTTI.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the implementation of APIs in the orc-rt/RTTI.h header.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/RTTI.h"

namespace orc_rt {

char RTTIRoot::ID = 0;
void RTTIRoot::anchor() {}

} // namespace orc_rt
