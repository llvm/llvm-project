//===-- MICmdCommands.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

namespace MICmnCommands {

//++
//============================================================================
// Details: MI Command are instantiated and registered automatically with the
//          Command Factory
//--
bool RegisterAll();
}
