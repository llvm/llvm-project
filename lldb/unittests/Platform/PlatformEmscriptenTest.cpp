//===-- PlatformEmscriptenTest.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Platform/Emscripten/PlatformEmscripten.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb_private::platform_emscripten;

TEST(PlatformEmscriptenTest, RecognizesEmscriptenTriple) {
  ArchSpec emscripten_arch("wasm32-unknown-emscripten");
  EXPECT_TRUE(PlatformEmscripten::CreateInstance(false, &emscripten_arch));
}
