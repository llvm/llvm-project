//===- TestOpenACC.cpp - OpenACC Test Registration ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains unified registration for all OpenACC test passes.
//
//===----------------------------------------------------------------------===//

namespace mlir {
namespace test {

// Forward declarations of individual test pass registration functions
void registerTestPointerLikeTypeInterfacePass();

// Unified registration function for all OpenACC tests
void registerTestOpenACC() { registerTestPointerLikeTypeInterfacePass(); }

} // namespace test
} // namespace mlir
