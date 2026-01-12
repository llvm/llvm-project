//===- Protocol.cpp - LSP JSON protocol unit tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/LSP/Protocol.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::lsp;
using namespace testing;

namespace {

TEST(ProtocolTest, DiagnosticTagPresent) {
  Diagnostic diagnostic;
  diagnostic.tags.push_back(DiagnosticTag::Unnecessary);

  llvm::json::Value json = toJSON(diagnostic);
  const llvm::json::Object *o = json.getAsObject();
  const llvm::json::Array *v = o->get("tags")->getAsArray();
  EXPECT_EQ(*v, llvm::json::Array{1});

  Diagnostic parsed;
  llvm::json::Path::Root root = llvm::json::Path::Root();
  bool success = fromJSON(json, parsed, llvm::json::Path(root));
  EXPECT_TRUE(success);
  ASSERT_EQ(parsed.tags.size(), (size_t)1);
  EXPECT_EQ(parsed.tags.at(0), DiagnosticTag::Unnecessary);
}

TEST(ProtocolTest, DiagnosticTagNotPresent) {
  Diagnostic diagnostic;

  llvm::json::Value json = toJSON(diagnostic);
  const llvm::json::Object *o = json.getAsObject();
  const llvm::json::Value *v = o->get("tags");
  EXPECT_EQ(v, nullptr);

  Diagnostic parsed;
  llvm::json::Path::Root root = llvm::json::Path::Root();
  bool success = fromJSON(json, parsed, llvm::json::Path(root));
  EXPECT_TRUE(success);
  EXPECT_TRUE(parsed.tags.empty());
}

} // namespace
