//===-- MockTildeExpressionResolver.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UNITTESTS_UTILITY_MOCKS_MOCK_TILDE_EXPRESSION_RESOLVER_H
#define LLDB_UNITTESTS_UTILITY_MOCKS_MOCK_TILDE_EXPRESSION_RESOLVER_H

#include "lldb/Utility/TildeExpressionResolver.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"

namespace lldb_private {
class MockTildeExpressionResolver : public TildeExpressionResolver {
  llvm::StringRef CurrentUser;
  llvm::StringMap<llvm::StringRef> UserDirectories;

public:
  MockTildeExpressionResolver(llvm::StringRef CurrentUser,
                              llvm::StringRef HomeDir);

  void AddKnownUser(llvm::StringRef User, llvm::StringRef HomeDir);
  void Clear();
  void SetCurrentUser(llvm::StringRef User);

  bool ResolveExact(llvm::StringRef Expr,
                    llvm::SmallVectorImpl<char> &Output) override;
  bool ResolvePartial(llvm::StringRef Expr, llvm::StringSet<> &Output) override;
};
} // namespace lldb_private

#endif
