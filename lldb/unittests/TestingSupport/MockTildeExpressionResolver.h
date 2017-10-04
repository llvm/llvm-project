//===--------------------- TildeExpressionResolver.h ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
