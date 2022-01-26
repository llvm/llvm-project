//===-- MockTildeExpressionResolver.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MockTildeExpressionResolver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Path.h"

using namespace lldb_private;
using namespace llvm;

MockTildeExpressionResolver::MockTildeExpressionResolver(StringRef CurrentUser,
                                                         StringRef HomeDir)
    : CurrentUser(CurrentUser) {
  UserDirectories.insert(std::make_pair(CurrentUser, HomeDir));
}

void MockTildeExpressionResolver::AddKnownUser(StringRef User,
                                               StringRef HomeDir) {
  assert(UserDirectories.find(User) == UserDirectories.end());
  UserDirectories.insert(std::make_pair(User, HomeDir));
}

void MockTildeExpressionResolver::Clear() {
  CurrentUser = StringRef();
  UserDirectories.clear();
}

void MockTildeExpressionResolver::SetCurrentUser(StringRef User) {
  assert(UserDirectories.find(User) != UserDirectories.end());
  CurrentUser = User;
}

bool MockTildeExpressionResolver::ResolveExact(StringRef Expr,
                                               SmallVectorImpl<char> &Output) {
  Output.clear();

  assert(!llvm::any_of(
      Expr, [](char c) { return llvm::sys::path::is_separator(c); }));
  assert(Expr.empty() || Expr[0] == '~');
  Expr = Expr.drop_front();
  if (Expr.empty()) {
    auto Dir = UserDirectories[CurrentUser];
    Output.append(Dir.begin(), Dir.end());
    return true;
  }

  for (const auto &User : UserDirectories) {
    if (User.getKey() != Expr)
      continue;
    Output.append(User.getValue().begin(), User.getValue().end());
    return true;
  }
  return false;
}

bool MockTildeExpressionResolver::ResolvePartial(StringRef Expr,
                                                 StringSet<> &Output) {
  Output.clear();

  assert(!llvm::any_of(
      Expr, [](char c) { return llvm::sys::path::is_separator(c); }));
  assert(Expr.empty() || Expr[0] == '~');
  Expr = Expr.drop_front();

  SmallString<16> QualifiedName("~");
  for (const auto &User : UserDirectories) {
    if (!User.getKey().startswith(Expr))
      continue;
    QualifiedName.resize(1);
    QualifiedName.append(User.getKey().begin(), User.getKey().end());
    Output.insert(QualifiedName);
  }

  return !Output.empty();
}
