//===-- Lua.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Lua.h"
#include "llvm/Support/FormatVariadic.h"

using namespace lldb_private;

llvm::Error Lua::Run(llvm::StringRef buffer) {
  std::lock_guard<std::mutex> lock(m_mutex);
  int error =
      luaL_loadbuffer(m_lua_state, buffer.data(), buffer.size(), "buffer") ||
      lua_pcall(m_lua_state, 0, 0, 0);
  if (!error)
    return llvm::Error::success();

  llvm::Error e = llvm::make_error<llvm::StringError>(
      llvm::formatv("{0}\n", lua_tostring(m_lua_state, -1)),
      llvm::inconvertibleErrorCode());
  // Pop error message from the stack.
  lua_pop(m_lua_state, 1);
  return e;
}
