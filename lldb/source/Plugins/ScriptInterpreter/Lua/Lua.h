//===-- ScriptInterpreterLua.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Lua_h_
#define liblldb_Lua_h_

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "lua.hpp"

#include <mutex>

namespace lldb_private {

extern "C" {
int luaopen_lldb(lua_State *L);
}

class Lua {
public:
  Lua() : m_lua_state(luaL_newstate()) {
    assert(m_lua_state);
    luaL_openlibs(m_lua_state);
    luaopen_lldb(m_lua_state);
  }

  ~Lua() {
    assert(m_lua_state);
    luaL_openlibs(m_lua_state);
  }

  llvm::Error Run(llvm::StringRef buffer);

private:
  std::mutex m_mutex;
  lua_State *m_lua_state;
};

} // namespace lldb_private

#endif // liblldb_Lua_h_
