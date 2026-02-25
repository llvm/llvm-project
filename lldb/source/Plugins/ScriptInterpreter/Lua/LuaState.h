//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_LUA_LUASTATE_H
#define LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_LUA_LUASTATE_H

#include "lldb/Core/StructuredDataImpl.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "lua.hpp"

namespace lldb_private {

extern "C" {
int luaopen_lldb(lua_State *L);
}

class LuaState {
public:
  LuaState();
  ~LuaState();

  llvm::Error Run(llvm::StringRef buffer);
  llvm::Error RegisterBreakpointCallback(void *baton, const char *body);
  llvm::Expected<bool>
  CallBreakpointCallback(void *baton, lldb::StackFrameSP stop_frame_sp,
                         lldb::BreakpointLocationSP bp_loc_sp,
                         StructuredData::ObjectSP extra_args_sp);
  llvm::Error RegisterWatchpointCallback(void *baton, const char *body);
  llvm::Expected<bool> CallWatchpointCallback(void *baton,
                                              lldb::StackFrameSP stop_frame_sp,
                                              lldb::WatchpointSP wp_sp);
  llvm::Error LoadModule(llvm::StringRef filename);
  llvm::Error CheckSyntax(llvm::StringRef buffer);
  llvm::Error ChangeIO(FILE *out, FILE *err);

private:
  lua_State *m_lua_state;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_LUA_LUA_H
