//===-- Lua.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Lua.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Utility/FileSpec.h"
#include "llvm/Support/FormatVariadic.h"

using namespace lldb_private;
using namespace lldb;

llvm::Error Lua::Run(llvm::StringRef buffer) {
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

llvm::Error Lua::LoadModule(llvm::StringRef filename) {
  FileSpec file(filename);
  if (!FileSystem::Instance().Exists(file)) {
    return llvm::make_error<llvm::StringError>("invalid path",
                                               llvm::inconvertibleErrorCode());
  }

  ConstString module_extension = file.GetFileNameExtension();
  if (module_extension != ".lua") {
    return llvm::make_error<llvm::StringError>("invalid extension",
                                               llvm::inconvertibleErrorCode());
  }

  int error = luaL_loadfile(m_lua_state, filename.data()) ||
              lua_pcall(m_lua_state, 0, 1, 0);
  if (error) {
    llvm::Error e = llvm::make_error<llvm::StringError>(
        llvm::formatv("{0}\n", lua_tostring(m_lua_state, -1)),
        llvm::inconvertibleErrorCode());
    // Pop error message from the stack.
    lua_pop(m_lua_state, 1);
    return e;
  }

  ConstString module_name = file.GetFileNameStrippingExtension();
  lua_setglobal(m_lua_state, module_name.GetCString());
  return llvm::Error::success();
}

llvm::Error Lua::ChangeIO(FILE *out, FILE *err) {
  assert(out != nullptr);
  assert(err != nullptr);

  lua_getglobal(m_lua_state, "io");

  lua_getfield(m_lua_state, -1, "stdout");
  if (luaL_Stream *s = static_cast<luaL_Stream *>(
          luaL_testudata(m_lua_state, -1, LUA_FILEHANDLE))) {
    s->f = out;
    lua_pop(m_lua_state, 1);
  } else {
    lua_pop(m_lua_state, 2);
    return llvm::make_error<llvm::StringError>("could not get stdout",
                                               llvm::inconvertibleErrorCode());
  }

  lua_getfield(m_lua_state, -1, "stdout");
  if (luaL_Stream *s = static_cast<luaL_Stream *>(
          luaL_testudata(m_lua_state, -1, LUA_FILEHANDLE))) {
    s->f = out;
    lua_pop(m_lua_state, 1);
  } else {
    lua_pop(m_lua_state, 2);
    return llvm::make_error<llvm::StringError>("could not get stderr",
                                               llvm::inconvertibleErrorCode());
  }

  lua_pop(m_lua_state, 1);
  return llvm::Error::success();
}
