//===-- RenderScriptScriptGroup.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RenderScriptScriptGroup_h_
#define liblldb_RenderScriptScriptGroup_h_

#include "lldb/Interpreter/CommandInterpreter.h"

lldb::CommandObjectSP NewCommandObjectRenderScriptScriptGroup(
    lldb_private::CommandInterpreter &interpreter);

#endif // liblldb_RenderScriptScriptGroup_h_
