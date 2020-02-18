//===-- RenderScriptScriptGroup.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_RENDERSCRIPT_RENDERSCRIPTRUNTIME_RENDERSCRIPTSCRIPTGROUP_H
#define LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_RENDERSCRIPT_RENDERSCRIPTRUNTIME_RENDERSCRIPTSCRIPTGROUP_H

#include "lldb/Interpreter/CommandInterpreter.h"

lldb::CommandObjectSP NewCommandObjectRenderScriptScriptGroup(
    lldb_private::CommandInterpreter &interpreter);

#endif // LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_RENDERSCRIPT_RENDERSCRIPTRUNTIME_RENDERSCRIPTSCRIPTGROUP_H
