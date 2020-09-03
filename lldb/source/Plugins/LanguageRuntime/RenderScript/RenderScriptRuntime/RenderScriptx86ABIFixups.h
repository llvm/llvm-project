//===-- RenderScriptx86ABIFixups.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_RENDERSCRIPT_RENDERSCRIPTRUNTIME_RENDERSCRIPTX86ABIFIXUPS_H
#define LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_RENDERSCRIPT_RENDERSCRIPTRUNTIME_RENDERSCRIPTX86ABIFIXUPS_H

#include "llvm/IR/Module.h"

namespace lldb_private {
namespace lldb_renderscript {

bool fixupX86FunctionCalls(llvm::Module &module);

bool fixupX86_64FunctionCalls(llvm::Module &module);
}
}
#endif
