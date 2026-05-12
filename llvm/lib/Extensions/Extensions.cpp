//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Anchor for pass plugins.
//
//===----------------------------------------------------------------------===//

#include "llvm/Plugins/PassPlugin.h"

#define HANDLE_EXTENSION(Ext)                                                  \
		llvm::PassPluginLibraryInfo get##Ext##PluginInfo();
#include "llvm/Support/Extension.def"
#undef HANDLE_EXTENSION

namespace llvm::details {
void extensions_anchor() {
#define HANDLE_EXTENSION(Ext)                                                  \
			get##Ext##PluginInfo();
#include "llvm/Support/Extension.def"
}
} // namespace llvm::details
