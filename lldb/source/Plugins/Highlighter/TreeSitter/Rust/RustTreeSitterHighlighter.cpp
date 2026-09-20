//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RustTreeSitterHighlighter.h"
#include "HighlightQuery.h"
#include "lldb/Target/Language.h"

using namespace lldb_private;

extern "C" {
const TSLanguage *tree_sitter_rust();
}

LLDB_PLUGIN_DEFINE_ADV(RustTreeSitterHighlighter, HighlighterTreeSitterRust)

const TSLanguage *RustTreeSitterHighlighter::GetLanguage() const {
  return tree_sitter_rust();
}

llvm::StringRef RustTreeSitterHighlighter::GetHighlightQuery() const {
  return highlight_query;
}

Highlighter *
RustTreeSitterHighlighter::CreateInstance(lldb::LanguageType language) {
  if (language == lldb::eLanguageTypeRust)
    return new RustTreeSitterHighlighter();
  return nullptr;
}

void RustTreeSitterHighlighter::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), GetPluginNameStatic(),
                                CreateInstance);
}

void RustTreeSitterHighlighter::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}
