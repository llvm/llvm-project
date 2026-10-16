//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SwiftTreeSitterHighlighter.h"
#include "HighlightQuery.h"
#include "lldb/Target/Language.h"

using namespace lldb_private;

LLDB_PLUGIN_DEFINE_ADV(SwiftTreeSitterHighlighter, HighlighterTreeSitterSwift)

extern "C" {
const TSLanguage *tree_sitter_swift();
}

const TSLanguage *SwiftTreeSitterHighlighter::GetLanguage() const {
  return tree_sitter_swift();
}

llvm::StringRef SwiftTreeSitterHighlighter::GetHighlightQuery() const {
  return highlight_query;
}

Highlighter *
SwiftTreeSitterHighlighter::CreateInstance(lldb::LanguageType language) {
  if (language == lldb::eLanguageTypeSwift)
    return new SwiftTreeSitterHighlighter();
  return nullptr;
}

void SwiftTreeSitterHighlighter::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), GetPluginNameStatic(),
                                CreateInstance);
}

void SwiftTreeSitterHighlighter::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}
