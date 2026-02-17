//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLDB_SOURCE_PLUGINS_HIGHLIGHTER_TREESITTER_SWIFT_SWIFTTREESITTERHIGHLIGHTER_H
#define LLDB_SOURCE_PLUGINS_HIGHLIGHTER_TREESITTER_SWIFT_SWIFTTREESITTERHIGHLIGHTER_H

#include "../TreeSitterHighlighter.h"
#include "llvm/ADT/StringRef.h"

namespace lldb_private {

class SwiftTreeSitterHighlighter : public TreeSitterHighlighter {
public:
  SwiftTreeSitterHighlighter() = default;
  ~SwiftTreeSitterHighlighter() override = default;

  llvm::StringRef GetName() const override { return "tree-sitter-swift"; }

  static Highlighter *CreateInstance(lldb::LanguageType language);

  static void Terminate();
  static void Initialize();

  static llvm::StringRef GetPluginNameStatic() {
    return "Tree-sitter Swift Highlighter";
  }
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

protected:
  const TSLanguage *GetLanguage() const override;
  llvm::StringRef GetHighlightQuery() const override;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_HIGHLIGHTER_TREESITTER_SWIFT_SWIFTTREESITTERHIGHLIGHTER_H
