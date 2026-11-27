//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGE_TREESITTERCOMMON_RUSTTREESITTERHIGHLIGHTER_H
#define LLDB_SOURCE_PLUGINS_LANGUAGE_TREESITTERCOMMON_RUSTTREESITTERHIGHLIGHTER_H

#include "../TreeSitterHighlighter.h"
#include "llvm/ADT/StringRef.h"

namespace lldb_private {

class RustTreeSitterHighlighter : public TreeSitterHighlighter {
public:
  RustTreeSitterHighlighter() = default;
  ~RustTreeSitterHighlighter() override = default;

  llvm::StringRef GetName() const override { return "tree-sitter-rust"; }

  static Highlighter *CreateInstance(lldb::LanguageType language);

  static void Terminate();
  static void Initialize();

  static llvm::StringRef GetPluginNameStatic() {
    return "Tree-sitter Rust Highlighter";
  }
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

protected:
  const TSLanguage *GetLanguage() const override;
  llvm::StringRef GetHighlightQuery() const override;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_LANGUAGE_TREESITTERCOMMON_RUSTTREESITTERHIGHLIGHTER_H
