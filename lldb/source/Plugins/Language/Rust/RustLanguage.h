//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGE_RUST_RUSTLANGUAGE_H
#define LLDB_SOURCE_PLUGINS_LANGUAGE_RUST_RUSTLANGUAGE_H
#include "lldb/Target/Language.h"

#include "llvm/ADT/StringRef.h"

#include "lldb/Utility/ConstString.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

namespace formatters {
SyntheticChildrenFrontEnd *
RustVectorSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                   lldb::ValueObjectSP valobj_sp);
}

class RustLanguage : public Language {
public:
  RustLanguage() = default;

  ~RustLanguage() override = default;

  lldb::LanguageType GetLanguageType() const override {
    return lldb::eLanguageTypeRust;
  }

  lldb::TypeCategoryImplSP GetFormatters() override;

  static void Initialize();

  static void Terminate();

  static lldb_private::Language *CreateInstance(lldb::LanguageType language);

  static llvm::StringRef GetPluginNameStatic();

  llvm::StringRef GetPluginName() override;

  uint32_t GetPluginVersion();

  bool IsSourceFile(llvm::StringRef file_path) const override;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_LANGUAGE_RUST_RUSTLANGUAGE_H
