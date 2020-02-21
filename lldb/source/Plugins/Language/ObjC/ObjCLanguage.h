//===-- ObjCLanguage.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGE_OBJC_OBJCLANGUAGE_H
#define LLDB_SOURCE_PLUGINS_LANGUAGE_OBJC_OBJCLANGUAGE_H

#include <cstring>
#include <vector>

#include "Plugins/Language/ClangCommon/ClangHighlighter.h"
#include "lldb/Target/Language.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class ObjCLanguage : public Language {
  ClangHighlighter m_highlighter;

public:
  class MethodName {
  public:
    enum Type { eTypeUnspecified, eTypeClassMethod, eTypeInstanceMethod };

    MethodName()
        : m_full(), m_class(), m_category(), m_selector(),
          m_type(eTypeUnspecified), m_category_is_valid(false) {}

    MethodName(const char *name, bool strict)
        : m_full(), m_class(), m_category(), m_selector(),
          m_type(eTypeUnspecified), m_category_is_valid(false) {
      SetName(name, strict);
    }
    MethodName(llvm::StringRef name, bool strict)
        : m_full(), m_class(), m_category(), m_selector(),
          m_type(eTypeUnspecified), m_category_is_valid(false) {
      SetName(name, strict);
    }

    void Clear();

    bool IsValid(bool strict) const {
      // If "strict" is true, the name must have everything specified including
      // the leading "+" or "-" on the method name
      if (strict && m_type == eTypeUnspecified)
        return false;
      // Other than that, m_full will only be filled in if the objective C
      // name is valid.
      return (bool)m_full;
    }

    bool HasCategory() { return !GetCategory().IsEmpty(); }

    Type GetType() const { return m_type; }

    ConstString GetFullName() const { return m_full; }

    ConstString GetFullNameWithoutCategory(bool empty_if_no_category);

    bool SetName(const char *name, bool strict);
    bool SetName(llvm::StringRef name, bool strict);

    ConstString GetClassName();

    ConstString GetClassNameWithCategory();

    ConstString GetCategory();

    ConstString GetSelector();

  protected:
    ConstString
        m_full; // Full name:   "+[NSString(my_additions) myStringWithCString:]"
    ConstString m_class; // Class name:  "NSString"
    ConstString
        m_class_category;   // Class with category: "NSString(my_additions)"
    ConstString m_category; // Category:    "my_additions"
    ConstString m_selector; // Selector:    "myStringWithCString:"
    Type m_type;
    bool m_category_is_valid;
  };

  ObjCLanguage() = default;

  ~ObjCLanguage() override = default;

  lldb::LanguageType GetLanguageType() const override {
    return lldb::eLanguageTypeObjC;
  }

  // Get all possible names for a method. Examples:
  // If method_name is "+[NSString(my_additions) myStringWithCString:]"
  //   variant_names[0] => "+[NSString myStringWithCString:]"
  // If name is specified without the leading '+' or '-' like
  // "[NSString(my_additions) myStringWithCString:]"
  //  variant_names[0] => "+[NSString(my_additions) myStringWithCString:]"
  //  variant_names[1] => "-[NSString(my_additions) myStringWithCString:]"
  //  variant_names[2] => "+[NSString myStringWithCString:]"
  //  variant_names[3] => "-[NSString myStringWithCString:]"
  std::vector<ConstString>
  GetMethodNameVariants(ConstString method_name) const override;

  lldb::TypeCategoryImplSP GetFormatters() override;

  std::vector<ConstString>
  GetPossibleFormattersMatches(ValueObject &valobj,
                               lldb::DynamicValueType use_dynamic) override;

  std::unique_ptr<TypeScavenger> GetTypeScavenger() override;

  bool GetFormatterPrefixSuffix(ValueObject &valobj, ConstString type_hint,
                                std::string &prefix,
                                std::string &suffix) override;

  bool IsNilReference(ValueObject &valobj) override;

  bool IsSourceFile(llvm::StringRef file_path) const override;

  const Highlighter *GetHighlighter() const override { return &m_highlighter; }

  // Static Functions
  static void Initialize();

  static void Terminate();

  static lldb_private::Language *CreateInstance(lldb::LanguageType language);

  static lldb_private::ConstString GetPluginNameStatic();

  static bool IsPossibleObjCMethodName(const char *name) {
    if (!name)
      return false;
    bool starts_right = (name[0] == '+' || name[0] == '-') && name[1] == '[';
    bool ends_right = (name[strlen(name) - 1] == ']');
    return (starts_right && ends_right);
  }

  static bool IsPossibleObjCSelector(const char *name) {
    if (!name)
      return false;

    if (strchr(name, ':') == nullptr)
      return true;
    else if (name[strlen(name) - 1] == ':')
      return true;
    else
      return false;
  }

  // PluginInterface protocol
  ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_LANGUAGE_OBJC_OBJCLANGUAGE_H
