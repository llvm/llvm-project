//===-- SwiftLanguage.h -----------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SwiftLanguage_h_
#define liblldb_SwiftLanguage_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Language.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class SwiftLanguage : public Language {
public:
  virtual ~SwiftLanguage() = default;

  SwiftLanguage() = default;

  lldb::LanguageType GetLanguageType() const override {
    return lldb::eLanguageTypeSwift;
  }

  bool IsTopLevelFunction(Function &function) override;

  virtual lldb::TypeCategoryImplSP GetFormatters() override;

  HardcodedFormatters::HardcodedSummaryFinder GetHardcodedSummaries() override;

  HardcodedFormatters::HardcodedSyntheticFinder
  GetHardcodedSynthetics() override;

  bool IsSourceFile(llvm::StringRef file_path) const override;

  std::vector<ConstString>
  GetPossibleFormattersMatches(ValueObject &valobj,
                               lldb::DynamicValueType use_dynamic) override;

  virtual lldb_private::formatters::StringPrinter::EscapingHelper
      GetStringPrinterEscapingHelper(
          lldb_private::formatters::StringPrinter::GetPrintableElementType)
          override;

  std::unique_ptr<TypeScavenger> GetTypeScavenger() override;

  const char *GetLanguageSpecificTypeLookupHelp() override;

  bool GetFormatterPrefixSuffix(ValueObject &valobj, ConstString type_hint,
                                std::string &prefix,
                                std::string &suffix) override;

  DumpValueObjectOptions::DeclPrintingHelper GetDeclPrintingHelper() override;

  LazyBool IsLogicalTrue(ValueObject &valobj, Status &error) override;

  bool IsUninitializedReference(ValueObject &valobj) override;

  bool GetFunctionDisplayName(const SymbolContext *sc,
                              const ExecutionContext *exe_ctx,
                              FunctionNameRepresentation representation,
                              Stream &s) override;

  void GetExceptionResolverDescription(bool catch_on, bool throw_on,
                                       Stream &s) override;

  //------------------------------------------------------------------
  // Static Functions
  //------------------------------------------------------------------
  static void Initialize();

  static void Terminate();

  static lldb_private::Language *CreateInstance(lldb::LanguageType language);

  static lldb_private::ConstString GetPluginNameStatic();

  //------------------------------------------------------------------
  // PluginInterface protocol
  //------------------------------------------------------------------
  virtual ConstString GetPluginName() override;

  virtual uint32_t GetPluginVersion() override;
};

} // namespace lldb_private

#endif // liblldb_SwiftLanguage_h_
