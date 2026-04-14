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
#include "lldb/Utility/ConstString.h"
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

  std::vector<Language::MethodNameVariant>
  GetMethodNameVariants(ConstString method_name) const override;

  lldb::TypeCategoryImplSP GetFormatters() override;

  HardcodedFormatters::HardcodedSummaryFinder GetHardcodedSummaries() override;

  HardcodedFormatters::HardcodedSyntheticFinder
  GetHardcodedSynthetics() override;

  bool IsSourceFile(llvm::StringRef file_path) const override;

  std::vector<FormattersMatchCandidate>
  GetPossibleFormattersMatches(ValueObject &valobj,
                               lldb::DynamicValueType use_dynamic) override;

  std::unique_ptr<TypeScavenger> GetTypeScavenger() override;

  const char *GetLanguageSpecificTypeLookupHelp() override;

  std::pair<llvm::StringRef, llvm::StringRef>
  GetFormatterPrefixSuffix(llvm::StringRef type_hint) override;

  DumpValueObjectOptions::DeclPrintingHelper GetDeclPrintingHelper() override;

  LazyBool IsLogicalTrue(ValueObject &valobj, Status &error) override;

  bool IsUninitializedReference(ValueObject &valobj) override;

  bool GetFunctionDisplayName(const SymbolContext &sc,
                              const ExecutionContext *exe_ctx,
                              FunctionNameRepresentation representation,
                              Stream &s) override;

  std::string GetDemangledFunctionName(const SymbolContext &sc,
                                       const ExecutionContext *exe_ctx);

  /// Returns the name of function up to the first generic or opening
  /// parenthesis.
  ///
  /// The following function will have the name "foo":
  /// \code{.swift}
  /// func foo<T>(bar: T) {}
  /// \endcode
  ///
  /// \param sc        The associated SymbolContext.
  /// \param exe_ctx   The associated ExecutionContext.
  /// \returns The name of a function as an std::string.
  std::string GetFunctionName(const SymbolContext &sc,
                              const ExecutionContext *exe_ctx);

  std::string GetFunctionTemplateArguments(const SymbolContext &sc,
                                           const ExecutionContext *exe_ctx);

  /// Returns the arguments of a function call with its generics if any.
  ///
  /// Calling GetFunctionDisplayArgs on the following function call will return
  /// "<Int>(bar=1)"
  /// \code{.swift}
  /// func foo<T>(bar: T) {}
  /// foo(1)
  /// \endcode
  ///
  /// \param sc        The associated SymbolContext.
  /// \param args      The VariableList that are passed to the function.
  /// \param exe_ctx   The associated ExecutionContext.
  /// \returns The generics and arguments of a function call as an std::string.
  std::string
  GetFunctionDisplayArgs(const SymbolContext &sc, VariableList &args,
                         const lldb_private::ExecutionContext *exe_ctx);

  void GetExceptionResolverDescription(bool catch_on, bool throw_on,
                                       Stream &s) override;

  ConstString
  GetDemangledFunctionNameWithoutArguments(Mangled mangled) const override;

  /// Returns whether two SymbolContexts correspond to funclets of the same
  /// async function.
  /// If either SymbolContext is not a funclet, nullopt is returned.
  std::optional<bool>
  AreEqualForFrameComparison(const SymbolContext &sc1,
                             const SymbolContext &sc2) const override;

  std::pair<lldb::FunctionNameType, std::optional<ConstString>>
  GetFunctionNameInfo(ConstString name) const override;

  std::string GetParentNameIfClosure(Function &func) const override;
  //------------------------------------------------------------------
  // Static Functions
  //------------------------------------------------------------------
  static void Initialize();

  static void Terminate();

  static lldb_private::Language *CreateInstance(lldb::LanguageType language);

  static llvm::StringRef GetPluginNameStatic() { return "swift"; }

  bool SymbolNameFitsToLanguage(const Mangled &mangled) const override;

  llvm::StringRef GetInstanceVariableName() override { return "self"; }

  bool HandleFrameFormatVariable(const SymbolContext &sc,
                                 const ExecutionContext *exe_ctx,
                                 FormatEntity::Entry::Type type,
                                 Stream &s) override;

  FormatEntity::Entry GetFunctionNameFormat() const override;

  /// Override that skips breakpoints inside await resume ("Q") async funclets.
  void FilterForLineBreakpoints(
      llvm::SmallVectorImpl<SymbolContext> &) const override;

  //------------------------------------------------------------------
  // PluginInterface protocol
  //------------------------------------------------------------------
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

private:
  static void DebuggerInitialize(Debugger &);
};

} // namespace lldb_private

#endif // liblldb_SwiftLanguage_h_
