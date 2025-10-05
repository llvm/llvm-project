//===- AttrOrTypeDefGen.cpp - MLIR AttrOrType definitions generator -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AttrOrTypeFormatGen.h"
#include "CppGenUtilities.h"
#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Class.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Interfaces.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TableGen/CodeGenHelpers.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "mlir-tblgen-attrortypedefgen"

using namespace mlir;
using namespace mlir::tblgen;
using llvm::Record;
using llvm::RecordKeeper;

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Find all the AttrOrTypeDef for the specified dialect. If no dialect
/// specified and can only find one dialect's defs, use that.
static void collectAllDefs(StringRef selectedDialect,
                           ArrayRef<const Record *> records,
                           SmallVectorImpl<AttrOrTypeDef> &resultDefs) {
  // Nothing to do if no defs were found.
  if (records.empty())
    return;

  auto defs = llvm::map_range(
      records, [&](const Record *rec) { return AttrOrTypeDef(rec); });
  if (selectedDialect.empty()) {
    // If a dialect was not specified, ensure that all found defs belong to the
    // same dialect.
    if (!llvm::all_equal(llvm::map_range(
            defs, [](const auto &def) { return def.getDialect(); }))) {
      llvm::PrintFatalError("defs belonging to more than one dialect. Must "
                            "select one via '--(attr|type)defs-dialect'");
    }
    resultDefs.assign(defs.begin(), defs.end());
  } else {
    // Otherwise, generate the defs that belong to the selected dialect.
    auto dialectDefs = llvm::make_filter_range(defs, [&](const auto &def) {
      return def.getDialect().getName() == selectedDialect;
    });
    resultDefs.assign(dialectDefs.begin(), dialectDefs.end());
  }
}

//===----------------------------------------------------------------------===//
// DefGen
//===----------------------------------------------------------------------===//

namespace {
class DefGen {
public:
  /// Create the attribute or type class.
  DefGen(const AttrOrTypeDef &def);

  void emitDecl(raw_ostream &os) const {
    if (storageCls && def.genStorageClass()) {
      llvm::NamespaceEmitter ns(os, def.getStorageNamespace());
      os << "struct " << def.getStorageClassName() << ";\n";
    }
    defCls.writeDeclTo(os);
  }
  void emitDef(raw_ostream &os) const {
    if (storageCls && def.genStorageClass()) {
      llvm::NamespaceEmitter ns(os, def.getStorageNamespace());
      storageCls->writeDeclTo(os); // everything is inline
    }
    defCls.writeDefTo(os);
  }

private:
  /// Add traits from the TableGen definition to the class.
  void createParentWithTraits();
  /// Emit top-level declarations: using declarations and any extra class
  /// declarations.
  void emitTopLevelDeclarations();
  /// Emit the function that returns the type or attribute name.
  void emitName();
  /// Emit the dialect name as a static member variable.
  void emitDialectName();
  /// Emit attribute or type builders.
  void emitBuilders();
  /// Emit a verifier declaration for custom verification (impl. provided by
  /// the users).
  void emitVerifierDecl();
  /// Emit a verifier that checks type constraints.
  void emitInvariantsVerifierImpl();
  /// Emit an entry poiunt for verification that calls the invariants and
  /// custom verifier.
  void emitInvariantsVerifier(bool hasImpl, bool hasCustomVerifier);
  /// Emit parsers and printers.
  void emitParserPrinter();
  /// Emit parameter accessors, if required.
  void emitAccessors();
  /// Emit interface methods.
  void emitInterfaceMethods();

  //===--------------------------------------------------------------------===//
  // Builder Emission

  /// Emit the default builder `Attribute::get`
  void emitDefaultBuilder();
  /// Emit the checked builder `Attribute::getChecked`
  void emitCheckedBuilder();
  /// Emit a custom builder.
  void emitCustomBuilder(const AttrOrTypeBuilder &builder);
  /// Emit a checked custom builder.
  void emitCheckedCustomBuilder(const AttrOrTypeBuilder &builder);

  //===--------------------------------------------------------------------===//
  // Interface Method Emission

  /// Emit methods for a trait.
  void emitTraitMethods(const InterfaceTrait &trait);
  /// Emit a trait method.
  void emitTraitMethod(const InterfaceMethod &method);

  //===--------------------------------------------------------------------===//
  // OpAsm{Type,Attr}Interface Default Method Emission

  /// Emit 'getAlias' method using mnemonic as alias.
  void emitMnemonicAliasMethod();

  //===--------------------------------------------------------------------===//
  // Storage Class Emission
  void emitStorageClass();
  /// Generate the storage class constructor.
  void emitStorageConstructor();
  /// Emit the key type `KeyTy`.
  void emitKeyType();
  /// Emit the equality comparison operator.
  void emitEquals();
  /// Emit the key hash function.
  void emitHashKey();
  /// Emit the function to construct the storage class.
  void emitConstruct();

  //===--------------------------------------------------------------------===//
  // Utility Function Declarations

  /// Get the method parameters for a def builder, where the first several
  /// parameters may be different.
  SmallVector<MethodParameter>
  getBuilderParams(std::initializer_list<MethodParameter> prefix) const;

  //===--------------------------------------------------------------------===//
  // Class fields

  /// The attribute or type definition.
  const AttrOrTypeDef &def;
  /// The list of attribute or type parameters.
  ArrayRef<AttrOrTypeParameter> params;
  /// The attribute or type class.
  Class defCls;
  /// An optional attribute or type storage class. The storage class will
  /// exist if and only if the def has more than zero parameters.
  std::optional<Class> storageCls;

  /// The C++ base value of the def, either "Attribute" or "Type".
  StringRef valueType;
  /// The prefix/suffix of the TableGen def name, either "Attr" or "Type".
  StringRef defType;
};
} // namespace

DefGen::DefGen(const AttrOrTypeDef &def)
    : def(def), params(def.getParameters()), defCls(def.getCppClassName()),
      valueType(isa<AttrDef>(def) ? "Attribute" : "Type"),
      defType(isa<AttrDef>(def) ? "Attr" : "Type") {
  // Check that all parameters have names.
  for (const AttrOrTypeParameter &param : def.getParameters())
    if (param.isAnonymous())
      llvm::PrintFatalError("all parameters must have a name");

  // If a storage class is needed, create one.
  if (def.getNumParameters() > 0)
    storageCls.emplace(def.getStorageClassName(), /*isStruct=*/true);

  // Create the parent class with any indicated traits.
  createParentWithTraits();
  // Emit top-level declarations.
  emitTopLevelDeclarations();
  // Emit builders for defs with parameters
  if (storageCls)
    emitBuilders();
  // Emit the type name.
  emitName();
  // Emit the dialect name.
  emitDialectName();
  // Emit verification of type constraints.
  bool genVerifyInvariantsImpl = def.genVerifyInvariantsImpl();
  if (storageCls && genVerifyInvariantsImpl)
    emitInvariantsVerifierImpl();
  // Emit the custom verifier (written by the user).
  bool genVerifyDecl = def.genVerifyDecl();
  if (storageCls && genVerifyDecl)
    emitVerifierDecl();
  // Emit the "verifyInvariants" function if there is any verification at all.
  if (storageCls)
    emitInvariantsVerifier(genVerifyInvariantsImpl, genVerifyDecl);
  // Emit the mnemonic, if there is one, and any associated parser and printer.
  if (def.getMnemonic())
    emitParserPrinter();
  // Emit accessors
  if (def.genAccessors())
    emitAccessors();
  // Emit trait interface methods
  emitInterfaceMethods();
  // Emit OpAsm{Type,Attr}Interface default methods
  if (def.genMnemonicAlias())
    emitMnemonicAliasMethod();
  defCls.finalize();
  // Emit a storage class if one is needed
  if (storageCls && def.genStorageClass())
    emitStorageClass();
}

void DefGen::createParentWithTraits() {
  ParentClass defParent(strfmt("::mlir::{0}::{1}Base", valueType, defType));
  defParent.addTemplateParam(def.getCppClassName());
  defParent.addTemplateParam(def.getCppBaseClassName());
  defParent.addTemplateParam(storageCls
                                 ? strfmt("{0}::{1}", def.getStorageNamespace(),
                                          def.getStorageClassName())
                                 : strfmt("::mlir::{0}Storage", valueType));
  SmallVector<std::string> traitNames =
      llvm::to_vector(llvm::map_range(def.getTraits(), [](auto &trait) {
        return isa<NativeTrait>(&trait)
                   ? cast<NativeTrait>(&trait)->getFullyQualifiedTraitName()
                   : cast<InterfaceTrait>(&trait)->getFullyQualifiedTraitName();
      }));
  for (auto &traitName : traitNames)
    defParent.addTemplateParam(traitName);

  // Add OpAsmInterface::Trait if we automatically generate mnemonic alias
  // method.
  std::string opAsmInterfaceTraitName =
      strfmt("::mlir::OpAsm{0}Interface::Trait", defType);
  if (def.genMnemonicAlias() &&
      !llvm::is_contained(traitNames, opAsmInterfaceTraitName)) {
    defParent.addTemplateParam(opAsmInterfaceTraitName);
  }
  defCls.addParent(std::move(defParent));
}

/// Include declarations specified on NativeTrait
static std::string formatExtraDeclarations(const AttrOrTypeDef &def) {
  SmallVector<StringRef> extraDeclarations;
  // Include extra class declarations from NativeTrait
  for (const auto &trait : def.getTraits()) {
    if (auto *attrOrTypeTrait = dyn_cast<tblgen::NativeTrait>(&trait)) {
      StringRef value = attrOrTypeTrait->getExtraConcreteClassDeclaration();
      if (value.empty())
        continue;
      extraDeclarations.push_back(value);
    }
  }
  if (std::optional<StringRef> extraDecl = def.getExtraDecls()) {
    extraDeclarations.push_back(*extraDecl);
  }
  return llvm::join(extraDeclarations, "\n");
}

/// Extra class definitions have a `$cppClass` substitution that is to be
/// replaced by the C++ class name.
static std::string formatExtraDefinitions(const AttrOrTypeDef &def) {
  SmallVector<StringRef> extraDefinitions;
  // Include extra class definitions from NativeTrait
  for (const auto &trait : def.getTraits()) {
    if (auto *attrOrTypeTrait = dyn_cast<tblgen::NativeTrait>(&trait)) {
      StringRef value = attrOrTypeTrait->getExtraConcreteClassDefinition();
      if (value.empty())
        continue;
      extraDefinitions.push_back(value);
    }
  }
  if (std::optional<StringRef> extraDef = def.getExtraDefs()) {
    extraDefinitions.push_back(*extraDef);
  }
  FmtContext ctx = FmtContext().addSubst("cppClass", def.getCppClassName());
  return tgfmt(llvm::join(extraDefinitions, "\n"), &ctx).str();
}

void DefGen::emitTopLevelDeclarations() {
  // Inherit constructors from the attribute or type class.
  defCls.declare<VisibilityDeclaration>(Visibility::Public);
  defCls.declare<UsingDeclaration>("Base::Base");

  // Emit the extra declarations first in case there's a definition in there.
  std::string extraDecl = formatExtraDeclarations(def);
  std::string extraDef = formatExtraDefinitions(def);
  defCls.declare<ExtraClassDeclaration>(std::move(extraDecl),
                                        std::move(extraDef));
}

void DefGen::emitName() {
  StringRef name;
  if (auto *attrDef = dyn_cast<AttrDef>(&def)) {
    name = attrDef->getAttrName();
  } else {
    auto *typeDef = cast<TypeDef>(&def);
    name = typeDef->getTypeName();
  }
  std::string nameDecl =
      strfmt("static constexpr ::llvm::StringLiteral name = \"{0}\";\n", name);
  defCls.declare<ExtraClassDeclaration>(std::move(nameDecl));
}

void DefGen::emitDialectName() {
  std::string decl =
      strfmt("static constexpr ::llvm::StringLiteral dialectName = \"{0}\";\n",
             def.getDialect().getName());
  defCls.declare<ExtraClassDeclaration>(std::move(decl));
}

void DefGen::emitBuilders() {
  if (!def.skipDefaultBuilders()) {
    emitDefaultBuilder();
    if (def.genVerifyDecl() || def.genVerifyInvariantsImpl())
      emitCheckedBuilder();
  }
  for (auto &builder : def.getBuilders()) {
    emitCustomBuilder(builder);
    if (def.genVerifyDecl() || def.genVerifyInvariantsImpl())
      emitCheckedCustomBuilder(builder);
  }
}

void DefGen::emitVerifierDecl() {
  defCls.declareStaticMethod(
      "::llvm::LogicalResult", "verify",
      getBuilderParams({{"::llvm::function_ref<::mlir::InFlightDiagnostic()>",
                         "emitError"}}));
}

static const char *const patternParameterVerificationCode = R"(
if (!({0})) {
  emitError() << "failed to verify '{1}': {2}";
  return ::mlir::failure();
}
)";

void DefGen::emitInvariantsVerifierImpl() {
  SmallVector<MethodParameter> builderParams = getBuilderParams(
      {{"::llvm::function_ref<::mlir::InFlightDiagnostic()>", "emitError"}});
  Method *verifier =
      defCls.addMethod("::llvm::LogicalResult", "verifyInvariantsImpl",
                       Method::Static, builderParams);
  verifier->body().indent();

  // Generate verification for each parameter that is a type constraint.
  for (auto it : llvm::enumerate(def.getParameters())) {
    const AttrOrTypeParameter &param = it.value();
    std::optional<Constraint> constraint = param.getConstraint();
    // No verification needed for parameters that are not type constraints.
    if (!constraint.has_value())
      continue;
    FmtContext ctx;
    // Note: Skip over the first method parameter (`emitError`).
    ctx.withSelf(builderParams[it.index() + 1].getName());
    std::string condition = tgfmt(constraint->getConditionTemplate(), &ctx);
    verifier->body() << formatv(patternParameterVerificationCode, condition,
                                param.getName(), constraint->getSummary())
                     << "\n";
  }
  verifier->body() << "return ::mlir::success();";
}

void DefGen::emitInvariantsVerifier(bool hasImpl, bool hasCustomVerifier) {
  if (!hasImpl && !hasCustomVerifier)
    return;
  defCls.declare<UsingDeclaration>("Base::getChecked");
  SmallVector<MethodParameter> builderParams = getBuilderParams(
      {{"::llvm::function_ref<::mlir::InFlightDiagnostic()>", "emitError"}});
  Method *verifier =
      defCls.addMethod("::llvm::LogicalResult", "verifyInvariants",
                       Method::Static, builderParams);
  verifier->body().indent();

  auto emitVerifierCall = [&](StringRef name) {
    verifier->body() << strfmt("if (::mlir::failed({0}(", name);
    llvm::interleaveComma(
        llvm::map_range(builderParams,
                        [](auto &param) { return param.getName(); }),
        verifier->body());
    verifier->body() << ")))\n";
    verifier->body() << "  return ::mlir::failure();\n";
  };

  if (hasImpl) {
    // Call the verifier that checks the type constraints.
    emitVerifierCall("verifyInvariantsImpl");
  }
  if (hasCustomVerifier) {
    // Call the custom verifier that is provided by the user.
    emitVerifierCall("verify");
  }
  verifier->body() << "return ::mlir::success();";
}

void DefGen::emitParserPrinter() {
  auto *mnemonic = defCls.addStaticMethod<Method::Constexpr>(
      "::llvm::StringLiteral", "getMnemonic");
  mnemonic->body().indent() << strfmt("return {\"{0}\"};", *def.getMnemonic());

  // Declare the parser and printer, if needed.
  bool hasAssemblyFormat = def.getAssemblyFormat().has_value();
  if (!def.hasCustomAssemblyFormat() && !hasAssemblyFormat)
    return;

  // Declare the parser.
  SmallVector<MethodParameter> parserParams;
  parserParams.emplace_back("::mlir::AsmParser &", "odsParser");
  if (isa<AttrDef>(&def))
    parserParams.emplace_back("::mlir::Type", "odsType");
  auto *parser = defCls.addMethod(strfmt("::mlir::{0}", valueType), "parse",
                                  hasAssemblyFormat ? Method::Static
                                                    : Method::StaticDeclaration,
                                  std::move(parserParams));
  // Declare the printer.
  auto props = hasAssemblyFormat ? Method::Const : Method::ConstDeclaration;
  Method *printer =
      defCls.addMethod("void", "print", props,
                       MethodParameter("::mlir::AsmPrinter &", "odsPrinter"));
  // Emit the bodies if we are using the declarative format.
  if (hasAssemblyFormat)
    return generateAttrOrTypeFormat(def, parser->body(), printer->body());
}

void DefGen::emitAccessors() {
  for (auto &param : params) {
    Method *m = defCls.addMethod(
        param.getCppAccessorType(), param.getAccessorName(),
        def.genStorageClass() ? Method::Const : Method::ConstDeclaration);
    // Generate accessor definitions only if we also generate the storage
    // class. Otherwise, let the user define the exact accessor definition.
    if (!def.genStorageClass())
      continue;
    m->body().indent() << "return getImpl()->" << param.getName() << ";";
  }
}

void DefGen::emitInterfaceMethods() {
  for (auto &traitDef : def.getTraits())
    if (auto *trait = dyn_cast<InterfaceTrait>(&traitDef))
      if (trait->shouldDeclareMethods())
        emitTraitMethods(*trait);
}

//===----------------------------------------------------------------------===//
// Builder Emission
//===----------------------------------------------------------------------===//

SmallVector<MethodParameter>
DefGen::getBuilderParams(std::initializer_list<MethodParameter> prefix) const {
  SmallVector<MethodParameter> builderParams;
  builderParams.append(prefix.begin(), prefix.end());
  for (auto &param : params)
    builderParams.emplace_back(param.getCppType(), param.getName());
  return builderParams;
}

void DefGen::emitDefaultBuilder() {
  Method *m = defCls.addStaticMethod(
      def.getCppClassName(), "get",
      getBuilderParams({{"::mlir::MLIRContext *", "context"}}));
  MethodBody &body = m->body().indent();
  auto scope = body.scope("return Base::get(context", ");");
  for (const auto &param : params)
    body << ", std::move(" << param.getName() << ")";
}

void DefGen::emitCheckedBuilder() {
  Method *m = defCls.addStaticMethod(
      def.getCppClassName(), "getChecked",
      getBuilderParams(
          {{"::llvm::function_ref<::mlir::InFlightDiagnostic()>", "emitError"},
           {"::mlir::MLIRContext *", "context"}}));
  MethodBody &body = m->body().indent();
  auto scope = body.scope("return Base::getChecked(emitError, context", ");");
  for (const auto &param : params)
    body << ", std::move(" << param.getName() << ")";
}

static SmallVector<MethodParameter>
getCustomBuilderParams(std::initializer_list<MethodParameter> prefix,
                       const AttrOrTypeBuilder &builder) {
  auto params = builder.getParameters();
  SmallVector<MethodParameter> builderParams;
  builderParams.append(prefix.begin(), prefix.end());
  if (!builder.hasInferredContextParameter())
    builderParams.emplace_back("::mlir::MLIRContext *", "context");
  for (auto &param : params) {
    builderParams.emplace_back(param.getCppType(), *param.getName(),
                               param.getDefaultValue());
  }
  return builderParams;
}

static std::string getSignature(const Method &m) {
  std::string signature;
  llvm::raw_string_ostream os(signature);
  raw_indented_ostream indentedOs(os);
  m.writeDeclTo(indentedOs);
  return signature;
}

static void emitDuplicatedBuilderError(const Method &currentMethod,
                                       StringRef methodName,
                                       const Class &defCls,
                                       const AttrOrTypeDef &def) {

  // Try to search for method that makes `get` redundant.
  auto loc = def.getDef()->getFieldLoc("builders");
  for (auto &method : defCls.getMethods()) {
    if (method->getName() == methodName &&
        method->makesRedundant(currentMethod)) {
      PrintError(loc, llvm::Twine("builder `") + methodName +
                          "` conflicts with an existing builder. ");
      PrintFatalNote(llvm::Twine("A new builder with signature:\n") +
                     getSignature(currentMethod) +
                     "\nis shadowed by an existing builder with signature:\n" +
                     getSignature(*method) +
                     "\nPlease remove one of the conflicting "
                     "definitions.");
    }
  }

  // This code shouldn't be reached, but leaving this here for potential future
  // use.
  PrintFatalError(loc, "Failed to generate builder " + methodName);
}

void DefGen::emitCustomBuilder(const AttrOrTypeBuilder &builder) {
  // Don't emit a body if there isn't one.
  auto props = builder.getBody() ? Method::Static : Method::StaticDeclaration;
  StringRef returnType = def.getCppClassName();
  if (std::optional<StringRef> builderReturnType = builder.getReturnType())
    returnType = *builderReturnType;

  llvm::StringRef methodName = "get";
  const auto parameters = getCustomBuilderParams({}, builder);
  Method *m = defCls.addMethod(returnType, methodName, props, parameters);

  // If method is pruned, report error and terminate.
  if (!m) {
    auto curMethod = Method(returnType, methodName, props, parameters);
    emitDuplicatedBuilderError(curMethod, methodName, defCls, def);
  }

  if (!builder.getBody())
    return;

  // Format the body and emit it.
  FmtContext ctx;
  ctx.addSubst("_get", "Base::get");
  if (!builder.hasInferredContextParameter())
    ctx.addSubst("_ctxt", "context");
  std::string bodyStr = tgfmt(*builder.getBody(), &ctx);
  m->body().indent().getStream().printReindented(bodyStr);
}

/// Replace all instances of 'from' to 'to' in `str` and return the new string.
static std::string replaceInStr(std::string str, StringRef from, StringRef to) {
  size_t pos = 0;
  while ((pos = str.find(from.data(), pos, from.size())) != std::string::npos)
    str.replace(pos, from.size(), to.data(), to.size());
  return str;
}

void DefGen::emitCheckedCustomBuilder(const AttrOrTypeBuilder &builder) {
  // Don't emit a body if there isn't one.
  auto props = builder.getBody() ? Method::Static : Method::StaticDeclaration;
  StringRef returnType = def.getCppClassName();
  if (std::optional<StringRef> builderReturnType = builder.getReturnType())
    returnType = *builderReturnType;

  llvm::StringRef methodName = "getChecked";
  auto parameters = getCustomBuilderParams(
      {{"::llvm::function_ref<::mlir::InFlightDiagnostic()>", "emitError"}},
      builder);
  Method *m = defCls.addMethod(returnType, methodName, props, parameters);

  // If method is pruned, report error and terminate.
  if (!m) {
    auto curMethod = Method(returnType, methodName, props, parameters);
    emitDuplicatedBuilderError(curMethod, methodName, defCls, def);
  }

  if (!builder.getBody())
    return;

  // Format the body and emit it. Replace $_get(...) with
  // Base::getChecked(emitError, ...)
  FmtContext ctx;
  if (!builder.hasInferredContextParameter())
    ctx.addSubst("_ctxt", "context");
  std::string bodyStr = replaceInStr(builder.getBody()->str(), "$_get(",
                                     "Base::getChecked(emitError, ");
  bodyStr = tgfmt(bodyStr, &ctx);
  m->body().indent().getStream().printReindented(bodyStr);
}

//===----------------------------------------------------------------------===//
// Interface Method Emission
//===----------------------------------------------------------------------===//

void DefGen::emitTraitMethods(const InterfaceTrait &trait) {
  // Get the set of methods that should always be declared.
  auto alwaysDeclaredMethods = trait.getAlwaysDeclaredMethods();
  StringSet<> alwaysDeclared;
  alwaysDeclared.insert_range(alwaysDeclaredMethods);

  Interface iface = trait.getInterface(); // causes strange bugs if elided
  for (auto &method : iface.getMethods()) {
    // Don't declare if the method has a body. Or if the method has a default
    // implementation and the def didn't request that it always be declared.
    if (method.getBody() || (method.getDefaultImplementation() &&
                             !alwaysDeclared.count(method.getName())))
      continue;
    emitTraitMethod(method);
  }
}

void DefGen::emitTraitMethod(const InterfaceMethod &method) {
  // All interface methods are declaration-only.
  auto props =
      method.isStatic() ? Method::StaticDeclaration : Method::ConstDeclaration;
  SmallVector<MethodParameter> params;
  for (auto &param : method.getArguments())
    params.emplace_back(param.type, param.name);
  defCls.addMethod(method.getReturnType(), method.getName(), props,
                   std::move(params));
}

//===----------------------------------------------------------------------===//
// OpAsm{Type,Attr}Interface Default Method Emission

void DefGen::emitMnemonicAliasMethod() {
  // If the mnemonic is not set, there is nothing to do.
  if (!def.getMnemonic())
    return;

  // Emit the mnemonic alias method.
  SmallVector<MethodParameter> params{{"::llvm::raw_ostream &", "os"}};
  Method *m = defCls.addMethod<Method::Const>("::mlir::OpAsmAliasResult",
                                              "getAlias", std::move(params));
  m->body().indent() << strfmt("os << \"{0}\";\n", *def.getMnemonic())
                     << "return ::mlir::OpAsmAliasResult::OverridableAlias;\n";
}

//===----------------------------------------------------------------------===//
// Storage Class Emission
//===----------------------------------------------------------------------===//

void DefGen::emitStorageConstructor() {
  Constructor *ctor =
      storageCls->addConstructor<Method::Inline>(getBuilderParams({}));
  for (auto &param : params) {
    std::string movedValue = ("std::move(" + param.getName() + ")").str();
    ctor->addMemberInitializer(param.getName(), movedValue);
  }
}

void DefGen::emitKeyType() {
  std::string keyType("std::tuple<");
  llvm::raw_string_ostream os(keyType);
  llvm::interleaveComma(params, os,
                        [&](auto &param) { os << param.getCppType(); });
  os << '>';
  storageCls->declare<UsingDeclaration>("KeyTy", std::move(os.str()));

  // Add a method to construct the key type from the storage.
  Method *m = storageCls->addConstMethod<Method::Inline>("KeyTy", "getAsKey");
  m->body().indent() << "return KeyTy(";
  llvm::interleaveComma(params, m->body().indent(),
                        [&](auto &param) { m->body() << param.getName(); });
  m->body() << ");";
}

void DefGen::emitEquals() {
  Method *eq = storageCls->addConstMethod<Method::Inline>(
      "bool", "operator==", MethodParameter("const KeyTy &", "tblgenKey"));
  auto &body = eq->body().indent();
  auto scope = body.scope("return (", ");");
  const auto eachFn = [&](auto it) {
    FmtContext ctx({{"_lhs", it.value().getName()},
                    {"_rhs", strfmt("std::get<{0}>(tblgenKey)", it.index())}});
    body << tgfmt(it.value().getComparator(), &ctx);
  };
  llvm::interleave(llvm::enumerate(params), body, eachFn, ") && (");
}

void DefGen::emitHashKey() {
  Method *hash = storageCls->addStaticInlineMethod(
      "::llvm::hash_code", "hashKey",
      MethodParameter("const KeyTy &", "tblgenKey"));
  auto &body = hash->body().indent();
  auto scope = body.scope("return ::llvm::hash_combine(", ");");
  llvm::interleaveComma(llvm::enumerate(params), body, [&](auto it) {
    body << llvm::formatv("std::get<{0}>(tblgenKey)", it.index());
  });
}

void DefGen::emitConstruct() {
  Method *construct = storageCls->addMethod(
      strfmt("{0} *", def.getStorageClassName()), "construct",
      def.hasStorageCustomConstructor() ? Method::StaticDeclaration
                                        : Method::StaticInline,
      MethodParameter(strfmt("::mlir::{0}StorageAllocator &", valueType),
                      "allocator"),
      MethodParameter("KeyTy &&", "tblgenKey"));
  if (!def.hasStorageCustomConstructor()) {
    auto &body = construct->body().indent();
    for (const auto &it : llvm::enumerate(params)) {
      body << formatv("auto {0} = std::move(std::get<{1}>(tblgenKey));\n",
                      it.value().getName(), it.index());
    }
    // Use the parameters' custom allocator code, if provided.
    FmtContext ctx = FmtContext().addSubst("_allocator", "allocator");
    for (auto &param : params) {
      if (std::optional<StringRef> allocCode = param.getAllocator()) {
        ctx.withSelf(param.getName()).addSubst("_dst", param.getName());
        body << tgfmt(*allocCode, &ctx) << '\n';
      }
    }
    auto scope =
        body.scope(strfmt("return new (allocator.allocate<{0}>()) {0}(",
                          def.getStorageClassName()),
                   ");");
    llvm::interleaveComma(params, body, [&](auto &param) {
      body << "std::move(" << param.getName() << ")";
    });
  }
}

void DefGen::emitStorageClass() {
  // Add the appropriate parent class.
  storageCls->addParent(strfmt("::mlir::{0}Storage", valueType));
  // Add the constructor.
  emitStorageConstructor();
  // Declare the key type.
  emitKeyType();
  // Add the comparison method.
  emitEquals();
  // Emit the key hash method.
  emitHashKey();
  // Emit the storage constructor. Just declare it if the user wants to define
  // it themself.
  emitConstruct();
  // Emit the storage class members as public, at the very end of the struct.
  storageCls->finalize();
  for (auto &param : params) {
    if (param.getCppType().contains("APInt") && !param.hasCustomComparator()) {
      PrintFatalError(
          def.getLoc(),
          "Using a raw APInt parameter without a custom comparator is "
          "not supported because an assert in the equality operator is "
          "triggered when the two APInts have different bit widths. This can "
          "lead to unexpected crashes. Use an `APIntParameter` or "
          "provide a custom comparator.");
    }
    storageCls->declare<Field>(param.getCppType(), param.getName());
  }
}

//===----------------------------------------------------------------------===//
// DefGenerator
//===----------------------------------------------------------------------===//

namespace {
/// This struct is the base generator used when processing tablegen interfaces.
class DefGenerator {
public:
  bool emitDecls(StringRef selectedDialect);
  bool emitDefs(StringRef selectedDialect);

protected:
  DefGenerator(ArrayRef<const Record *> defs, raw_ostream &os,
               StringRef defType, StringRef valueType, bool isAttrGenerator)
      : defRecords(defs), os(os), defType(defType), valueType(valueType),
        isAttrGenerator(isAttrGenerator) {
    // Sort by occurrence in file.
    llvm::sort(defRecords, [](const Record *lhs, const Record *rhs) {
      return lhs->getID() < rhs->getID();
    });
  }

  /// Emit the list of def type names.
  void emitTypeDefList(ArrayRef<AttrOrTypeDef> defs);
  /// Emit the code to dispatch between different defs during parsing/printing.
  void emitParsePrintDispatch(ArrayRef<AttrOrTypeDef> defs);

  /// The set of def records to emit.
  std::vector<const Record *> defRecords;
  /// The attribute or type class to emit.
  /// The stream to emit to.
  raw_ostream &os;
  /// The prefix of the tablegen def name, e.g. Attr or Type.
  StringRef defType;
  /// The C++ base value type of the def, e.g. Attribute or Type.
  StringRef valueType;
  /// Flag indicating if this generator is for Attributes. False if the
  /// generator is for types.
  bool isAttrGenerator;
};

/// A specialized generator for AttrDefs.
struct AttrDefGenerator : public DefGenerator {
  AttrDefGenerator(const RecordKeeper &records, raw_ostream &os)
      : DefGenerator(records.getAllDerivedDefinitionsIfDefined("AttrDef"), os,
                     "Attr", "Attribute", /*isAttrGenerator=*/true) {}
};
/// A specialized generator for TypeDefs.
struct TypeDefGenerator : public DefGenerator {
  TypeDefGenerator(const RecordKeeper &records, raw_ostream &os)
      : DefGenerator(records.getAllDerivedDefinitionsIfDefined("TypeDef"), os,
                     "Type", "Type", /*isAttrGenerator=*/false) {}
};
} // namespace

//===----------------------------------------------------------------------===//
// GEN: Declarations
//===----------------------------------------------------------------------===//

/// Print this above all the other declarations. Contains type declarations used
/// later on.
static const char *const typeDefDeclHeader = R"(
namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
)";

bool DefGenerator::emitDecls(StringRef selectedDialect) {
  emitSourceFileHeader((defType + "Def Declarations").str(), os);
  llvm::IfDefEmitter scope(os, "GET_" + defType.upper() + "DEF_CLASSES");

  // Output the common "header".
  os << typeDefDeclHeader;

  SmallVector<AttrOrTypeDef, 16> defs;
  collectAllDefs(selectedDialect, defRecords, defs);
  if (defs.empty())
    return false;
  {
    DialectNamespaceEmitter nsEmitter(os, defs.front().getDialect());

    // Declare all the def classes first (in case they reference each other).
    for (const AttrOrTypeDef &def : defs) {
      std::string comments = tblgen::emitSummaryAndDescComments(
          def.getSummary(), def.getDescription());
      if (!comments.empty()) {
        os << comments << "\n";
      }
      os << "class " << def.getCppClassName() << ";\n";
    }

    // Emit the declarations.
    for (const AttrOrTypeDef &def : defs)
      DefGen(def).emitDecl(os);
  }
  // Emit the TypeID explicit specializations to have a single definition for
  // each of these.
  for (const AttrOrTypeDef &def : defs)
    if (!def.getDialect().getCppNamespace().empty())
      os << "MLIR_DECLARE_EXPLICIT_TYPE_ID("
         << def.getDialect().getCppNamespace() << "::" << def.getCppClassName()
         << ")\n";

  return false;
}

//===----------------------------------------------------------------------===//
// GEN: Def List
//===----------------------------------------------------------------------===//

void DefGenerator::emitTypeDefList(ArrayRef<AttrOrTypeDef> defs) {
  llvm::IfDefEmitter scope(os, "GET_" + defType.upper() + "DEF_LIST");
  auto interleaveFn = [&](const AttrOrTypeDef &def) {
    os << def.getDialect().getCppNamespace() << "::" << def.getCppClassName();
  };
  llvm::interleave(defs, os, interleaveFn, ",\n");
  os << "\n";
}

//===----------------------------------------------------------------------===//
// GEN: Definitions
//===----------------------------------------------------------------------===//

/// The code block for default attribute parser/printer dispatch boilerplate.
/// {0}: the dialect fully qualified class name.
/// {1}: the optional code for the dynamic attribute parser dispatch.
/// {2}: the optional code for the dynamic attribute printer dispatch.
static const char *const dialectDefaultAttrPrinterParserDispatch = R"(
/// Parse an attribute registered to this dialect.
::mlir::Attribute {0}::parseAttribute(::mlir::DialectAsmParser &parser,
                                      ::mlir::Type type) const {{
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef attrTag;
  {{
    ::mlir::Attribute attr;
    auto parseResult = generatedAttributeParser(parser, &attrTag, type, attr);
    if (parseResult.has_value())
      return attr;
  }
  {1}
  parser.emitError(typeLoc) << "unknown attribute `"
      << attrTag << "` in dialect `" << getNamespace() << "`";
  return {{};
}
/// Print an attribute registered to this dialect.
void {0}::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {{
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  {2}
}
)";

/// The code block for dynamic attribute parser dispatch boilerplate.
static const char *const dialectDynamicAttrParserDispatch = R"(
  {
    ::mlir::Attribute genAttr;
    auto parseResult = parseOptionalDynamicAttr(attrTag, parser, genAttr);
    if (parseResult.has_value()) {
      if (::mlir::succeeded(parseResult.value()))
        return genAttr;
      return Attribute();
    }
  }
)";

/// The code block for dynamic type printer dispatch boilerplate.
static const char *const dialectDynamicAttrPrinterDispatch = R"(
  if (::mlir::succeeded(printIfDynamicAttr(attr, printer)))
    return;
)";

/// The code block for default type parser/printer dispatch boilerplate.
/// {0}: the dialect fully qualified class name.
/// {1}: the optional code for the dynamic type parser dispatch.
/// {2}: the optional code for the dynamic type printer dispatch.
static const char *const dialectDefaultTypePrinterParserDispatch = R"(
/// Parse a type registered to this dialect.
::mlir::Type {0}::parseType(::mlir::DialectAsmParser &parser) const {{
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef mnemonic;
  ::mlir::Type genType;
  auto parseResult = generatedTypeParser(parser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;
  {1}
  parser.emitError(typeLoc) << "unknown  type `"
      << mnemonic << "` in dialect `" << getNamespace() << "`";
  return {{};
}
/// Print a type registered to this dialect.
void {0}::printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const {{
  if (::mlir::succeeded(generatedTypePrinter(type, printer)))
    return;
  {2}
}
)";

/// The code block for dynamic type parser dispatch boilerplate.
static const char *const dialectDynamicTypeParserDispatch = R"(
  {
    auto parseResult = parseOptionalDynamicType(mnemonic, parser, genType);
    if (parseResult.has_value()) {
      if (::mlir::succeeded(parseResult.value()))
        return genType;
      return ::mlir::Type();
    }
  }
)";

/// The code block for dynamic type printer dispatch boilerplate.
static const char *const dialectDynamicTypePrinterDispatch = R"(
  if (::mlir::succeeded(printIfDynamicType(type, printer)))
    return;
)";

/// Emit the dialect printer/parser dispatcher. User's code should call these
/// functions from their dialect's print/parse methods.
void DefGenerator::emitParsePrintDispatch(ArrayRef<AttrOrTypeDef> defs) {
  if (llvm::none_of(defs, [](const AttrOrTypeDef &def) {
        return def.getMnemonic().has_value();
      })) {
    return;
  }
  // Declare the parser.
  SmallVector<MethodParameter> params = {{"::mlir::AsmParser &", "parser"},
                                         {"::llvm::StringRef *", "mnemonic"}};
  if (isAttrGenerator)
    params.emplace_back("::mlir::Type", "type");
  params.emplace_back(strfmt("::mlir::{0} &", valueType), "value");
  Method parse("::mlir::OptionalParseResult",
               strfmt("generated{0}Parser", valueType), Method::StaticInline,
               std::move(params));
  // Declare the printer.
  Method printer("::llvm::LogicalResult",
                 strfmt("generated{0}Printer", valueType), Method::StaticInline,
                 {{strfmt("::mlir::{0}", valueType), "def"},
                  {"::mlir::AsmPrinter &", "printer"}});

  // The parser dispatch uses a KeywordSwitch, matching on the mnemonic and
  // calling the def's parse function.
  parse.body() << "  return "
                  "::mlir::AsmParser::KeywordSwitch<::mlir::"
                  "OptionalParseResult>(parser)\n";
  const char *const getValueForMnemonic =
      R"(    .Case({0}::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {{
      value = {0}::{1};
      return ::mlir::success(!!value);
    })
)";

  // The printer dispatch uses llvm::TypeSwitch to find and call the correct
  // printer.
  printer.body() << "  return ::llvm::TypeSwitch<::mlir::" << valueType
                 << ", ::llvm::LogicalResult>(def)";
  const char *const printValue = R"(    .Case<{0}>([&](auto t) {{
      printer << {0}::getMnemonic();{1}
      return ::mlir::success();
    })
)";
  for (auto &def : defs) {
    if (!def.getMnemonic())
      continue;
    bool hasParserPrinterDecl =
        def.hasCustomAssemblyFormat() || def.getAssemblyFormat();
    std::string defClass = strfmt(
        "{0}::{1}", def.getDialect().getCppNamespace(), def.getCppClassName());

    // If the def has no parameters or parser code, invoke a normal `get`.
    std::string parseOrGet =
        hasParserPrinterDecl
            ? strfmt("parse(parser{0})", isAttrGenerator ? ", type" : "")
            : "get(parser.getContext())";
    parse.body() << llvm::formatv(getValueForMnemonic, defClass, parseOrGet);

    // If the def has no parameters and no printer, just print the mnemonic.
    StringRef printDef = "";
    if (hasParserPrinterDecl)
      printDef = "\nt.print(printer);";
    printer.body() << llvm::formatv(printValue, defClass, printDef);
  }
  parse.body() << "    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {\n"
                  "      *mnemonic = keyword;\n"
                  "      return std::nullopt;\n"
                  "    });";
  printer.body() << "    .Default([](auto) { return ::mlir::failure(); });";

  raw_indented_ostream indentedOs(os);
  parse.writeDeclTo(indentedOs);
  printer.writeDeclTo(indentedOs);
}

bool DefGenerator::emitDefs(StringRef selectedDialect) {
  emitSourceFileHeader((defType + "Def Definitions").str(), os);

  SmallVector<AttrOrTypeDef, 16> defs;
  collectAllDefs(selectedDialect, defRecords, defs);
  if (defs.empty())
    return false;
  emitTypeDefList(defs);

  llvm::IfDefEmitter scope(os, "GET_" + defType.upper() + "DEF_CLASSES");
  emitParsePrintDispatch(defs);
  for (const AttrOrTypeDef &def : defs) {
    {
      DialectNamespaceEmitter ns(os, def.getDialect());
      DefGen gen(def);
      gen.emitDef(os);
    }
    // Emit the TypeID explicit specializations to have a single symbol def.
    if (!def.getDialect().getCppNamespace().empty())
      os << "MLIR_DEFINE_EXPLICIT_TYPE_ID("
         << def.getDialect().getCppNamespace() << "::" << def.getCppClassName()
         << ")\n";
  }

  Dialect firstDialect = defs.front().getDialect();

  // Emit the default parser/printer for Attributes if the dialect asked for it.
  if (isAttrGenerator && firstDialect.useDefaultAttributePrinterParser()) {
    DialectNamespaceEmitter nsEmitter(os, firstDialect);
    if (firstDialect.isExtensible()) {
      os << llvm::formatv(dialectDefaultAttrPrinterParserDispatch,
                          firstDialect.getCppClassName(),
                          dialectDynamicAttrParserDispatch,
                          dialectDynamicAttrPrinterDispatch);
    } else {
      os << llvm::formatv(dialectDefaultAttrPrinterParserDispatch,
                          firstDialect.getCppClassName(), "", "");
    }
  }

  // Emit the default parser/printer for Types if the dialect asked for it.
  if (!isAttrGenerator && firstDialect.useDefaultTypePrinterParser()) {
    DialectNamespaceEmitter nsEmitter(os, firstDialect);
    if (firstDialect.isExtensible()) {
      os << llvm::formatv(dialectDefaultTypePrinterParserDispatch,
                          firstDialect.getCppClassName(),
                          dialectDynamicTypeParserDispatch,
                          dialectDynamicTypePrinterDispatch);
    } else {
      os << llvm::formatv(dialectDefaultTypePrinterParserDispatch,
                          firstDialect.getCppClassName(), "", "");
    }
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Constraints
//===----------------------------------------------------------------------===//

/// Find all type constraints for which a C++ function should be generated.
static std::vector<Constraint> getAllCppConstraints(const RecordKeeper &records,
                                                    StringRef constraintKind) {
  std::vector<Constraint> result;
  for (const Record *def :
       records.getAllDerivedDefinitionsIfDefined(constraintKind)) {
    // Ignore constraints defined outside of the top-level file.
    if (llvm::SrcMgr.FindBufferContainingLoc(def->getLoc()[0]) !=
        llvm::SrcMgr.getMainFileID())
      continue;
    Constraint constr(def);
    // Generate C++ function only if "cppFunctionName" is set.
    if (!constr.getCppFunctionName())
      continue;
    result.push_back(constr);
  }
  return result;
}

static std::vector<Constraint>
getAllCppTypeConstraints(const RecordKeeper &records) {
  return getAllCppConstraints(records, "TypeConstraint");
}

static std::vector<Constraint>
getAllCppAttrConstraints(const RecordKeeper &records) {
  return getAllCppConstraints(records, "AttrConstraint");
}

/// Emit the declarations for the given constraints, of the form:
/// `bool <constraintCppFunctionName>(<parameterTypeName> <parameterName>);`
static void emitConstraintDecls(const std::vector<Constraint> &constraints,
                                raw_ostream &os, StringRef parameterTypeName,
                                StringRef parameterName) {
  static const char *const constraintDecl = "bool {0}({1} {2});\n";
  for (Constraint constr : constraints)
    os << strfmt(constraintDecl, *constr.getCppFunctionName(),
                 parameterTypeName, parameterName);
}

static void emitTypeConstraintDecls(const RecordKeeper &records,
                                    raw_ostream &os) {
  emitConstraintDecls(getAllCppTypeConstraints(records), os, "::mlir::Type",
                      "type");
}

static void emitAttrConstraintDecls(const RecordKeeper &records,
                                    raw_ostream &os) {
  emitConstraintDecls(getAllCppAttrConstraints(records), os,
                      "::mlir::Attribute", "attr");
}

/// Emit the definitions for the given constraints, of the form:
/// `bool <constraintCppFunctionName>(<parameterTypeName> <parameterName>) {
///   return (<condition>); }`
/// where `<condition>` is the condition template with the `self` variable
/// replaced with the `selfName` parameter.
static void emitConstraintDefs(const std::vector<Constraint> &constraints,
                               raw_ostream &os, StringRef parameterTypeName,
                               StringRef selfName) {
  static const char *const constraintDef = R"(
bool {0}({1} {2}) {
return ({3});
}
)";

  for (Constraint constr : constraints) {
    FmtContext ctx;
    ctx.withSelf(selfName);
    std::string condition = tgfmt(constr.getConditionTemplate(), &ctx);
    os << strfmt(constraintDef, *constr.getCppFunctionName(), parameterTypeName,
                 selfName, condition);
  }
}

static void emitTypeConstraintDefs(const RecordKeeper &records,
                                   raw_ostream &os) {
  emitConstraintDefs(getAllCppTypeConstraints(records), os, "::mlir::Type",
                     "type");
}

static void emitAttrConstraintDefs(const RecordKeeper &records,
                                   raw_ostream &os) {
  emitConstraintDefs(getAllCppAttrConstraints(records), os, "::mlir::Attribute",
                     "attr");
}

//===----------------------------------------------------------------------===//
// GEN: Registration hooks
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AttrDef
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory attrdefGenCat("Options for -gen-attrdef-*");
static llvm::cl::opt<std::string>
    attrDialect("attrdefs-dialect",
                llvm::cl::desc("Generate attributes for this dialect"),
                llvm::cl::cat(attrdefGenCat), llvm::cl::CommaSeparated);

static mlir::GenRegistration
    genAttrDefs("gen-attrdef-defs", "Generate AttrDef definitions",
                [](const RecordKeeper &records, raw_ostream &os) {
                  AttrDefGenerator generator(records, os);
                  return generator.emitDefs(attrDialect);
                });
static mlir::GenRegistration
    genAttrDecls("gen-attrdef-decls", "Generate AttrDef declarations",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   AttrDefGenerator generator(records, os);
                   return generator.emitDecls(attrDialect);
                 });

static mlir::GenRegistration
    genAttrConstrDefs("gen-attr-constraint-defs",
                      "Generate attribute constraint definitions",
                      [](const RecordKeeper &records, raw_ostream &os) {
                        emitAttrConstraintDefs(records, os);
                        return false;
                      });
static mlir::GenRegistration
    genAttrConstrDecls("gen-attr-constraint-decls",
                       "Generate attribute constraint declarations",
                       [](const RecordKeeper &records, raw_ostream &os) {
                         emitAttrConstraintDecls(records, os);
                         return false;
                       });

//===----------------------------------------------------------------------===//
// TypeDef
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory typedefGenCat("Options for -gen-typedef-*");
static llvm::cl::opt<std::string>
    typeDialect("typedefs-dialect",
                llvm::cl::desc("Generate types for this dialect"),
                llvm::cl::cat(typedefGenCat), llvm::cl::CommaSeparated);

static mlir::GenRegistration
    genTypeDefs("gen-typedef-defs", "Generate TypeDef definitions",
                [](const RecordKeeper &records, raw_ostream &os) {
                  TypeDefGenerator generator(records, os);
                  return generator.emitDefs(typeDialect);
                });
static mlir::GenRegistration
    genTypeDecls("gen-typedef-decls", "Generate TypeDef declarations",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   TypeDefGenerator generator(records, os);
                   return generator.emitDecls(typeDialect);
                 });

static mlir::GenRegistration
    genTypeConstrDefs("gen-type-constraint-defs",
                      "Generate type constraint definitions",
                      [](const RecordKeeper &records, raw_ostream &os) {
                        emitTypeConstraintDefs(records, os);
                        return false;
                      });
static mlir::GenRegistration
    genTypeConstrDecls("gen-type-constraint-decls",
                       "Generate type constraint declarations",
                       [](const RecordKeeper &records, raw_ostream &os) {
                         emitTypeConstraintDecls(records, os);
                         return false;
                       });
