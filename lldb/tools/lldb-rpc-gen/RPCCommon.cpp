//===-- RPCCommon.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RPCCommon.h"

#include "clang/AST/AST.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Mangle.h"
#include "clang/Lex/Lexer.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>

using namespace clang;

// We intentionally do not generate some classes because they are currently
// inconvenient, they aren't really used by most consumers, or we're not sure
// why they exist.
static constexpr llvm::StringRef DisallowedClasses[] = {
    "SBCommunication", // This class is pretty much unused by consumers, so we
                       // skip it.
    "SBInputReader",   // This class is pretty much unused by consumers, so we
                       // skip it.
    "SBCommandPluginInterface", // This class uses virtual functions, and the SB
                                // API should not have those, so we skip this
                                // class.
    "SBCommand", // There's nothing too difficult about this one, but many of
                 // its methods take a SBCommandPluginInterface pointer so
                 // there's no reason to support this.
};

// NOTE: In lldb-rpc-gen, we use mangled names when we need to work with
// functions. We do this because we support many functions that have overloads,
// and mangled names have no ambiguity which makes it easier to keep track of.
// This is also possible since the LLDB SB API is stable.

// We intentionally avoid generating certain methods either because they are
// difficult to support correctly or they aren't really used much from C++.
// NOTE: These methods are marked as deprecated using LLDB_DEPRECATED.
// Normally this macro defines to the deprecated annotation, but this
// functionality is removed in SBDefines.h when generating SWIG bindings which
// we use for testing. Because of this, there is no annotation for the tool to
// pick up on so this list will be used while we have this restriction in
// SBDefines.h.
static constexpr llvm::StringRef DisallowedMethods[] = {
    // The threading functionality in SBHostOS is deprecated and thus we do not
    // generate them. It would be ideal to add the annotations to the methods
    // and then support not generating deprecated methods. However, without
    // annotations the generator generates most things correctly. This one is
    // problematic because it returns a pointer to an "opaque" structure
    // (thread_t) that is not `void *`, so special casing it is more effort than
    // it's worth.
    "_ZN4lldb8SBHostOS10ThreadJoinEP17_opaque_pthread_tPPvPNS_7SBErrorE",
    "_ZN4lldb8SBHostOS12ThreadCancelEP17_opaque_pthread_tPNS_7SBErrorE",
    "_ZN4lldb8SBHostOS12ThreadCreateEPKcPFPvS3_ES3_PNS_7SBErrorE",
    "_ZN4lldb8SBHostOS12ThreadDetachEP17_opaque_pthread_tPNS_7SBErrorE",
    "_ZN4lldb8SBHostOS13ThreadCreatedEPKc",
};

static constexpr llvm::StringRef ClassesWithoutDefaultCtor[] = {
    "SBHostOS",
    "SBReproducer",
};

static constexpr llvm::StringRef ClassesWithoutCopyOperations[] = {
    "SBHostOS",
    "SBReproducer",
    "SBStream",
    "SBProgress",
};

static constexpr llvm::StringRef MethodsWithPointerPlusLen[] = {
    "_ZN4lldb6SBData11ReadRawDataERNS_7SBErrorEyPvm",
    "_ZN4lldb6SBData7SetDataERNS_7SBErrorEPKvmNS_9ByteOrderEh",
    "_ZN4lldb6SBData20SetDataWithOwnershipERNS_7SBErrorEPKvmNS_9ByteOrderEh",
    "_ZN4lldb6SBData25CreateDataFromUInt64ArrayENS_9ByteOrderEjPym",
    "_ZN4lldb6SBData25CreateDataFromUInt32ArrayENS_9ByteOrderEjPjm",
    "_ZN4lldb6SBData25CreateDataFromSInt64ArrayENS_9ByteOrderEjPxm",
    "_ZN4lldb6SBData25CreateDataFromSInt32ArrayENS_9ByteOrderEjPim",
    "_ZN4lldb6SBData25CreateDataFromDoubleArrayENS_9ByteOrderEjPdm",
    "_ZN4lldb6SBData22SetDataFromUInt64ArrayEPym",
    "_ZN4lldb6SBData22SetDataFromUInt32ArrayEPjm",
    "_ZN4lldb6SBData22SetDataFromSInt64ArrayEPxm",
    "_ZN4lldb6SBData22SetDataFromSInt32ArrayEPim",
    "_ZN4lldb6SBData22SetDataFromDoubleArrayEPdm",
    "_ZN4lldb10SBDebugger22GetDefaultArchitectureEPcm",
    "_ZN4lldb10SBDebugger13DispatchInputEPvPKvm",
    "_ZN4lldb10SBDebugger13DispatchInputEPKvm",
    "_ZN4lldb6SBFile4ReadEPhmPm",
    "_ZN4lldb6SBFile5WriteEPKhmPm",
    "_ZNK4lldb10SBFileSpec7GetPathEPcm",
    "_ZN4lldb10SBFileSpec11ResolvePathEPKcPcm",
    "_ZN4lldb8SBModule10GetVersionEPjj",
    "_ZN4lldb12SBModuleSpec12SetUUIDBytesEPKhm",
    "_ZNK4lldb9SBProcess9GetSTDOUTEPcm",
    "_ZNK4lldb9SBProcess9GetSTDERREPcm",
    "_ZNK4lldb9SBProcess19GetAsyncProfileDataEPcm",
    "_ZN4lldb9SBProcess10ReadMemoryEyPvmRNS_7SBErrorE",
    "_ZN4lldb9SBProcess11WriteMemoryEyPKvmRNS_7SBErrorE",
    "_ZN4lldb9SBProcess21ReadCStringFromMemoryEyPvmRNS_7SBErrorE",
    "_ZNK4lldb16SBStructuredData14GetStringValueEPcm",
    "_ZN4lldb8SBTarget23BreakpointCreateByNamesEPPKcjjRKNS_"
    "14SBFileSpecListES6_",
    "_ZN4lldb8SBTarget10ReadMemoryENS_9SBAddressEPvmRNS_7SBErrorE",
    "_ZN4lldb8SBTarget15GetInstructionsENS_9SBAddressEPKvm",
    "_ZN4lldb8SBTarget25GetInstructionsWithFlavorENS_9SBAddressEPKcPKvm",
    "_ZN4lldb8SBTarget15GetInstructionsEyPKvm",
    "_ZN4lldb8SBTarget25GetInstructionsWithFlavorEyPKcPKvm",
    "_ZN4lldb8SBThread18GetStopDescriptionEPcm",
    // The below mangled names are used for dummy methods in shell tests
    // that test the emitters' output. If you're adding any new mangled names
    // from the actual SB API to this list please add them above.
    "_ZN4lldb33SBRPC_"
    "CHECKCONSTCHARPTRPTRWITHLEN27CheckConstCharPtrPtrWithLenEPPKcm",
    "_ZN4lldb19SBRPC_CHECKARRAYPTR13CheckArrayPtrEPPKcm",
    "_ZN4lldb18SBRPC_CHECKVOIDPTR12CheckVoidPtrEPvm",
};

// These classes inherit from rpc::ObjectRef directly (as opposed to
// rpc::LocalObjectRef). Changing them from ObjectRef to LocalObjectRef is ABI
// breaking, so we preserve that compatibility here.
//
// lldb-rpc-gen emits classes as LocalObjectRefs by default.
//
// FIXME: Does it matter which one it emits by default?
static constexpr llvm::StringRef ClassesThatInheritFromObjectRef[] = {
    "SBAddress",
    "SBBreakpointName",
    "SBCommandInterpreter",
    "SBCommandReturnObject",
    "SBError",
    "SBExecutionContext",
    "SBExpressionOptions",
    "SBFileSpec",
    "SBFileSpecList",
    "SBFormat",
    "SBFunction",
    "SBHistoricalFrame",
    "SBHistoricalLineEntry",
    "SBHistoricalLineEntryList",
    "SBLineEntry",
    "SBStream",
    "SBStringList",
    "SBStructuredData",
    "SBSymbolContext",
    "SBSymbolContextList",
    "SBTypeMember",
    "SBTypeSummaryOptions",
    "SBValueList",
};

QualType lldb_rpc_gen::GetUnderlyingType(QualType T) {
  QualType UnderlyingType;
  if (T->isPointerType())
    UnderlyingType = T->getPointeeType();
  else if (T->isReferenceType())
    UnderlyingType = T.getNonReferenceType();
  else
    UnderlyingType = T;

  return UnderlyingType;
}

QualType lldb_rpc_gen::GetUnqualifiedUnderlyingType(QualType T) {
  return GetUnderlyingType(T).getUnqualifiedType();
}

std::string lldb_rpc_gen::GetMangledName(ASTContext &Context,
                                         CXXMethodDecl *MDecl) {
  std::string Mangled;
  llvm::raw_string_ostream MangledStream(Mangled);

  GlobalDecl GDecl;
  if (const auto *CtorDecl = dyn_cast<CXXConstructorDecl>(MDecl))
    GDecl = GlobalDecl(CtorDecl, Ctor_Complete);
  else if (const auto *DtorDecl = dyn_cast<CXXDestructorDecl>(MDecl))
    GDecl = GlobalDecl(DtorDecl, Dtor_Deleting);
  else
    GDecl = GlobalDecl(MDecl);

  MangleContext *MC = Context.createMangleContext();
  MC->mangleName(GDecl, MangledStream);
  return Mangled;
}

bool lldb_rpc_gen::TypeIsFromLLDBPrivate(QualType T) {
  auto CheckTypeForLLDBPrivate = [](const Type *Ty) {
    if (!Ty)
      return false;
    const auto *CXXRDecl = Ty->getAsCXXRecordDecl();
    if (!CXXRDecl)
      return false;
    const auto *NSDecl =
        llvm::dyn_cast<NamespaceDecl>(CXXRDecl->getDeclContext());
    if (!NSDecl)
      return false;
    return NSDecl->getName() == "lldb_private";
  };

  // First, get the underlying type (remove qualifications and strip off any
  // pointers/references). Then we'll need to desugar this type. This will
  // remove things like typedefs, so instead of seeing "lldb::DebuggerSP" we'll
  // actually see something like "std::shared_ptr<lldb_private::Debugger>".
  QualType UnqualifiedUnderlyingType = GetUnqualifiedUnderlyingType(T);
  const Type *DesugaredType =
      UnqualifiedUnderlyingType->getUnqualifiedDesugaredType();
  assert(DesugaredType && "DesugaredType from a valid Type is nullptr!");

  // Check the type itself.
  if (CheckTypeForLLDBPrivate(DesugaredType))
    return true;

  // If that didn't work, it's possible that the type has a template argument
  // that is an lldb_private type.
  if (const auto *TemplateSDecl =
          llvm::dyn_cast_or_null<ClassTemplateSpecializationDecl>(
              DesugaredType->getAsCXXRecordDecl())) {
    for (const TemplateArgument &TA :
         TemplateSDecl->getTemplateArgs().asArray()) {
      if (TA.getKind() != TemplateArgument::Type)
        continue;
      if (CheckTypeForLLDBPrivate(TA.getAsType().getTypePtr()))
        return true;
    }
  }
  return false;
}

bool lldb_rpc_gen::TypeIsSBClass(QualType T) {
  QualType UnqualifiedUnderlyingType = GetUnqualifiedUnderlyingType(T);
  const auto *CXXRDecl = UnqualifiedUnderlyingType->getAsCXXRecordDecl();
  if (!CXXRDecl)
    return false; // SB Classes are always C++ classes

  return CXXRDecl->getName().starts_with("SB");
}

bool lldb_rpc_gen::TypeIsConstCharPtr(QualType T) {
  if (!T->isPointerType())
    return false;

  QualType UnderlyingType = T->getPointeeType();
  if (!UnderlyingType.isConstQualified())
    return false;

  // NOTE: We should be able to do `UnderlyingType->isCharType` but that will
  // return true for `const uint8_t *` since that is effectively an unsigned
  // char pointer. We currently do not support pointers other than `const char
  // *` and `const char **`.

  // NOTE: Checking that the underlying type is a signed integer works on Darwin
  // platforms, but Linux platforms expect that the underlying type is an
  // unsigned integer.
  return UnderlyingType->isSpecificBuiltinType(BuiltinType::Char_S) ||
         UnderlyingType->isSpecificBuiltinType(BuiltinType::SChar) ||
         UnderlyingType->isSpecificBuiltinType(BuiltinType::Char_U) ||
         UnderlyingType->isSpecificBuiltinType(BuiltinType::UChar);
}

bool lldb_rpc_gen::TypeIsConstCharPtrPtr(QualType T) {
  if (!T->isPointerType())
    return false;

  return TypeIsConstCharPtr(T->getPointeeType());
}

bool lldb_rpc_gen::TypeIsDisallowedClass(QualType T) {
  QualType UUT = GetUnqualifiedUnderlyingType(T);
  const auto *CXXRDecl = UUT->getAsCXXRecordDecl();
  if (!CXXRDecl)
    return false;

  llvm::StringRef DeclName = CXXRDecl->getName();
  for (const llvm::StringRef DisallowedClass : DisallowedClasses)
    if (DeclName == DisallowedClass)
      return true;
  return false;
}

bool lldb_rpc_gen::TypeIsCallbackFunctionPointer(QualType T) {
  return T->isFunctionPointerType();
}

bool lldb_rpc_gen::MethodIsDisallowed(ASTContext &Context,
                                      CXXMethodDecl *MDecl) {
  bool isDisallowed = false;
  std::string MangledName = lldb_rpc_gen::GetMangledName(Context, MDecl);
  if (llvm::is_contained(DisallowedMethods, MangledName))
    isDisallowed = true;

  if (MDecl->hasAttrs()) {
    for (auto *attr : MDecl->getAttrs()) {
      if (strcmp(attr->getAttrName()->getNameStart(), "deprecated") == 0)
        isDisallowed = true;
    }
  }
  return isDisallowed;
}

// NOTE: There's possibly a more clever way to do this, but we're keeping
// the string replacement way here. Here is why it is written this way:
// By the time we have already created a `Method` object, we have extracted the
// `QualifiedName` and the relevant QualTypes for parameters/return types, many
// of which contains "lldb::" in them. To change it in a way that would be
// friendly to liblldbrpc, we would need to have a way of replacing that
// namespace at the time of creating a Method, and only for liblldbrpc methods.
// IMO this would complicate Method more than what I'm doing here, and not
// necessarily for any more benefit.
// In clang-tools-extra, there is a ChangeNamespaces tool which tries to do
// something similar to this. It also operates primarily on string replacement,
// but uses more sophisticated clang tooling to do so.
// For now, this will do what we need it to do.
std::string
lldb_rpc_gen::ReplaceLLDBNamespaceWithRPCNamespace(std::string Name) {
  const char *lldb_namespace = "lldb::";
  auto Pos = Name.find(lldb_namespace);
  while (Pos != std::string::npos) {
    constexpr size_t SizeOfLLDBNamespace = 6;
    Name.replace(Pos, SizeOfLLDBNamespace, "lldb_rpc::");
    Pos = Name.find(lldb_namespace);
  }
  return Name;
}

std::string lldb_rpc_gen::StripLLDBNamespace(std::string Name) {
  const char *lldb_namespace = "lldb::";
  auto Pos = Name.find(lldb_namespace);
  if (Pos != std::string::npos) {
    constexpr size_t SizeOfLLDBNamespace = 6;
    Name = Name.substr(Pos + SizeOfLLDBNamespace);
  }
  return Name;
}

bool lldb_rpc_gen::SBClassRequiresDefaultCtor(const std::string &ClassName) {
  return !llvm::is_contained(ClassesWithoutDefaultCtor, ClassName);
}

bool lldb_rpc_gen::SBClassRequiresCopyCtorAssign(const std::string &ClassName) {
  return !llvm::is_contained(ClassesWithoutCopyOperations, ClassName);
}

bool lldb_rpc_gen::SBClassInheritsFromObjectRef(const std::string &ClassName) {
  return llvm::is_contained(ClassesThatInheritFromObjectRef, ClassName);
}

std::string lldb_rpc_gen::GetSBClassNameFromType(QualType T) {
  assert(lldb_rpc_gen::TypeIsSBClass(T) &&
         "Cannot get SBClass name from non-SB class type!");

  QualType UnqualifiedUnderlyingType = GetUnqualifiedUnderlyingType(T);
  const auto *CXXRDecl = UnqualifiedUnderlyingType->getAsCXXRecordDecl();
  assert(CXXRDecl && "SB class was not CXXRecordDecl!");
  if (!CXXRDecl)
    return std::string();

  return CXXRDecl->getName().str();
}
lldb_rpc_gen::Method::Method(CXXMethodDecl *MDecl, const PrintingPolicy &Policy,
                             ASTContext &Context)
    : Policy(Policy), Context(Context),
      QualifiedName(MDecl->getQualifiedNameAsString()),
      BaseName(MDecl->getNameAsString()),
      MangledName(lldb_rpc_gen::GetMangledName(Context, MDecl)),
      ReturnType(MDecl->getReturnType()), IsConst(MDecl->isConst()),
      IsInstance(MDecl->isInstance()), IsCtor(isa<CXXConstructorDecl>(MDecl)),
      IsCopyAssign(MDecl->isCopyAssignmentOperator()),
      IsMoveAssign(MDecl->isMoveAssignmentOperator()),
      IsDtor(isa<CXXDestructorDecl>(MDecl)),
      IsConversionMethod(isa<CXXConversionDecl>(MDecl)) {
  uint8_t UnnamedArgIdx = 0;
  bool PrevParamWasPointer = false;
  for (const auto *ParamDecl : MDecl->parameters()) {
    Param param;
    if (ParamDecl->hasDefaultArg())
      param.DefaultValueText =
          Lexer::getSourceText(
              CharSourceRange::getTokenRange(
                  ParamDecl->getDefaultArg()->getSourceRange()),
              Context.getSourceManager(), Context.getLangOpts())
              .str();

    param.IsFollowedByLen = false;
    param.Name = ParamDecl->getNameAsString();
    // If the parameter has no name, we'll generate one
    if (param.Name.empty()) {
      param.Name = "arg" + std::to_string(UnnamedArgIdx);
      UnnamedArgIdx++;
    }
    param.Type = ParamDecl->getType();

    // FIXME: Instead of using this heuristic, the ideal thing would be to add
    // annotations to the SBAPI methods themselves. For now, we have a list of
    // methods that we know will need this.
    if (PrevParamWasPointer) {
      PrevParamWasPointer = false;
      const bool IsIntegerType = param.Type->isIntegerType() &&
                                 !param.Type->isBooleanType() &&
                                 !param.Type->isEnumeralType();
      if (IsIntegerType && llvm::is_contained(MethodsWithPointerPlusLen,
                                              llvm::StringRef(MangledName)))
        Params.back().IsFollowedByLen = true;
    }

    if (param.Type->isPointerType() &&
        !lldb_rpc_gen::TypeIsConstCharPtr(param.Type) &&
        !param.Type->isFunctionPointerType())
      PrevParamWasPointer = true;

    if (param.Type->isFunctionPointerType())
      ContainsFunctionPointerParameter = true;

    Params.push_back(param);
  }

  if (IsInstance)
    ThisType = MDecl->getThisType();

  if (const auto *CtorDecl = dyn_cast<CXXConstructorDecl>(MDecl)) {
    IsExplicitCtorOrConversionMethod = CtorDecl->isExplicit();
    IsCopyCtor = CtorDecl->isCopyConstructor();
    IsMoveCtor = CtorDecl->isMoveConstructor();
  } else if (const auto *ConversionDecl = dyn_cast<CXXConversionDecl>(MDecl))
    IsExplicitCtorOrConversionMethod = ConversionDecl->isExplicit();
}

// Adding a '<' allows us to use Methods in ordered containers.
// The ordering is on memory addresses.
bool lldb_rpc_gen::Method::operator<(const lldb_rpc_gen::Method &rhs) const {
  return this < &rhs;
}

std::string
lldb_rpc_gen::Method::CreateParamListAsString(GenerationKind Generation,
                                              bool IncludeDefaultValue) const {
  assert((!IncludeDefaultValue || Generation == eLibrary) &&
         "Default values should only be emitted on the library side!");

  std::vector<std::string> ParamList;

  if (Generation == eLibrary && RequiresConnectionParameter())
    ParamList.push_back("const rpc::Connection &connection");

  for (const auto &Param : Params) {
    std::string ParamString;
    llvm::raw_string_ostream ParamStringStream(ParamString);

    if (Generation == eLibrary)
      ParamStringStream << lldb_rpc_gen::ReplaceLLDBNamespaceWithRPCNamespace(
          Param.Type.getAsString(Policy));
    else
      ParamStringStream << Param.Type.getAsString(Policy);

    ParamStringStream << " " << Param.Name;
    if (IncludeDefaultValue && Generation == eLibrary &&
        !Param.DefaultValueText.empty())
      ParamStringStream << " = "
                        << lldb_rpc_gen::ReplaceLLDBNamespaceWithRPCNamespace(
                               Param.DefaultValueText);

    ParamList.push_back(ParamString);
  }

  return llvm::join(ParamList, ", ");
}

bool lldb_rpc_gen::Method::RequiresConnectionParameter() const {
  if (!IsCtor && IsInstance)
    return false;
  if (IsCopyCtor || IsMoveCtor)
    return false;
  for (const auto &Param : Params) {
    // We can re-use the connection from our parameter if possible.
    // Const-qualified parameters are input parameters and already
    // have a valid connection to provide to the current method.
    if (TypeIsSBClass(Param.Type) &&
        GetUnderlyingType(Param.Type).isConstQualified())
      return false;
  }

  return true;
}
