//===-- SwiftLanguageRuntimeRemoteAST.cpp --------------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2020 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "SwiftLanguageRuntimeImpl.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Timer.h"
#include "swift/AST/ASTContext.h"
#include "swift/AST/ASTMangler.h"
#include "swift/AST/ASTWalker.h"
#include "swift/RemoteAST/RemoteAST.h"

using namespace lldb;
using namespace lldb_private;

namespace {
class ASTVerifier : public swift::ASTWalker {
  bool hasMissingPatterns = false;

  PreWalkAction walkToDeclPre(swift::Decl *D) override {
    if (auto *PBD = llvm::dyn_cast<swift::PatternBindingDecl>(D)) {
      if (PBD->getPatternList().empty()) {
        hasMissingPatterns = true;
        return Action::SkipChildren();
      }
    }
    return Action::Continue();
  }

public:
  /// Detect (one form of) incomplete types. These may appear if
  /// member variables have Clang-imported types that couldn't be
  /// resolved.
  static bool Verify(swift::Decl *D) {
    if (!D)
      return false;

    ASTVerifier verifier;
    D->walk(verifier);
    return !verifier.hasMissingPatterns;
  }
};

} // namespace

namespace lldb_private {

swift::remoteAST::RemoteASTContext &
SwiftLanguageRuntimeImpl::GetRemoteASTContext(SwiftASTContext &swift_ast_ctx) {
  // If we already have a remote AST context for this AST context,
  // return it.
  auto known = m_remote_ast_contexts.find(swift_ast_ctx.GetASTContext());
  if (known != m_remote_ast_contexts.end())
    return *known->second;

  // Initialize a new remote AST context.
  (void)GetReflectionContext();
  auto remote_ast_up = std::make_unique<swift::remoteAST::RemoteASTContext>(
      *swift_ast_ctx.GetASTContext(), GetMemoryReader());
  auto &remote_ast = *remote_ast_up;
  m_remote_ast_contexts.insert(
      {swift_ast_ctx.GetASTContext(), std::move(remote_ast_up)});
  return remote_ast;
}

void SwiftLanguageRuntimeImpl::ReleaseAssociatedRemoteASTContext(
    swift::ASTContext *ctx) {
  m_remote_ast_contexts.erase(ctx);
}

llvm::Optional<uint64_t>
SwiftLanguageRuntimeImpl::GetMemberVariableOffsetRemoteAST(
    CompilerType instance_type, ValueObject *instance,
    llvm::StringRef member_name) {
  auto scratch_ctx =
      instance_type.GetTypeSystem().dyn_cast_or_null<SwiftASTContext>();
  if (scratch_ctx == nullptr || scratch_ctx->HasFatalErrors())
    return {};

  auto *remote_ast = &GetRemoteASTContext(*scratch_ctx);
  // Check whether we've already cached this offset.
  swift::TypeBase *swift_type =
      scratch_ctx->GetCanonicalSwiftType(instance_type).getPointer();
  if (swift_type == nullptr)
    return {};

  // Perform the cache lookup.
  MemberID key{swift_type, ConstString(member_name).GetCString()};
  auto it = m_member_offsets.find(key);
  if (it != m_member_offsets.end())
    return it->second;

  // Dig out metadata describing the type, if it's easy to find.
  // FIXME: the Remote AST library should make this easier.
  swift::remote::RemoteAddress optmeta(nullptr);
  const swift::TypeKind type_kind = swift_type->getKind();
  switch (type_kind) {
  case swift::TypeKind::Class:
  case swift::TypeKind::BoundGenericClass: {
    LLDB_LOGF(GetLog(LLDBLog::Types),
              "[MemberVariableOffsetResolver] type is a class - trying to "
              "get metadata for valueobject %s",
              (instance ? instance->GetName().AsCString() : "<null>"));
    if (instance) {
      lldb::addr_t pointer = instance->GetPointerValue();
      if (!pointer || pointer == LLDB_INVALID_ADDRESS)
        break;
      swift::remote::RemoteAddress address(pointer);
      if (auto metadata = remote_ast->getHeapMetadataForObject(address))
        optmeta = metadata.getValue();
    }
    LLDB_LOGF(GetLog(LLDBLog::Types),
              "[MemberVariableOffsetResolver] optmeta = 0x%" PRIx64,
              optmeta.getAddressData());
    break;
  }

  default:
    // Bind generic parameters if necessary.
    if (instance && swift_type->hasTypeParameter())
      if (auto *frame = instance->GetExecutionContextRef().GetFrameSP().get())
        if (auto bound = BindGenericTypeParameters(*frame, instance_type)) {
          LLDB_LOGF(
              GetLog(LLDBLog::Types),
              "[MemberVariableOffsetResolver] resolved non-class type = %s",
              bound.GetTypeName().AsCString());

          swift_type = scratch_ctx->GetCanonicalSwiftType(bound).getPointer();
          MemberID key{swift_type, ConstString(member_name).GetCString()};
          auto it = m_member_offsets.find(key);
          if (it != m_member_offsets.end())
            return it->second;
        }
  }

  // Try to determine whether it is safe to use RemoteAST.  RemoteAST
  // is faster than RemoteMirrors, but can't do dynamic types (checked
  // inside RemoteAST) or incomplete types (checked here).
  bool safe_to_use_remote_ast = true;
  if (swift::Decl *type_decl = swift_type->getNominalOrBoundGenericNominal())
    safe_to_use_remote_ast &= ASTVerifier::Verify(type_decl);

  // Use RemoteAST to determine the member offset.
  if (safe_to_use_remote_ast) {
    swift::remoteAST::Result<uint64_t> result =
        remote_ast->getOffsetOfMember(swift_type, optmeta, member_name);
    if (result) {
      LLDB_LOGF(GetLog(LLDBLog::Types),
                "[MemberVariableOffsetResolver] offset discovered = %" PRIu64,
                (uint64_t)result.getValue());

      // Cache this result.
      MemberID key{swift_type, ConstString(member_name).GetCString()};
      m_member_offsets.insert({key, result.getValue()});
      return result.getValue();
    }

    const auto &failure = result.getFailure();
    LLDB_LOGF(GetLog(LLDBLog::Types),
              "[MemberVariableOffsetResolver] failure: %s",
              failure.render().c_str());
  }
  return {};
}

#ifndef NDEBUG
ConstString SwiftLanguageRuntimeImpl::GetDynamicTypeName_ClassRemoteAST(
    ValueObject &in_value, lldb::addr_t instance_ptr) {
  // Dynamic type resolution in RemoteAST might pull in other Swift modules, so
  // use the scratch context where such operations are legal and safe.

  llvm::Optional<SwiftScratchContextReader> maybe_scratch_ctx =
      in_value.GetSwiftScratchContext();
  if (!maybe_scratch_ctx)
    return {};
  auto scratch_ctx = maybe_scratch_ctx->get();
  if (!scratch_ctx)
    return {};
  SwiftASTContext *swift_ast_ctx = scratch_ctx->GetSwiftASTContext();
  if (!swift_ast_ctx)
    return {};

  auto &remote_ast = GetRemoteASTContext(*swift_ast_ctx);
  auto remote_ast_metadata_address = remote_ast.getHeapMetadataForObject(
      swift::remote::RemoteAddress(instance_ptr));
  if (remote_ast_metadata_address) {
    auto instance_type = remote_ast.getTypeForRemoteTypeMetadata(
        remote_ast_metadata_address.getValue(),
        /*skipArtificial=*/true);
    if (instance_type) {
      auto ref_type = ToCompilerType(instance_type.getValue());
      ConstString name = ref_type.GetMangledTypeName();
      if (!name)
        LLDB_LOG(GetLog(LLDBLog::Types), "could not get type metadata:{0}\n",
                 instance_type.getFailure().render());
      return name;
    }
  }
  return {};
}

llvm::Optional<std::pair<CompilerType, Address>>
SwiftLanguageRuntimeImpl::GetDynamicTypeAndAddress_ProtocolRemoteAST(
    ValueObject &in_value, CompilerType protocol_type, bool use_local_buffer,
    lldb::addr_t existential_address) {
  // Dynamic type resolution in RemoteAST might pull in other Swift
  // modules, so use the scratch context where such operations are
  // legal and safe.
  llvm::Optional<SwiftScratchContextReader> maybe_scratch_ctx =
      in_value.GetSwiftScratchContext();
  if (!maybe_scratch_ctx)
    return {};
  auto scratch_ctx = maybe_scratch_ctx->get();
  if (!scratch_ctx)
    return {};
  SwiftASTContext *swift_ast_ctx = scratch_ctx->GetSwiftASTContext();
  if (!swift_ast_ctx)
    return {};

  swift::remote::RemoteAddress remote_existential(existential_address);
  auto &remote_ast = GetRemoteASTContext(*swift_ast_ctx);
  auto swift_type = swift_ast_ctx->GetSwiftType(protocol_type);
  if (!swift_type)
    return {};
  if (use_local_buffer)
    PushLocalBuffer(existential_address, in_value.GetByteSize().value_or(0));

  auto result = remote_ast.getDynamicTypeAndAddressForExistential(
      remote_existential, swift_type);
  if (use_local_buffer)
    PopLocalBuffer();

  if (!result.isSuccess())
    return {};

  auto type_and_address = result.getValue();

  CompilerType type = ToCompilerType(type_and_address.InstanceType);
  Address address;
  address.SetRawAddress(type_and_address.PayloadAddress.getAddressData());
  return {{type, address}};
}
#endif

CompilerType SwiftLanguageRuntimeImpl::BindGenericTypeParametersRemoteAST(
    StackFrame &stack_frame, CompilerType base_type) {
  LLDB_SCOPED_TIMER();

  // If this is a TypeRef type, bind that.
  auto sc = stack_frame.GetSymbolContext(lldb::eSymbolContextEverything);
  if (auto ts =
          base_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwiftTypeRef>())
    return BindGenericTypeParameters(stack_frame, *ts,
                                     base_type.GetMangledTypeName());

  Status error;
  auto &target = m_process.GetTarget();
  assert(IsScratchContextLocked(target) &&
         "Swift scratch context not locked ahead of archetype binding");

  // A failing Clang import in a module context permanently damages
  // that module context.  Binding archetypes can trigger an import of
  // another module, so switch to a scratch context where such an
  // operation is safe.
  llvm::Optional<SwiftScratchContextReader> maybe_scratch_ctx =
      target.GetSwiftScratchContext(error, stack_frame);
  if (!maybe_scratch_ctx)
    return base_type;
  auto scratch_ctx = maybe_scratch_ctx->get();
  if (!scratch_ctx)
    return base_type;
  SwiftASTContext *swift_ast_ctx = scratch_ctx->GetSwiftASTContext();
  if (!swift_ast_ctx)
    return base_type;
  base_type = swift_ast_ctx->ImportType(base_type, error);

  if (base_type.GetTypeInfo() & lldb::eTypeIsSwift) {
    swift::Type target_swift_type(swift_ast_ctx->GetSwiftType(base_type));
    if (target_swift_type->hasArchetype())
      target_swift_type = target_swift_type->mapTypeOutOfContext().getPointer();

    // FIXME: This is wrong, but it doesn't actually matter right now since
    // all conformances are always visible
    auto *module_decl = swift_ast_ctx->GetASTContext()->getStdlibModule();

    // Replace opaque types with their underlying types when possible.
    swift::Mangle::ASTMangler mangler(true);

    // Rewrite all dynamic self types to their static self types.
    target_swift_type =
        target_swift_type.transform([](swift::Type type) -> swift::Type {
          if (auto *dynamic_self =
                  llvm::dyn_cast<swift::DynamicSelfType>(type.getPointer()))
            return dynamic_self->getSelfType();
          return type;
        });

    // Thicken generic metatypes. Once substituted, they should always
    // be thick. TypeRef::subst() does the same transformation.
    target_swift_type =
        target_swift_type.transform([](swift::Type type) -> swift::Type {
          using namespace swift;
          const auto thin = MetatypeRepresentation::Thin;
          const auto thick = MetatypeRepresentation::Thick;
          if (auto *metatype = dyn_cast<AnyMetatypeType>(type.getPointer()))
            if (metatype->hasRepresentation() &&
                metatype->getRepresentation() == thin &&
                metatype->getInstanceType()->hasTypeParameter())
              return MetatypeType::get(metatype->getInstanceType(), thick);
          return type;
        });

    while (target_swift_type->hasOpaqueArchetype()) {
      auto old_type = target_swift_type;
      target_swift_type = target_swift_type.subst(
          [&](swift::SubstitutableType *type) -> swift::Type {
            auto opaque_type =
                llvm::dyn_cast<swift::OpaqueTypeArchetypeType>(type);
            if (!opaque_type || !opaque_type->getInterfaceType()
                                     ->is<swift::GenericTypeParamType>())
              return type;

            // Try to find the symbol for the opaque type descriptor in the
            // process.
            auto mangled_name = ConstString(
                mangler.mangleOpaqueTypeDescriptor(opaque_type->getDecl()));

            SymbolContextList found;
            target.GetImages().FindSymbolsWithNameAndType(
                mangled_name, eSymbolTypeData, found);

            if (found.GetSize() == 0)
              return type;

            swift::Type result_type;

            for (unsigned i = 0, e = found.GetSize(); i < e; ++i) {
              SymbolContext found_sc;
              if (!found.GetContextAtIndex(i, found_sc))
                continue;

              // See if the symbol has an address.
              if (!found_sc.symbol)
                continue;

              auto addr = found_sc.symbol->GetAddress().GetLoadAddress(&target);
              if (!addr || addr == LLDB_INVALID_ADDRESS)
                continue;

              // Ask RemoteAST to get the underlying type out of the
              // descriptor.
              auto &remote_ast = GetRemoteASTContext(*swift_ast_ctx);
              auto genericParam = opaque_type->getInterfaceType()
                                      ->getAs<swift::GenericTypeParamType>();
              auto underlying_type_result =
                  remote_ast.getUnderlyingTypeForOpaqueType(
                      swift::remote::RemoteAddress(addr),
                      opaque_type->getSubstitutions(),
                      genericParam->getIndex());

              if (!underlying_type_result)
                continue;

              // If we haven't yet gotten an underlying type, use this as our
              // possible result.
              if (!result_type) {
                result_type = underlying_type_result.getValue();
              }
              // If we have two possibilities, they should match.
              else if (!result_type->isEqual(
                           underlying_type_result.getValue())) {
                return type;
              }
            }

            if (!result_type)
              return type;

            return result_type;
          },
          swift::LookUpConformanceInModule(module_decl),
          swift::SubstFlags::DesugarMemberTypes |
              swift::SubstFlags::SubstituteOpaqueArchetypes);

      // Stop if we've reached a fixpoint where we can't further resolve
      // opaque types.
      if (old_type->isEqual(target_swift_type))
        break;
    }

    target_swift_type = target_swift_type.subst(
        [this, &stack_frame,
         &swift_ast_ctx](swift::SubstitutableType *type) -> swift::Type {
          StreamString type_name;
          if (!SwiftLanguageRuntime::GetAbstractTypeName(type_name, type))
            return type;
          CompilerType concrete_type = this->GetConcreteType(
              &stack_frame, ConstString(type_name.GetString()));
          Status import_error;
          CompilerType target_concrete_type =
              swift_ast_ctx->ImportType(concrete_type, import_error);

          if (target_concrete_type.IsValid())
            return swift::Type(swift_ast_ctx->GetSwiftType(target_concrete_type));

          return type;
        },
        swift::LookUpConformanceInModule(module_decl),
        swift::SubstFlags::DesugarMemberTypes);
    assert(target_swift_type);

    return ToCompilerType({target_swift_type.getPointer()});
  }
  return base_type;
}

SwiftLanguageRuntimeImpl::MetadataPromise::MetadataPromise(
    ValueObject &for_object, SwiftLanguageRuntimeImpl &runtime,
    lldb::addr_t location)
    : m_for_object_sp(for_object.GetSP()), m_swift_runtime(runtime),
      m_metadata_location(location) {}

CompilerType
SwiftLanguageRuntimeImpl::MetadataPromise::FulfillTypePromise(Status *error) {
  if (error)
    error->Clear();

  Log *log(GetLog(LLDBLog::Types));

  if (log)
    log->Printf("[MetadataPromise] asked to fulfill type promise at location "
                "0x%" PRIx64,
                m_metadata_location);

  if (m_compiler_type.has_value())
    return m_compiler_type.value();

  llvm::Optional<SwiftScratchContextReader> maybe_swift_scratch_ctx =
      m_for_object_sp->GetSwiftScratchContext();
  if (!maybe_swift_scratch_ctx) {
    error->SetErrorString("couldn't get Swift scratch context");
    return CompilerType();
  }
  auto scratch_ctx = maybe_swift_scratch_ctx->get();
  if (!scratch_ctx) {
    error->SetErrorString("couldn't get Swift scratch context");
    return CompilerType();
  }
  SwiftASTContext *swift_ast_ctx = scratch_ctx->GetSwiftASTContext();
  if (!swift_ast_ctx) {
    error->SetErrorString("couldn't get Swift scratch context");
    return CompilerType();
  }
  auto &remote_ast = m_swift_runtime.GetRemoteASTContext(*swift_ast_ctx);
  swift::remoteAST::Result<swift::Type> result =
      remote_ast.getTypeForRemoteTypeMetadata(
          swift::remote::RemoteAddress(m_metadata_location));

  if (result) {
    m_compiler_type = {swift_ast_ctx->weak_from_this(),
                       result.getValue().getPointer()};
    if (log)
      log->Printf("[MetadataPromise] result is type %s",
                  m_compiler_type->GetTypeName().AsCString());
    return m_compiler_type.value();
  } else {
    const auto &failure = result.getFailure();
    if (error)
      error->SetErrorStringWithFormat("error in resolving type: %s",
                                      failure.render().c_str());
    if (log)
      log->Printf("[MetadataPromise] failure: %s", failure.render().c_str());
    return (m_compiler_type = CompilerType()).value();
  }
}

SwiftLanguageRuntimeImpl::MetadataPromiseSP
SwiftLanguageRuntimeImpl::GetMetadataPromise(lldb::addr_t addr,
                                             ValueObject &for_object) {
  llvm::Optional<SwiftScratchContextReader> maybe_swift_scratch_ctx =
      for_object.GetSwiftScratchContext();
  if (!maybe_swift_scratch_ctx)
    return nullptr;
  auto scratch_ctx = maybe_swift_scratch_ctx->get();
  if (!scratch_ctx)
    return nullptr;
  SwiftASTContext *swift_ast_ctx = scratch_ctx->GetSwiftASTContext();
  if (!swift_ast_ctx)
    return nullptr;
  if (swift_ast_ctx->HasFatalErrors())
    return nullptr;
  if (addr == 0 || addr == LLDB_INVALID_ADDRESS)
    return nullptr;

  auto key = std::make_pair(swift_ast_ctx->GetASTContext(), addr);
  auto iter = m_promises_map.find(key);
  if (iter != m_promises_map.end())
    return iter->second;

  SwiftLanguageRuntimeImpl::MetadataPromiseSP promise_sp(
      new SwiftLanguageRuntimeImpl::MetadataPromise(for_object, *this, addr));
  m_promises_map.insert({key, promise_sp});
  return promise_sp;
}

SwiftLanguageRuntimeImpl::MetadataPromiseSP
SwiftLanguageRuntimeImpl::GetPromiseForTypeNameAndFrame(const char *type_name,
                                                        StackFrame *frame) {
  if (!frame || !type_name || !type_name[0])
    return nullptr;

  StreamString type_metadata_ptr_var_name;
  type_metadata_ptr_var_name.Printf("$%s", type_name);
  VariableList *var_list = frame->GetVariableList(false, nullptr);
  if (!var_list)
    return nullptr;

  VariableSP var_sp(var_list->FindVariable(
      ConstString(type_metadata_ptr_var_name.GetData())));
  if (!var_sp)
    return nullptr;

  ValueObjectSP metadata_ptr_var_sp(
      frame->GetValueObjectForFrameVariable(var_sp, lldb::eNoDynamicValues));
  if (!metadata_ptr_var_sp ||
      metadata_ptr_var_sp->UpdateValueIfNeeded() == false)
    return nullptr;

  lldb::addr_t metadata_location(metadata_ptr_var_sp->GetValueAsUnsigned(0));
  if (metadata_location == 0 || metadata_location == LLDB_INVALID_ADDRESS)
    return nullptr;
  return GetMetadataPromise(metadata_location, *metadata_ptr_var_sp);
}

} // namespace lldb_private
