#include "PdbAstBuilder.h"

#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/LazyRandomTypeCollection.h"
#include "llvm/DebugInfo/CodeView/SymbolDeserializer.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/CodeView/SymbolRecordHelpers.h"
#include "llvm/DebugInfo/CodeView/TypeDeserializer.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"
#include "llvm/Demangle/MicrosoftDemangle.h"

#include "lldb/Core/Module.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangExternalASTSourceCommon.h"
#include "lldb/Symbol/ClangUtil.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/LLDBAssert.h"

#include "PdbUtil.h"
#include "UdtRecordCompleter.h"

using namespace lldb_private;
using namespace lldb_private::npdb;
using namespace llvm::codeview;
using namespace llvm::pdb;

static llvm::Optional<PdbCompilandSymId> FindSymbolScope(PdbIndex &index,
                                                         PdbCompilandSymId id) {
  CVSymbol sym = index.ReadSymbolRecord(id);
  if (symbolOpensScope(sym.kind())) {
    // If this exact symbol opens a scope, we can just directly access its
    // parent.
    id.offset = getScopeParentOffset(sym);
    // Global symbols have parent offset of 0.  Return llvm::None to indicate
    // this.
    if (id.offset == 0)
      return llvm::None;
    return id;
  }

  // Otherwise we need to start at the beginning and iterate forward until we
  // reach (or pass) this particular symbol
  CompilandIndexItem &cii = index.compilands().GetOrCreateCompiland(id.modi);
  const CVSymbolArray &syms = cii.m_debug_stream.getSymbolArray();

  auto begin = syms.begin();
  auto end = syms.at(id.offset);
  std::vector<PdbCompilandSymId> scope_stack;

  while (begin != end) {
    if (id.offset == begin.offset()) {
      // We have a match!  Return the top of the stack
      if (scope_stack.empty())
        return llvm::None;
      return scope_stack.back();
    }
    if (begin.offset() > id.offset) {
      // We passed it.  We couldn't even find this symbol record.
      lldbassert(false && "Invalid compiland symbol id!");
      return llvm::None;
    }

    // We haven't found the symbol yet.  Check if we need to open or close the
    // scope stack.
    if (symbolOpensScope(begin->kind())) {
      // We can use the end offset of the scope to determine whether or not
      // we can just outright skip this entire scope.
      uint32_t scope_end = getScopeEndOffset(*begin);
      if (scope_end < id.modi) {
        begin = syms.at(scope_end);
      } else {
        // The symbol we're looking for is somewhere in this scope.
        scope_stack.emplace_back(id.modi, begin.offset());
      }
    } else if (symbolEndsScope(begin->kind())) {
      scope_stack.pop_back();
    }
    ++begin;
  }

  return llvm::None;
}

static clang::TagTypeKind TranslateUdtKind(const TagRecord &cr) {
  switch (cr.Kind) {
  case TypeRecordKind::Class:
    return clang::TTK_Class;
  case TypeRecordKind::Struct:
    return clang::TTK_Struct;
  case TypeRecordKind::Union:
    return clang::TTK_Union;
  case TypeRecordKind::Interface:
    return clang::TTK_Interface;
  case TypeRecordKind::Enum:
    return clang::TTK_Enum;
  default:
    lldbassert(false && "Invalid tag record kind!");
    return clang::TTK_Struct;
  }
}

static bool IsCVarArgsFunction(llvm::ArrayRef<TypeIndex> args) {
  if (args.empty())
    return false;
  return args.back() == TypeIndex::None();
}

static bool
AnyScopesHaveTemplateParams(llvm::ArrayRef<llvm::ms_demangle::Node *> scopes) {
  for (llvm::ms_demangle::Node *n : scopes) {
    auto *idn = static_cast<llvm::ms_demangle::IdentifierNode *>(n);
    if (idn->TemplateParams)
      return true;
  }
  return false;
}

static ClangASTContext &GetClangASTContext(ObjectFile &obj) {
  TypeSystem *ts =
      obj.GetModule()->GetTypeSystemForLanguage(lldb::eLanguageTypeC_plus_plus);
  lldbassert(ts);
  return static_cast<ClangASTContext &>(*ts);
}

static llvm::Optional<clang::CallingConv>
TranslateCallingConvention(llvm::codeview::CallingConvention conv) {
  using CC = llvm::codeview::CallingConvention;
  switch (conv) {

  case CC::NearC:
  case CC::FarC:
    return clang::CallingConv::CC_C;
  case CC::NearPascal:
  case CC::FarPascal:
    return clang::CallingConv::CC_X86Pascal;
  case CC::NearFast:
  case CC::FarFast:
    return clang::CallingConv::CC_X86FastCall;
  case CC::NearStdCall:
  case CC::FarStdCall:
    return clang::CallingConv::CC_X86StdCall;
  case CC::ThisCall:
    return clang::CallingConv::CC_X86ThisCall;
  case CC::NearVector:
    return clang::CallingConv::CC_X86VectorCall;
  default:
    return llvm::None;
  }
}

static llvm::Optional<CVTagRecord>
GetNestedTagRecord(const NestedTypeRecord &Record, const CVTagRecord &parent,
                   TpiStream &tpi) {
  // An LF_NESTTYPE is essentially a nested typedef / using declaration, but it
  // is also used to indicate the primary definition of a nested class.  That is
  // to say, if you have:
  // struct A {
  //   struct B {};
  //   using C = B;
  // };
  // Then in the debug info, this will appear as:
  // LF_STRUCTURE `A::B` [type index = N]
  // LF_STRUCTURE `A`
  //   LF_NESTTYPE [name = `B`, index = N]
  //   LF_NESTTYPE [name = `C`, index = N]
  // In order to accurately reconstruct the decl context hierarchy, we need to
  // know which ones are actual definitions and which ones are just aliases.

  // If it's a simple type, then this is something like `using foo = int`.
  if (Record.Type.isSimple())
    return llvm::None;

  CVType cvt = tpi.getType(Record.Type);

  if (!IsTagRecord(cvt))
    return llvm::None;

  // If it's an inner definition, then treat whatever name we have here as a
  // single component of a mangled name.  So we can inject it into the parent's
  // mangled name to see if it matches.
  CVTagRecord child = CVTagRecord::create(cvt);
  std::string qname = parent.asTag().getUniqueName();
  if (qname.size() < 4 || child.asTag().getUniqueName().size() < 4)
    return llvm::None;

  // qname[3] is the tag type identifier (struct, class, union, etc).  Since the
  // inner tag type is not necessarily the same as the outer tag type, re-write
  // it to match the inner tag type.
  qname[3] = child.asTag().getUniqueName()[3];
  std::string piece = Record.Name;
  piece.push_back('@');
  qname.insert(4, std::move(piece));
  if (qname != child.asTag().UniqueName)
    return llvm::None;

  return std::move(child);
}

PdbAstBuilder::PdbAstBuilder(ObjectFile &obj, PdbIndex &index)
    : m_index(index), m_clang(GetClangASTContext(obj)) {
  BuildParentMap();
}

clang::DeclContext &PdbAstBuilder::GetTranslationUnitDecl() {
  return *m_clang.GetTranslationUnitDecl();
}

std::pair<clang::DeclContext *, std::string>
PdbAstBuilder::CreateDeclInfoForType(const TagRecord &record, TypeIndex ti) {
  // FIXME: Move this to GetDeclContextContainingUID.

  llvm::ms_demangle::Demangler demangler;
  StringView sv(record.UniqueName.begin(), record.UniqueName.size());
  llvm::ms_demangle::TagTypeNode *ttn = demangler.parseTagUniqueName(sv);
  llvm::ms_demangle::IdentifierNode *idn =
      ttn->QualifiedName->getUnqualifiedIdentifier();
  std::string uname = idn->toString();

  llvm::ms_demangle::NodeArrayNode *name_components =
      ttn->QualifiedName->Components;
  llvm::ArrayRef<llvm::ms_demangle::Node *> scopes(name_components->Nodes,
                                                   name_components->Count - 1);

  clang::DeclContext *context = m_clang.GetTranslationUnitDecl();

  // If this type doesn't have a parent type in the debug info, then the best we
  // can do is to say that it's either a series of namespaces (if the scope is
  // non-empty), or the translation unit (if the scope is empty).
  auto parent_iter = m_parent_types.find(ti);
  if (parent_iter == m_parent_types.end()) {
    if (scopes.empty())
      return {context, uname};

    // If there is no parent in the debug info, but some of the scopes have
    // template params, then this is a case of bad debug info.  See, for
    // example, llvm.org/pr39607.  We don't want to create an ambiguity between
    // a NamespaceDecl and a CXXRecordDecl, so instead we create a class at
    // global scope with the fully qualified name.
    if (AnyScopesHaveTemplateParams(scopes))
      return {context, record.Name};

    for (llvm::ms_demangle::Node *scope : scopes) {
      auto *nii = static_cast<llvm::ms_demangle::NamedIdentifierNode *>(scope);
      std::string str = nii->toString();
      context = m_clang.GetUniqueNamespaceDeclaration(str.c_str(), context);
    }
    return {context, uname};
  }

  // Otherwise, all we need to do is get the parent type of this type and
  // recurse into our lazy type creation / AST reconstruction logic to get an
  // LLDB TypeSP for the parent.  This will cause the AST to automatically get
  // the right DeclContext created for any parent.
  clang::QualType parent_qt = GetOrCreateType(parent_iter->second);

  context = clang::TagDecl::castToDeclContext(parent_qt->getAsTagDecl());
  return {context, uname};
}

void PdbAstBuilder::BuildParentMap() {
  LazyRandomTypeCollection &types = m_index.tpi().typeCollection();

  for (auto ti = types.getFirst(); ti; ti = types.getNext(*ti)) {
    CVType type = types.getType(*ti);
    if (!IsTagRecord(type))
      continue;

    CVTagRecord tag = CVTagRecord::create(type);
    // We're looking for LF_NESTTYPE records in the field list, so ignore
    // forward references (no field list), and anything without a nested class
    // (since there won't be any LF_NESTTYPE records).
    if (tag.asTag().isForwardRef() || !tag.asTag().containsNestedClass())
      continue;

    struct ProcessTpiStream : public TypeVisitorCallbacks {
      ProcessTpiStream(PdbIndex &index, TypeIndex parent,
                       const CVTagRecord &parent_cvt,
                       llvm::DenseMap<TypeIndex, TypeIndex> &parents)
          : index(index), parents(parents), parent(parent),
            parent_cvt(parent_cvt) {}

      PdbIndex &index;
      llvm::DenseMap<TypeIndex, TypeIndex> &parents;
      TypeIndex parent;
      const CVTagRecord &parent_cvt;

      llvm::Error visitKnownMember(CVMemberRecord &CVR,
                                   NestedTypeRecord &Record) override {
        llvm::Optional<CVTagRecord> tag =
            GetNestedTagRecord(Record, parent_cvt, index.tpi());
        if (!tag)
          return llvm::ErrorSuccess();

        parents[Record.Type] = parent;
        if (!tag->asTag().isForwardRef())
          return llvm::ErrorSuccess();

        llvm::Expected<TypeIndex> full_decl =
            index.tpi().findFullDeclForForwardRef(Record.Type);
        if (!full_decl) {
          llvm::consumeError(full_decl.takeError());
          return llvm::ErrorSuccess();
        }
        parents[*full_decl] = parent;
        return llvm::ErrorSuccess();
      }
    };

    CVType field_list = m_index.tpi().getType(tag.asTag().FieldList);
    ProcessTpiStream process(m_index, *ti, tag, m_parent_types);
    llvm::Error error = visitMemberRecordStream(field_list.data(), process);
    if (error)
      llvm::consumeError(std::move(error));
  }
}

clang::Decl *PdbAstBuilder::GetOrCreateSymbolForId(PdbCompilandSymId id) {
  CVSymbol cvs = m_index.ReadSymbolRecord(id);

  switch (cvs.kind()) {
  case S_GPROC32:
  case S_LPROC32:
    return GetOrCreateFunctionDecl(id);
  case S_GDATA32:
  case S_LDATA32:
  case S_GTHREAD32:
  case S_CONSTANT:
    // global variable
    return nullptr;
  case S_BLOCK32:
    return GetOrCreateBlockDecl(id);
  case S_REGISTER:
  case S_REGREL32:
  case S_LOCAL: {
    clang::DeclContext *scope = GetParentDeclContext(id);
    clang::Decl *scope_decl = clang::Decl::castFromDeclContext(scope);
    PdbCompilandSymId scope_id(id.modi, m_decl_to_status[scope_decl].uid);
    return GetOrCreateLocalVariableDecl(scope_id, id);
  }
  default:
    return nullptr;
  }
}

clang::Decl *PdbAstBuilder::GetOrCreateDeclForUid(PdbSymUid uid) {
  if (clang::Decl *result = TryGetDecl(uid))
    return result;

  clang::Decl *result = nullptr;
  switch (uid.kind()) {
  case PdbSymUidKind::CompilandSym:
    result = GetOrCreateSymbolForId(uid.asCompilandSym());
    break;
  case PdbSymUidKind::Type: {
    clang::QualType qt = GetOrCreateType(uid.asTypeSym());
    if (auto *tag = qt->getAsTagDecl()) {
      result = tag;
      break;
    }
    return nullptr;
  }
  default:
    return nullptr;
  }
  m_uid_to_decl[toOpaqueUid(uid)] = result;
  return result;
}

clang::DeclContext *PdbAstBuilder::GetOrCreateDeclContextForUid(PdbSymUid uid) {
  clang::Decl *decl = GetOrCreateDeclForUid(uid);
  if (!decl)
    return nullptr;

  return clang::Decl::castToDeclContext(decl);
}

clang::DeclContext *PdbAstBuilder::GetParentDeclContext(PdbSymUid uid) {
  // We must do this *without* calling GetOrCreate on the current uid, as
  // that would be an infinite recursion.
  switch (uid.kind()) {
  case PdbSymUidKind::CompilandSym: {
    llvm::Optional<PdbCompilandSymId> scope =
        FindSymbolScope(m_index, uid.asCompilandSym());
    if (!scope)
      return &GetTranslationUnitDecl();
    return GetOrCreateDeclContextForUid(*scope);
  }
  case PdbSymUidKind::Type:
    // It could be a namespace, class, or global.  We don't support nested
    // functions yet.  Anyway, we just need to consult the parent type map.
    break;
  case PdbSymUidKind::FieldListMember:
    // In this case the parent DeclContext is the one for the class that this
    // member is inside of.
    break;
  default:
    break;
  }
  return &GetTranslationUnitDecl();
}

bool PdbAstBuilder::CompleteType(clang::QualType qt) {
  clang::TagDecl *tag = qt->getAsTagDecl();
  if (!tag)
    return false;

  return CompleteTagDecl(*tag);
}

bool PdbAstBuilder::CompleteTagDecl(clang::TagDecl &tag) {
  // If this is not in our map, it's an error.
  auto status_iter = m_decl_to_status.find(&tag);
  lldbassert(status_iter != m_decl_to_status.end());

  // If it's already complete, just return.
  DeclStatus &status = status_iter->second;
  if (status.resolved)
    return true;

  PdbTypeSymId type_id = PdbSymUid(status.uid).asTypeSym();

  lldbassert(IsTagRecord(type_id, m_index.tpi()));

  clang::QualType tag_qt = m_clang.getASTContext()->getTypeDeclType(&tag);
  ClangASTContext::SetHasExternalStorage(tag_qt.getAsOpaquePtr(), false);

  TypeIndex tag_ti = type_id.index;
  CVType cvt = m_index.tpi().getType(tag_ti);
  if (cvt.kind() == LF_MODIFIER)
    tag_ti = LookThroughModifierRecord(cvt);

  PdbTypeSymId best_ti = GetBestPossibleDecl(tag_ti, m_index.tpi());
  cvt = m_index.tpi().getType(best_ti.index);
  lldbassert(IsTagRecord(cvt));

  if (IsForwardRefUdt(cvt)) {
    // If we can't find a full decl for this forward ref anywhere in the debug
    // info, then we have no way to complete it.
    return false;
  }

  TypeIndex field_list_ti = GetFieldListIndex(cvt);
  CVType field_list_cvt = m_index.tpi().getType(field_list_ti);
  if (field_list_cvt.kind() != LF_FIELDLIST)
    return false;

  // Visit all members of this class, then perform any finalization necessary
  // to complete the class.
  CompilerType ct = ToCompilerType(tag_qt);
  UdtRecordCompleter completer(best_ti, ct, tag, *this, m_index.tpi());
  auto error =
      llvm::codeview::visitMemberRecordStream(field_list_cvt.data(), completer);
  completer.complete();

  status.resolved = true;
  if (!error)
    return true;

  llvm::consumeError(std::move(error));
  return false;
}

clang::QualType PdbAstBuilder::CreateSimpleType(TypeIndex ti) {
  if (ti == TypeIndex::NullptrT())
    return GetBasicType(lldb::eBasicTypeNullPtr);

  if (ti.getSimpleMode() != SimpleTypeMode::Direct) {
    clang::QualType direct_type = GetOrCreateType(ti.makeDirect());
    return m_clang.getASTContext()->getPointerType(direct_type);
  }

  if (ti.getSimpleKind() == SimpleTypeKind::NotTranslated)
    return {};

  lldb::BasicType bt = GetCompilerTypeForSimpleKind(ti.getSimpleKind());
  if (bt == lldb::eBasicTypeInvalid)
    return {};

  return GetBasicType(bt);
}

clang::QualType PdbAstBuilder::CreatePointerType(const PointerRecord &pointer) {
  clang::QualType pointee_type = GetOrCreateType(pointer.ReferentType);

  if (pointer.isPointerToMember()) {
    MemberPointerInfo mpi = pointer.getMemberInfo();
    clang::QualType class_type = GetOrCreateType(mpi.ContainingType);

    return m_clang.getASTContext()->getMemberPointerType(
        pointee_type, class_type.getTypePtr());
  }

  clang::QualType pointer_type;
  if (pointer.getMode() == PointerMode::LValueReference)
    pointer_type =
        m_clang.getASTContext()->getLValueReferenceType(pointee_type);
  else if (pointer.getMode() == PointerMode::RValueReference)
    pointer_type =
        m_clang.getASTContext()->getRValueReferenceType(pointee_type);
  else
    pointer_type = m_clang.getASTContext()->getPointerType(pointee_type);

  if ((pointer.getOptions() & PointerOptions::Const) != PointerOptions::None)
    pointer_type.addConst();

  if ((pointer.getOptions() & PointerOptions::Volatile) != PointerOptions::None)
    pointer_type.addVolatile();

  if ((pointer.getOptions() & PointerOptions::Restrict) != PointerOptions::None)
    pointer_type.addRestrict();

  return pointer_type;
}

clang::QualType
PdbAstBuilder::CreateModifierType(const ModifierRecord &modifier) {

  clang::QualType unmodified_type = GetOrCreateType(modifier.ModifiedType);

  if ((modifier.Modifiers & ModifierOptions::Const) != ModifierOptions::None)
    unmodified_type.addConst();
  if ((modifier.Modifiers & ModifierOptions::Volatile) != ModifierOptions::None)
    unmodified_type.addVolatile();

  return unmodified_type;
}

clang::QualType PdbAstBuilder::CreateRecordType(PdbTypeSymId id,
                                                const TagRecord &record) {
  clang::DeclContext *decl_context = nullptr;
  std::string uname;
  std::tie(decl_context, uname) = CreateDeclInfoForType(record, id.index);

  clang::TagTypeKind ttk = TranslateUdtKind(record);
  lldb::AccessType access =
      (ttk == clang::TTK_Class) ? lldb::eAccessPrivate : lldb::eAccessPublic;

  ClangASTMetadata metadata;
  metadata.SetUserID(toOpaqueUid(id));
  metadata.SetIsDynamicCXXType(false);

  CompilerType ct =
      m_clang.CreateRecordType(decl_context, access, uname.c_str(), ttk,
                               lldb::eLanguageTypeC_plus_plus, &metadata);

  lldbassert(ct.IsValid());

  ClangASTContext::StartTagDeclarationDefinition(ct);

  // Even if it's possible, don't complete it at this point. Just mark it
  // forward resolved, and if/when LLDB needs the full definition, it can
  // ask us.
  clang::QualType result =
      clang::QualType::getFromOpaquePtr(ct.GetOpaqueQualType());

  ClangASTContext::SetHasExternalStorage(result.getAsOpaquePtr(), true);
  return result;
}

clang::Decl *PdbAstBuilder::TryGetDecl(PdbSymUid uid) const {
  auto iter = m_uid_to_decl.find(toOpaqueUid(uid));
  if (iter != m_uid_to_decl.end())
    return iter->second;
  return nullptr;
}

clang::NamespaceDecl *
PdbAstBuilder::GetOrCreateNamespaceDecl(llvm::StringRef name,
                                        clang::DeclContext &context) {
  return m_clang.GetUniqueNamespaceDeclaration(name.str().c_str(), &context);
}

clang::BlockDecl *
PdbAstBuilder::GetOrCreateBlockDecl(PdbCompilandSymId block_id) {
  if (clang::Decl *decl = TryGetDecl(block_id))
    return llvm::dyn_cast<clang::BlockDecl>(decl);

  clang::DeclContext *scope = GetParentDeclContext(block_id);

  clang::BlockDecl *block_decl = m_clang.CreateBlockDeclaration(scope);
  m_uid_to_decl.insert({toOpaqueUid(block_id), block_decl});
  return block_decl;
}

clang::VarDecl *
PdbAstBuilder::GetOrCreateLocalVariableDecl(PdbCompilandSymId scope_id,
                                            PdbCompilandSymId var_id) {
  if (clang::Decl *decl = TryGetDecl(var_id))
    return llvm::dyn_cast<clang::VarDecl>(decl);

  clang::DeclContext *scope = GetOrCreateDeclContextForUid(scope_id);

  CVSymbol var = m_index.ReadSymbolRecord(var_id);
  VariableInfo var_info = GetVariableNameInfo(var);
  clang::QualType qt = GetOrCreateType(var_info.type);

  clang::VarDecl *var_decl =
      m_clang.CreateVariableDeclaration(scope, var_info.name.str().c_str(), qt);

  m_uid_to_decl[toOpaqueUid(var_id)] = var_decl;
  return var_decl;
}

clang::QualType PdbAstBuilder::GetBasicType(lldb::BasicType type) {
  CompilerType ct = m_clang.GetBasicType(type);
  return clang::QualType::getFromOpaquePtr(ct.GetOpaqueQualType());
}

clang::QualType PdbAstBuilder::CreateType(PdbTypeSymId type) {
  if (type.index.isSimple())
    return CreateSimpleType(type.index);

  CVType cvt = m_index.tpi().getType(type.index);

  if (cvt.kind() == LF_MODIFIER) {
    ModifierRecord modifier;
    llvm::cantFail(
        TypeDeserializer::deserializeAs<ModifierRecord>(cvt, modifier));
    return CreateModifierType(modifier);
  }

  if (cvt.kind() == LF_POINTER) {
    PointerRecord pointer;
    llvm::cantFail(
        TypeDeserializer::deserializeAs<PointerRecord>(cvt, pointer));
    return CreatePointerType(pointer);
  }

  if (IsTagRecord(cvt)) {
    CVTagRecord tag = CVTagRecord::create(cvt);
    if (tag.kind() == CVTagRecord::Union)
      return CreateRecordType(type.index, tag.asUnion());
    if (tag.kind() == CVTagRecord::Enum)
      return CreateEnumType(type.index, tag.asEnum());
    return CreateRecordType(type.index, tag.asClass());
  }

  if (cvt.kind() == LF_ARRAY) {
    ArrayRecord ar;
    llvm::cantFail(TypeDeserializer::deserializeAs<ArrayRecord>(cvt, ar));
    return CreateArrayType(ar);
  }

  if (cvt.kind() == LF_PROCEDURE) {
    ProcedureRecord pr;
    llvm::cantFail(TypeDeserializer::deserializeAs<ProcedureRecord>(cvt, pr));
    return CreateProcedureType(pr);
  }

  return {};
}

clang::QualType PdbAstBuilder::GetOrCreateType(PdbTypeSymId type) {
  lldb::user_id_t uid = toOpaqueUid(type);
  auto iter = m_uid_to_type.find(uid);
  if (iter != m_uid_to_type.end())
    return iter->second;

  PdbTypeSymId best_type = GetBestPossibleDecl(type, m_index.tpi());

  clang::QualType qt;
  if (best_type.index != type.index) {
    // This is a forward decl.  Call GetOrCreate on the full decl, then map the
    // forward decl id to the full decl QualType.
    clang::QualType qt = GetOrCreateType(best_type);
    m_uid_to_type[toOpaqueUid(type)] = qt;
    return qt;
  }

  // This is either a full decl, or a forward decl with no matching full decl
  // in the debug info.
  qt = CreateType(type);
  m_uid_to_type[toOpaqueUid(type)] = qt;
  if (IsTagRecord(type, m_index.tpi())) {
    clang::TagDecl *tag = qt->getAsTagDecl();
    lldbassert(m_decl_to_status.count(tag) == 0);

    DeclStatus &status = m_decl_to_status[tag];
    status.uid = uid;
    status.resolved = false;
  }
  return qt;
}

clang::FunctionDecl *
PdbAstBuilder::GetOrCreateFunctionDecl(PdbCompilandSymId func_id) {
  if (clang::Decl *decl = TryGetDecl(func_id))
    return llvm::dyn_cast<clang::FunctionDecl>(decl);

  clang::DeclContext *parent = GetParentDeclContext(PdbSymUid(func_id));

  CVSymbol cvs = m_index.ReadSymbolRecord(func_id);
  ProcSym proc(static_cast<SymbolRecordKind>(cvs.kind()));
  llvm::cantFail(SymbolDeserializer::deserializeAs<ProcSym>(cvs, proc));

  PdbTypeSymId type_id(proc.FunctionType);
  clang::QualType qt = GetOrCreateType(type_id);

  clang::StorageClass storage = clang::SC_None;
  if (proc.Kind == SymbolRecordKind::ProcSym)
    storage = clang::SC_Static;

  const clang::FunctionProtoType *func_type =
      llvm::dyn_cast<clang::FunctionProtoType>(qt);

  CompilerType func_ct = ToCompilerType(qt);

  clang::FunctionDecl *function_decl = m_clang.CreateFunctionDeclaration(
      parent, proc.Name.str().c_str(), func_ct, storage, false);

  lldbassert(m_uid_to_decl.count(toOpaqueUid(func_id)) == 0);
  m_uid_to_decl[toOpaqueUid(func_id)] = function_decl;

  CreateFunctionParameters(func_id, *function_decl, func_type->getNumParams());

  return function_decl;
}

void PdbAstBuilder::CreateFunctionParameters(PdbCompilandSymId func_id,
                                             clang::FunctionDecl &function_decl,
                                             uint32_t param_count) {
  CompilandIndexItem *cii = m_index.compilands().GetCompiland(func_id.modi);
  CVSymbolArray scope =
      cii->m_debug_stream.getSymbolArrayForScope(func_id.offset);

  auto begin = scope.begin();
  auto end = scope.end();
  std::vector<clang::ParmVarDecl *> params;
  while (begin != end && param_count > 0) {
    uint32_t record_offset = begin.offset();
    CVSymbol sym = *begin++;

    TypeIndex param_type;
    llvm::StringRef param_name;
    switch (sym.kind()) {
    case S_REGREL32: {
      RegRelativeSym reg(SymbolRecordKind::RegRelativeSym);
      cantFail(SymbolDeserializer::deserializeAs<RegRelativeSym>(sym, reg));
      param_type = reg.Type;
      param_name = reg.Name;
      break;
    }
    case S_REGISTER: {
      RegisterSym reg(SymbolRecordKind::RegisterSym);
      cantFail(SymbolDeserializer::deserializeAs<RegisterSym>(sym, reg));
      param_type = reg.Index;
      param_name = reg.Name;
      break;
    }
    case S_LOCAL: {
      LocalSym local(SymbolRecordKind::LocalSym);
      cantFail(SymbolDeserializer::deserializeAs<LocalSym>(sym, local));
      if ((local.Flags & LocalSymFlags::IsParameter) == LocalSymFlags::None)
        continue;
      param_type = local.Type;
      param_name = local.Name;
      break;
    }
    case S_BLOCK32:
      // All parameters should come before the first block.  If that isn't the
      // case, then perhaps this is bad debug info that doesn't contain
      // information about all parameters.
      return;
    default:
      continue;
    }

    PdbCompilandSymId param_uid(func_id.modi, record_offset);
    clang::QualType qt = GetOrCreateType(param_type);

    CompilerType param_type_ct(&m_clang, qt.getAsOpaquePtr());
    clang::ParmVarDecl *param = m_clang.CreateParameterDeclaration(
        &function_decl, param_name.str().c_str(), param_type_ct,
        clang::SC_None);
    lldbassert(m_uid_to_decl.count(toOpaqueUid(param_uid)) == 0);

    m_uid_to_decl[toOpaqueUid(param_uid)] = param;
    params.push_back(param);
    --param_count;
  }

  if (!params.empty())
    m_clang.SetFunctionParameters(&function_decl, params.data(), params.size());
}

clang::QualType PdbAstBuilder::CreateEnumType(PdbTypeSymId id,
                                              const EnumRecord &er) {
  clang::DeclContext *decl_context = nullptr;
  std::string uname;
  std::tie(decl_context, uname) = CreateDeclInfoForType(er, id.index);
  clang::QualType underlying_type = GetOrCreateType(er.UnderlyingType);

  Declaration declaration;
  CompilerType enum_ct = m_clang.CreateEnumerationType(
      uname.c_str(), decl_context, declaration, ToCompilerType(underlying_type),
      er.isScoped());

  ClangASTContext::StartTagDeclarationDefinition(enum_ct);
  ClangASTContext::SetHasExternalStorage(enum_ct.GetOpaqueQualType(), true);

  return clang::QualType::getFromOpaquePtr(enum_ct.GetOpaqueQualType());
}

clang::QualType PdbAstBuilder::CreateArrayType(const ArrayRecord &ar) {
  clang::QualType element_type = GetOrCreateType(ar.ElementType);

  uint64_t element_count =
      ar.Size / GetSizeOfType({ar.ElementType}, m_index.tpi());

  CompilerType array_ct = m_clang.CreateArrayType(ToCompilerType(element_type),
                                                  element_count, false);
  return clang::QualType::getFromOpaquePtr(array_ct.GetOpaqueQualType());
}

clang::QualType
PdbAstBuilder::CreateProcedureType(const ProcedureRecord &proc) {
  TpiStream &stream = m_index.tpi();
  CVType args_cvt = stream.getType(proc.ArgumentList);
  ArgListRecord args;
  llvm::cantFail(
      TypeDeserializer::deserializeAs<ArgListRecord>(args_cvt, args));

  llvm::ArrayRef<TypeIndex> arg_indices = llvm::makeArrayRef(args.ArgIndices);
  bool is_variadic = IsCVarArgsFunction(arg_indices);
  if (is_variadic)
    arg_indices = arg_indices.drop_back();

  std::vector<CompilerType> arg_types;
  arg_types.reserve(arg_indices.size());

  for (TypeIndex arg_index : arg_indices) {
    clang::QualType arg_type = GetOrCreateType(arg_index);
    arg_types.push_back(ToCompilerType(arg_type));
  }

  clang::QualType return_type = GetOrCreateType(proc.ReturnType);

  llvm::Optional<clang::CallingConv> cc =
      TranslateCallingConvention(proc.CallConv);
  if (!cc)
    return {};

  CompilerType return_ct = ToCompilerType(return_type);
  CompilerType func_sig_ast_type = m_clang.CreateFunctionType(
      return_ct, arg_types.data(), arg_types.size(), is_variadic, 0, *cc);

  return clang::QualType::getFromOpaquePtr(
      func_sig_ast_type.GetOpaqueQualType());
}

CompilerDecl PdbAstBuilder::ToCompilerDecl(clang::Decl &decl) {
  return {&m_clang, &decl};
}

CompilerType PdbAstBuilder::ToCompilerType(clang::QualType qt) {
  return {&m_clang, qt.getAsOpaquePtr()};
}

CompilerDeclContext
PdbAstBuilder::ToCompilerDeclContext(clang::DeclContext &context) {
  return {&m_clang, &context};
}

void PdbAstBuilder::Dump(Stream &stream) { m_clang.Dump(stream); }
