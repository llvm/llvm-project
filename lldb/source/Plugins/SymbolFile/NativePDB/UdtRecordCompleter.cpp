#include "UdtRecordCompleter.h"

#include "PdbAstBuilder.h"
#include "PdbIndex.h"
#include "PdbSymUid.h"
#include "PdbUtil.h"

#include "Plugins/ExpressionParser/Clang/ClangASTImporter.h"
#include "Plugins/ExpressionParser/Clang/ClangUtil.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"

#include "llvm/DebugInfo/CodeView/SymbolDeserializer.h"
#include "llvm/DebugInfo/CodeView/TypeDeserializer.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/PDB/Native/GlobalsStream.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"

using namespace llvm::codeview;
using namespace llvm::pdb;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::npdb;

using Error = llvm::Error;

UdtRecordCompleter::UdtRecordCompleter(
    PdbTypeSymId id, CompilerType &derived_ct, clang::TagDecl &tag_decl,
    PdbAstBuilder &ast_builder, PdbIndex &index,
    llvm::DenseMap<lldb::opaque_compiler_type_t,
                   llvm::SmallSet<std::pair<llvm::StringRef, CompilerType>, 8>>
        &cxx_record_map)
    : m_id(id), m_derived_ct(derived_ct), m_tag_decl(tag_decl),
      m_ast_builder(ast_builder), m_index(index),
      m_cxx_record_map(cxx_record_map) {
  CVType cvt = m_index.tpi().getType(m_id.index);
  switch (cvt.kind()) {
  case LF_ENUM:
    llvm::cantFail(TypeDeserializer::deserializeAs<EnumRecord>(cvt, m_cvr.er));
    break;
  case LF_UNION:
    llvm::cantFail(TypeDeserializer::deserializeAs<UnionRecord>(cvt, m_cvr.ur));
    break;
  case LF_CLASS:
  case LF_STRUCTURE:
    llvm::cantFail(TypeDeserializer::deserializeAs<ClassRecord>(cvt, m_cvr.cr));
    break;
  default:
    llvm_unreachable("unreachable!");
  }
}

clang::QualType UdtRecordCompleter::AddBaseClassForTypeIndex(
    llvm::codeview::TypeIndex ti, llvm::codeview::MemberAccess access,
    llvm::Optional<uint64_t> vtable_idx) {
  PdbTypeSymId type_id(ti);
  clang::QualType qt = m_ast_builder.GetOrCreateType(type_id);

  CVType udt_cvt = m_index.tpi().getType(ti);

  std::unique_ptr<clang::CXXBaseSpecifier> base_spec =
      m_ast_builder.clang().CreateBaseClassSpecifier(
          qt.getAsOpaquePtr(), TranslateMemberAccess(access),
          vtable_idx.hasValue(), udt_cvt.kind() == LF_CLASS);
  lldbassert(base_spec);

  m_bases.push_back(
      std::make_pair(vtable_idx.getValueOr(0), std::move(base_spec)));

  return qt;
}

void UdtRecordCompleter::AddMethod(llvm::StringRef name, TypeIndex type_idx,
                                   MemberAccess access, MethodOptions options,
                                   MemberAttributes attrs) {
  clang::QualType method_qt =
      m_ast_builder.GetOrCreateType(PdbTypeSymId(type_idx));
  m_ast_builder.CompleteType(method_qt);
  CompilerType method_ct = m_ast_builder.ToCompilerType(method_qt);
  lldb::opaque_compiler_type_t derived_opaque_ty = m_derived_ct.GetOpaqueQualType();
  auto iter = m_cxx_record_map.find(derived_opaque_ty);
  if (iter != m_cxx_record_map.end()) {
    if (iter->getSecond().contains({name, method_ct})) {
      return;
    }
  }

  lldb::AccessType access_type = TranslateMemberAccess(access);
  bool is_artificial = (options & MethodOptions::CompilerGenerated) ==
                       MethodOptions::CompilerGenerated;
  m_ast_builder.clang().AddMethodToCXXRecordType(
      derived_opaque_ty, name.data(), nullptr, method_ct,
      access_type, attrs.isVirtual(), attrs.isStatic(), false, false, false,
      is_artificial);

  m_cxx_record_map[derived_opaque_ty].insert({name, method_ct});
}

Error UdtRecordCompleter::visitKnownMember(CVMemberRecord &cvr,
                                           BaseClassRecord &base) {
  clang::QualType base_qt =
      AddBaseClassForTypeIndex(base.Type, base.getAccess());

  auto decl =
      m_ast_builder.clang().GetAsCXXRecordDecl(base_qt.getAsOpaquePtr());
  lldbassert(decl);

  auto offset = clang::CharUnits::fromQuantity(base.getBaseOffset());
  m_layout.base_offsets.insert(std::make_pair(decl, offset));

  return llvm::Error::success();
}

Error UdtRecordCompleter::visitKnownMember(CVMemberRecord &cvr,
                                           VirtualBaseClassRecord &base) {
  AddBaseClassForTypeIndex(base.BaseType, base.getAccess(), base.VTableIndex);

  return Error::success();
}

Error UdtRecordCompleter::visitKnownMember(CVMemberRecord &cvr,
                                           ListContinuationRecord &cont) {
  return Error::success();
}

Error UdtRecordCompleter::visitKnownMember(CVMemberRecord &cvr,
                                           VFPtrRecord &vfptr) {
  return Error::success();
}

Error UdtRecordCompleter::visitKnownMember(
    CVMemberRecord &cvr, StaticDataMemberRecord &static_data_member) {
  clang::QualType member_type =
      m_ast_builder.GetOrCreateType(PdbTypeSymId(static_data_member.Type));

  m_ast_builder.CompleteType(member_type);

  CompilerType member_ct = m_ast_builder.ToCompilerType(member_type);

  lldb::AccessType access =
      TranslateMemberAccess(static_data_member.getAccess());
  auto decl = TypeSystemClang::AddVariableToRecordType(
      m_derived_ct, static_data_member.Name, member_ct, access);

  // Static constant members may be a const[expr] declaration.
  // Query the symbol's value as the variable initializer if valid.
  if (member_ct.IsConst()) {
    std::string qual_name = decl->getQualifiedNameAsString();

    auto results =
        m_index.globals().findRecordsByName(qual_name, m_index.symrecords());

    for (const auto &result : results) {
      if (result.second.kind() == SymbolKind::S_CONSTANT) {
        ConstantSym constant(SymbolRecordKind::ConstantSym);
        cantFail(SymbolDeserializer::deserializeAs<ConstantSym>(result.second,
                                                                constant));

        clang::QualType qual_type = decl->getType();
        unsigned type_width = decl->getASTContext().getIntWidth(qual_type);
        unsigned constant_width = constant.Value.getBitWidth();

        if (qual_type->isIntegralOrEnumerationType()) {
          if (type_width >= constant_width) {
            TypeSystemClang::SetIntegerInitializerForVariable(
                decl, constant.Value.extOrTrunc(type_width));
          } else {
            LLDB_LOG(GetLogIfAllCategoriesSet(LIBLLDB_LOG_AST),
                     "Class '{0}' has a member '{1}' of type '{2}' ({3} bits) "
                     "which resolves to a wider constant value ({4} bits). "
                     "Ignoring constant.",
                     m_derived_ct.GetTypeName(), static_data_member.Name,
                     member_ct.GetTypeName(), type_width, constant_width);
          }
        } else {
          lldb::BasicType basic_type_enum = member_ct.GetBasicTypeEnumeration();
          switch (basic_type_enum) {
          case lldb::eBasicTypeFloat:
          case lldb::eBasicTypeDouble:
          case lldb::eBasicTypeLongDouble:
            if (type_width == constant_width) {
              TypeSystemClang::SetFloatingInitializerForVariable(
                  decl, basic_type_enum == lldb::eBasicTypeFloat
                            ? llvm::APFloat(constant.Value.bitsToFloat())
                            : llvm::APFloat(constant.Value.bitsToDouble()));
              decl->setConstexpr(true);
            } else {
              LLDB_LOG(
                  GetLogIfAllCategoriesSet(LIBLLDB_LOG_AST),
                  "Class '{0}' has a member '{1}' of type '{2}' ({3} bits) "
                  "which resolves to a constant value of mismatched width "
                  "({4} bits). Ignoring constant.",
                  m_derived_ct.GetTypeName(), static_data_member.Name,
                  member_ct.GetTypeName(), type_width, constant_width);
            }
            break;
          default:
            break;
          }
        }
        break;
      }
    }
  }

  // FIXME: Add a PdbSymUid namespace for field list members and update
  // the m_uid_to_decl map with this decl.
  return Error::success();
}

Error UdtRecordCompleter::visitKnownMember(CVMemberRecord &cvr,
                                           NestedTypeRecord &nested) {
  return Error::success();
}

Error UdtRecordCompleter::visitKnownMember(CVMemberRecord &cvr,
                                           DataMemberRecord &data_member) {

  uint64_t offset = data_member.FieldOffset * 8;
  uint32_t bitfield_width = 0;

  TypeIndex ti(data_member.Type);
  if (!ti.isSimple()) {
    CVType cvt = m_index.tpi().getType(ti);
    if (cvt.kind() == LF_BITFIELD) {
      BitFieldRecord bfr;
      llvm::cantFail(TypeDeserializer::deserializeAs<BitFieldRecord>(cvt, bfr));
      offset += bfr.BitOffset;
      bitfield_width = bfr.BitSize;
      ti = bfr.Type;
    }
  }

  clang::QualType member_qt = m_ast_builder.GetOrCreateType(PdbTypeSymId(ti));
  m_ast_builder.CompleteType(member_qt);

  lldb::AccessType access = TranslateMemberAccess(data_member.getAccess());

  clang::FieldDecl *decl = TypeSystemClang::AddFieldToRecordType(
      m_derived_ct, data_member.Name, m_ast_builder.ToCompilerType(member_qt),
      access, bitfield_width);
  // FIXME: Add a PdbSymUid namespace for field list members and update
  // the m_uid_to_decl map with this decl.

  m_layout.field_offsets.insert(std::make_pair(decl, offset));

  return Error::success();
}

Error UdtRecordCompleter::visitKnownMember(CVMemberRecord &cvr,
                                           OneMethodRecord &one_method) {
  AddMethod(one_method.Name, one_method.Type, one_method.getAccess(),
            one_method.getOptions(), one_method.Attrs);

  return Error::success();
}

Error UdtRecordCompleter::visitKnownMember(CVMemberRecord &cvr,
                                           OverloadedMethodRecord &overloaded) {
  TypeIndex method_list_idx = overloaded.MethodList;

  CVType method_list_type = m_index.tpi().getType(method_list_idx);
  assert(method_list_type.kind() == LF_METHODLIST);

  MethodOverloadListRecord method_list;
  llvm::cantFail(TypeDeserializer::deserializeAs<MethodOverloadListRecord>(
      method_list_type, method_list));

  for (const OneMethodRecord &method : method_list.Methods)
    AddMethod(overloaded.Name, method.Type, method.getAccess(),
              method.getOptions(), method.Attrs);

  return Error::success();
}

Error UdtRecordCompleter::visitKnownMember(CVMemberRecord &cvr,
                                           EnumeratorRecord &enumerator) {
  Declaration decl;
  llvm::StringRef name = DropNameScope(enumerator.getName());

  m_ast_builder.clang().AddEnumerationValueToEnumerationType(
      m_derived_ct, decl, name.str().c_str(), enumerator.Value);
  return Error::success();
}

void UdtRecordCompleter::complete() {
  // Ensure the correct order for virtual bases.
  std::stable_sort(m_bases.begin(), m_bases.end(),
                   [](const IndexedBase &lhs, const IndexedBase &rhs) {
                     return lhs.first < rhs.first;
                   });

  std::vector<std::unique_ptr<clang::CXXBaseSpecifier>> bases;
  bases.reserve(m_bases.size());
  for (auto &ib : m_bases)
    bases.push_back(std::move(ib.second));

  TypeSystemClang &clang = m_ast_builder.clang();
  clang.TransferBaseClasses(m_derived_ct.GetOpaqueQualType(), std::move(bases));

  clang.AddMethodOverridesForCXXRecordType(m_derived_ct.GetOpaqueQualType());
  TypeSystemClang::BuildIndirectFields(m_derived_ct);
  TypeSystemClang::CompleteTagDeclarationDefinition(m_derived_ct);

  if (auto *record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(&m_tag_decl)) {
    m_ast_builder.importer().SetRecordLayout(record_decl, m_layout);
  }
}
