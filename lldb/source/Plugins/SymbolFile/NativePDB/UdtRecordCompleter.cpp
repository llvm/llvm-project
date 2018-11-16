#include "UdtRecordCompleter.h"

#include "PdbIndex.h"
#include "PdbSymUid.h"
#include "PdbUtil.h"
#include "SymbolFileNativePDB.h"

#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"

#include "llvm/DebugInfo/CodeView/TypeDeserializer.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"

using namespace llvm::codeview;
using namespace llvm::pdb;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::npdb;

using Error = llvm::Error;

UdtRecordCompleter::UdtRecordCompleter(PdbTypeSymId id,
                                       CompilerType &derived_ct,
                                       clang::TagDecl &tag_decl,
                                       SymbolFileNativePDB &symbol_file)
    : m_id(id), m_derived_ct(derived_ct), m_tag_decl(tag_decl),
      m_symbol_file(symbol_file) {
  TpiStream &tpi = symbol_file.m_index->tpi();
  CVType cvt = tpi.getType(m_id.index);
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

lldb::opaque_compiler_type_t UdtRecordCompleter::AddBaseClassForTypeIndex(
    llvm::codeview::TypeIndex ti, llvm::codeview::MemberAccess access) {
  TypeSP base_type = m_symbol_file.GetOrCreateType(ti);
  CompilerType base_ct = base_type->GetFullCompilerType();

  CVType udt_cvt = m_symbol_file.m_index->tpi().getType(ti);

  lldb::opaque_compiler_type_t base_qt = base_ct.GetOpaqueQualType();
  std::unique_ptr<clang::CXXBaseSpecifier> base_spec =
      m_symbol_file.GetASTContext().CreateBaseClassSpecifier(
          base_qt, TranslateMemberAccess(access), false,
          udt_cvt.kind() == LF_CLASS);
  lldbassert(base_spec);
  m_bases.push_back(std::move(base_spec));
  return base_qt;
}

Error UdtRecordCompleter::visitKnownMember(CVMemberRecord &cvr,
                                           BaseClassRecord &base) {
  lldb::opaque_compiler_type_t base_qt =
      AddBaseClassForTypeIndex(base.Type, base.getAccess());

  auto decl = m_symbol_file.GetASTContext().GetAsCXXRecordDecl(base_qt);
  lldbassert(decl);

  auto offset = clang::CharUnits::fromQuantity(base.getBaseOffset());
  m_layout.base_offsets.insert(std::make_pair(decl, offset));

  return llvm::Error::success();
}

Error UdtRecordCompleter::visitKnownMember(CVMemberRecord &cvr,
                                           VirtualBaseClassRecord &base) {
  AddBaseClassForTypeIndex(base.BaseType, base.getAccess());

  // FIXME: Handle virtual base offsets.
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
  TypeSP member_type = m_symbol_file.GetOrCreateType(static_data_member.Type);
  CompilerType complete_member_type = member_type->GetFullCompilerType();

  lldb::AccessType access =
      TranslateMemberAccess(static_data_member.getAccess());
  ClangASTContext::AddVariableToRecordType(
      m_derived_ct, static_data_member.Name, complete_member_type, access);

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

  TypeSP member_type;
  TpiStream &tpi = m_symbol_file.m_index->tpi();
  TypeIndex ti(data_member.Type);
  if (!ti.isSimple()) {
    CVType cvt = tpi.getType(ti);
    if (cvt.kind() == LF_BITFIELD) {
      BitFieldRecord bfr;
      llvm::cantFail(TypeDeserializer::deserializeAs<BitFieldRecord>(cvt, bfr));
      offset += bfr.BitOffset;
      bitfield_width = bfr.BitSize;
      ti = bfr.Type;
    }
  }

  member_type = m_symbol_file.GetOrCreateType(ti);
  CompilerType complete_member_type = member_type->GetFullCompilerType();
  lldb::AccessType access = TranslateMemberAccess(data_member.getAccess());

  clang::FieldDecl *decl = ClangASTContext::AddFieldToRecordType(
      m_derived_ct, data_member.Name, complete_member_type, access,
      bitfield_width);
  // FIXME: Add a PdbSymUid namespace for field list members and update
  // the m_uid_to_decl map with this decl.

  m_layout.field_offsets.insert(std::make_pair(decl, offset));

  return Error::success();
}

Error UdtRecordCompleter::visitKnownMember(CVMemberRecord &cvr,
                                           OneMethodRecord &one_method) {
  return Error::success();
}

Error UdtRecordCompleter::visitKnownMember(CVMemberRecord &cvr,
                                           OverloadedMethodRecord &overloaded) {
  return Error::success();
}

Error UdtRecordCompleter::visitKnownMember(CVMemberRecord &cvr,
                                           EnumeratorRecord &enumerator) {
  ClangASTContext &clang = m_symbol_file.GetASTContext();

  Declaration decl;
  llvm::StringRef name = DropNameScope(enumerator.getName());
  TypeSP underlying_type =
      m_symbol_file.GetOrCreateType(m_cvr.er.getUnderlyingType());

  uint64_t byte_size = underlying_type->GetByteSize();
  clang.AddEnumerationValueToEnumerationType(
      m_derived_ct, decl, name.str().c_str(), enumerator.Value.getSExtValue(),
      byte_size * 8);
  return Error::success();
}

void UdtRecordCompleter::complete() {
  ClangASTContext &clang = m_symbol_file.GetASTContext();
  clang.TransferBaseClasses(m_derived_ct.GetOpaqueQualType(),
                            std::move(m_bases));

  clang.AddMethodOverridesForCXXRecordType(m_derived_ct.GetOpaqueQualType());
  ClangASTContext::BuildIndirectFields(m_derived_ct);
  ClangASTContext::CompleteTagDeclarationDefinition(m_derived_ct);

  if (auto *record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(&m_tag_decl)) {
    m_symbol_file.GetASTImporter().InsertRecordDecl(record_decl, m_layout);
  }
}
