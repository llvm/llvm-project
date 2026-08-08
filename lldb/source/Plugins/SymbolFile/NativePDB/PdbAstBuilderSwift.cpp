//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PdbAstBuilderSwift.h"

#include "PdbUtil.h"
#include "SymbolFileNativePDB.h"

#include "Plugins/TypeSystem/Swift/TypeSystemSwiftTypeRef.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/TypeDeserializer.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"
#include "llvm/Support/ErrorHandling.h"

#include "swift/Demangling/Demangle.h"

using namespace lldb_private;
using namespace lldb_private::npdb;
using namespace llvm::codeview;
using namespace llvm::pdb;

namespace {
/// Fills `mangled_name` if `cr` is shaped like a bound generic.
struct BoundGenericVisitor : public TypeVisitorCallbacks {
  TpiStream &tpi;
  llvm::StringRef outer_prefix;
  llvm::StringRef mangled_name;
  bool seen = false;

  BoundGenericVisitor(TpiStream &tpi, llvm::StringRef outer_prefix)
      : tpi(tpi), outer_prefix(outer_prefix) {}

  llvm::Error visitKnownMember(CVMemberRecord &,
                               DataMemberRecord &member) override {
    if (seen) {
      // More than one member = doesn't match the pattern.
      mangled_name = {};
      return llvm::Error::success();
    }
    seen = true;
    CVType inner_cvt = tpi.getType(member.Type);
    if (!llvm::is_contained({LF_STRUCTURE, LF_CLASS}, inner_cvt.kind()))
      return llvm::Error::success();
    ClassRecord inner;
    if (llvm::Error err =
            TypeDeserializer::deserializeAs<ClassRecord>(inner_cvt, inner))
      return err;
    llvm::StringRef inner_mangled = inner.Name;
    if (!inner_mangled.consume_front(outer_prefix) ||
        inner_mangled != member.Name ||
        !swift::Demangle::isSwiftSymbol(member.Name))
      return llvm::Error::success();
    mangled_name = member.Name;
    return llvm::Error::success();
  }
};
} // namespace

llvm::Expected<llvm::StringRef>
PdbAstBuilderSwift::MaybeUnwrapBoundGeneric(const ClassRecord &cr,
                                            TpiStream &tpi) {
  if (!cr.Name.ends_with("::<unnamed-tag>") || cr.FieldList.isNoneType())
    return llvm::StringRef();

  CVType field_list_cvt = tpi.getType(cr.FieldList);
  if (field_list_cvt.kind() != LF_FIELDLIST)
    return llvm::StringRef();

  FieldListRecord field_list;
  if (llvm::Error err = TypeDeserializer::deserializeAs<FieldListRecord>(
          field_list_cvt, field_list))
    return std::move(err);

  llvm::StringRef outer_prefix =
      cr.Name.drop_back(llvm::StringRef("<unnamed-tag>").size());
  BoundGenericVisitor visitor(tpi, outer_prefix);
  if (llvm::Error err = visitMemberRecordStream(field_list.Data, visitor))
    return std::move(err);

  return visitor.mangled_name;
}

PdbAstBuilderSwift::PdbAstBuilderSwift(TypeSystemSwiftTypeRef &swift_ts)
    : m_swift_ts(swift_ts) {}

CompilerType PdbAstBuilderSwift::CreateType(PdbTypeSymId type,
                                            TpiStream &tpi) {
  if (type.index.isSimple())
    return {};

  CVType cvt = tpi.getType(type.index);

  llvm::StringRef decorated;
  switch (cvt.kind()) {
  case LF_STRUCTURE:
  case LF_CLASS: {
    ClassRecord cr;
    if (auto err = TypeDeserializer::deserializeAs<ClassRecord>(cvt, cr)) {
      LLDB_LOG_ERROR(GetLog(LLDBLog::Symbols), std::move(err),
                     "Failed to deserialize ClassRecord: {0}");
      return {};
    }
    if (cr.hasUniqueName()) {
      decorated = cr.UniqueName;
    } else {
      auto unwrapped = MaybeUnwrapBoundGeneric(cr, tpi);
      if (!unwrapped) {
        LLDB_LOG_ERROR(GetLog(LLDBLog::Symbols), unwrapped.takeError(),
                       "Deserialization failure while checking for Swift bound generic: {0}");
        return {};
      }
      if (unwrapped->empty())
        return {};
      decorated = *unwrapped;
    }
    break;
  }
  case LF_ENUM: {
    EnumRecord er;
    if (auto err = TypeDeserializer::deserializeAs<EnumRecord>(cvt, er)) {
      LLDB_LOG_ERROR(GetLog(LLDBLog::Symbols), std::move(err),
                     "Failed to deserialize EnumRecord: {0}");
      return {};
    }
    if (!er.hasUniqueName())
      return {};
    decorated = er.UniqueName;
    break;
  }
  case LF_MODIFIER: {
    ModifierRecord mfr;
    if (auto err = TypeDeserializer::deserializeAs<ModifierRecord>(cvt, mfr)) {
      LLDB_LOG_ERROR(GetLog(LLDBLog::Symbols), std::move(err),
                     "Failed to deserialize ModifierRecord: {0}");
      return {};
    }
    return GetOrCreateType(PdbTypeSymId(mfr.ModifiedType, false));
  }
  default:
    return {};
  }

  if (!swift::Demangle::isSwiftSymbol(decorated))
    return {};

  return m_swift_ts.GetTypeFromMangledTypename(ConstString(decorated));
}

CompilerType PdbAstBuilderSwift::GetOrCreateType(PdbTypeSymId type) {
  if (type.index.isNoneType())
    return {};

  lldb::user_id_t uid = toOpaqueUid(type);
  if (auto iter = m_uid_to_type.find(uid); iter != m_uid_to_type.end())
    return iter->second;

  auto *pdb = llvm::dyn_cast<SymbolFileNativePDB>(
      m_swift_ts.GetSymbolFile()->GetBackingSymbolFile());
  if (!pdb) {
    lldbassert(false && "PdbAstBuilderSwift called from outside NativePDB context.");
    return {};
  }
  PdbIndex &index = pdb->GetIndex();
  PdbTypeSymId best_type = GetBestPossibleDecl(type, index.tpi());

  CompilerType ct = best_type.index == type.index
                        ? CreateType(type, index.tpi())
                        : GetOrCreateType(best_type);
  if (ct)
    m_uid_to_type[uid] = ct;
  return ct;
}

void PdbAstBuilderSwift::Dump(Stream &stream, llvm::StringRef filter,
                              bool show_color) {
  m_swift_ts.Dump(stream.AsRawOstream(), filter, show_color);
}
