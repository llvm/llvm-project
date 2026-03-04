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

#include "llvm/DebugInfo/CodeView/TypeDeserializer.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"
#include "llvm/Support/ErrorHandling.h"

#include "swift/Demangling/Demangle.h"

using namespace lldb_private;
using namespace lldb_private::npdb;
using namespace llvm::codeview;
using namespace llvm::pdb;

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
    if (!cr.hasUniqueName())
      return {};
    decorated = cr.UniqueName;
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

void PdbAstBuilderSwift::Dump(Stream &stream, llvm::StringRef filter) {
  m_swift_ts.Dump(stream.AsRawOstream(), filter);
}
