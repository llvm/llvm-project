//===-- GNUstepNSArray.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GNUstepFormatters.h"

#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/Target/Language.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Stream.h"
#include "lldb/ValueObject/ValueObject.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

namespace {

/// GSArray, GSInlineArray and GSMutableArray all start with
/// `id *_contents_array; unsigned _count;` (Source/GSPrivate.h). Where the
/// element buffer lives (a separate allocation, or right after the instance
/// for GSInlineArray) does not matter: _contents_array is always the absolute
/// address of element 0.
struct ArrayContents {
  addr_t elements = LLDB_INVALID_ADDRESS;
  uint64_t count = 0;
};

std::optional<ArrayContents> ReadArray(ValueObject &valobj) {
  ValueObjectSP contents_sp = GNUstepGetIvar(valobj, "_contents_array");
  ValueObjectSP count_sp = GNUstepGetIvar(valobj, "_count");
  if (!contents_sp || !count_sp)
    return std::nullopt;
  ArrayContents contents;
  contents.elements = contents_sp->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
  contents.count = count_sp->GetValueAsUnsigned(0);
  if (contents.count && contents.elements == LLDB_INVALID_ADDRESS)
    return std::nullopt;
  return contents;
}

class GNUstepNSArraySyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  GNUstepNSArraySyntheticFrontEnd(ValueObjectSP valobj_sp)
      : SyntheticChildrenFrontEnd(*valobj_sp) {
    if (valobj_sp) {
      m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();
      if (ProcessSP process_sp = valobj_sp->GetProcessSP())
        m_ptr_size = process_sp->GetAddressByteSize();
      // Children are created as `id` so that each element resolves its own
      // dynamic type and formatter, exactly like the Apple frontends do.
      if (TargetSP target_sp = valobj_sp->GetTargetSP())
        if (TypeSystemClangSP scratch_ts_sp =
                ScratchTypeSystemClang::GetForTarget(*target_sp))
          m_id_type = scratch_ts_sp->GetBasicType(eBasicTypeObjCID);
    }
  }

  llvm::Expected<uint32_t> CalculateNumChildren() override {
    return m_contents.count;
  }

  ValueObjectSP GetChildAtIndex(uint32_t idx) override {
    if (idx >= m_contents.count || !m_id_type.IsValid())
      return {};
    StreamString name;
    name.Printf("[%u]", idx);
    return CreateChildValueObjectFromAddress(
        name.GetString(), m_contents.elements + idx * m_ptr_size, m_exe_ctx_ref,
        m_id_type);
  }

  lldb::ChildCacheState Update() override {
    m_contents = ArrayContents();
    if (std::optional<ArrayContents> contents = ReadArray(m_backend))
      m_contents = *contents;
    return lldb::ChildCacheState::eRefetch;
  }

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override {
    if (std::optional<size_t> idx = ExtractIndexFromString(name.GetCString()))
      if (*idx < m_contents.count)
        return *idx;
    return llvm::createStringError("Type has no child named '%s'",
                                   name.AsCString(""));
  }

private:
  ExecutionContextRef m_exe_ctx_ref;
  uint8_t m_ptr_size = 8;
  CompilerType m_id_type;
  ArrayContents m_contents;
};

} // namespace

bool lldb_private::formatters::GNUstepNSArraySummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  if (!IsGNUstepObjCRuntime(valobj))
    return false;
  std::optional<ArrayContents> contents = ReadArray(valobj);
  if (!contents)
    return false;

  static constexpr llvm::StringLiteral g_TypeHint("NSArray");
  llvm::StringRef prefix, suffix;
  if (Language *language = Language::FindPlugin(options.GetLanguage()))
    std::tie(prefix, suffix) = language->GetFormatterPrefixSuffix(g_TypeHint);
  stream << prefix;
  stream.Printf("%" PRIu64 " %s%s", contents->count, "element",
                contents->count == 1 ? "" : "s");
  stream << suffix;
  return true;
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::GNUstepNSArraySyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp || !IsGNUstepObjCRuntime(*valobj_sp))
    return nullptr;
  return new GNUstepNSArraySyntheticFrontEnd(valobj_sp);
}
