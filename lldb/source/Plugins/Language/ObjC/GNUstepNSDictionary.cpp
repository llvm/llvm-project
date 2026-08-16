//===-- GNUstepNSDictionary.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// gnustep-base's GSDictionary, GSSet and GSCountedSet all embed a
// `GSIMapTable_t map` (Headers/GNUstepBase/GSIMap.h): `bucketCount` buckets,
// each `{ nodeCount, firstNode }`, chaining nodes `{ nextInBucket, key
// [, value] }` through nextInBucket. Sets instantiate the map without the
// value member, so nodes are read through their debug-info types rather
// than at fixed offsets.
//
//===----------------------------------------------------------------------===//

#include "GNUstepFormatters.h"
#include "NSDictionary.h"

#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/Target/Language.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/Stream.h"
#include "lldb/ValueObject/ValueObject.h"

#include <set>
#include <vector>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

namespace {

struct MapEntry {
  addr_t key = 0;
  addr_t value = 0;
};

/// The whole map's worth of entries, in bucket order (which is hash order,
/// like -objectEnumerator).
struct MapContents {
  uint64_t node_count = 0;
  std::vector<MapEntry> entries;
};

uint64_t ReadUnsignedMember(ValueObject &value, llvm::StringRef name,
                            uint64_t fail = 0) {
  if (ValueObjectSP member_sp = value.GetChildMemberWithName(name))
    return member_sp->GetValueAsUnsigned(fail);
  return fail;
}

/// A GSIMapKey/GSIMapVal is a union whose members are all pointer sized;
/// its `obj` (or `nsu`) member is the entry.
uint64_t ReadUnionWord(ValueObject &value, llvm::StringRef name) {
  ValueObjectSP member_sp = value.GetChildMemberWithName(name);
  if (!member_sp)
    return 0;
  if (ValueObjectSP first_sp = member_sp->GetChildAtIndex(0))
    return first_sp->GetValueAsUnsigned(0);
  return member_sp->GetValueAsUnsigned(0);
}

/// Reads just the element count.
std::optional<uint64_t> ReadNodeCount(ValueObject &valobj) {
  ValueObjectSP map_sp = GNUstepGetIvar(valobj, "map");
  if (!map_sp)
    return std::nullopt;
  ValueObjectSP count_sp = map_sp->GetChildMemberWithName("nodeCount");
  if (!count_sp)
    return std::nullopt;
  return count_sp->GetValueAsUnsigned(0);
}

/// Walks every bucket. \p want_value is false for sets, whose nodes have no
/// value member (GSSet.m sets GSI_MAP_HAS_VALUE to 0). Bounded by
/// nodeCount, by the bucket count, and by never revisiting a node.
std::optional<MapContents> ReadMap(ValueObject &valobj, bool want_value) {
  ValueObjectSP map_sp = GNUstepGetIvar(valobj, "map");
  if (!map_sp)
    return std::nullopt;
  MapContents contents;
  contents.node_count = ReadUnsignedMember(*map_sp, "nodeCount");
  const uint64_t bucket_count = ReadUnsignedMember(*map_sp, "bucketCount");
  ValueObjectSP buckets_sp = map_sp->GetChildMemberWithName("buckets");
  if (!buckets_sp)
    return std::nullopt;
  if (contents.node_count == 0)
    return contents;
  // A table cannot sensibly have more buckets than a few times its nodes;
  // anything else is a misread and would make the walk unbounded.
  if (bucket_count == 0 || bucket_count > contents.node_count * 8 + 64)
    return std::nullopt;

  std::set<addr_t> seen;
  for (uint64_t b = 0;
       b < bucket_count && contents.entries.size() < contents.node_count; ++b) {
    ValueObjectSP bucket_sp = buckets_sp->GetSyntheticArrayMember(b, true);
    if (!bucket_sp)
      break;
    ValueObjectSP node_sp = bucket_sp->GetChildMemberWithName("firstNode");
    while (node_sp && node_sp->GetValueAsUnsigned(0) != 0 &&
           contents.entries.size() < contents.node_count) {
      const addr_t node_addr = node_sp->GetValueAsUnsigned(0);
      if (!seen.insert(node_addr).second)
        return std::nullopt; // cycle: corrupt or racing table
      Status error;
      ValueObjectSP node_struct_sp = node_sp->Dereference(error);
      if (!node_struct_sp || error.Fail())
        break;
      MapEntry entry;
      entry.key = ReadUnionWord(*node_struct_sp, "key");
      if (want_value)
        entry.value = ReadUnionWord(*node_struct_sp, "value");
      contents.entries.push_back(entry);
      node_sp = node_struct_sp->GetChildMemberWithName("nextInBucket");
    }
  }
  return contents;
}

/// Presents each entry as `[i] = { key, value }` (dictionaries) or as the
/// key object itself (sets).
class GNUstepMapSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  GNUstepMapSyntheticFrontEnd(ValueObjectSP valobj_sp, bool is_dictionary)
      : SyntheticChildrenFrontEnd(*valobj_sp), m_is_dictionary(is_dictionary) {
    if (valobj_sp) {
      m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();
      if (ProcessSP process_sp = valobj_sp->GetProcessSP()) {
        m_ptr_size = process_sp->GetAddressByteSize();
        m_order = process_sp->GetByteOrder();
      }
      if (TargetSP target_sp = valobj_sp->GetTargetSP()) {
        if (TypeSystemClangSP scratch_ts_sp =
                ScratchTypeSystemClang::GetForTarget(*target_sp))
          m_id_type = scratch_ts_sp->GetBasicType(eBasicTypeObjCID);
        if (m_is_dictionary)
          m_pair_type = GetLLDBNSPairType(target_sp);
      }
    }
  }

  llvm::Expected<uint32_t> CalculateNumChildren() override {
    return m_contents.entries.size();
  }

  ValueObjectSP GetChildAtIndex(uint32_t idx) override {
    if (idx >= m_contents.entries.size())
      return {};
    if (m_children[idx])
      return m_children[idx];
    const MapEntry &entry = m_contents.entries[idx];
    StreamString name;
    name.Printf("[%u]", idx);
    if (!m_is_dictionary) {
      // A set element is the key object itself.
      m_children[idx] =
          CreateChildFromWords(name.GetString(), {entry.key}, m_id_type);
      return m_children[idx];
    }
    if (!m_pair_type.IsValid())
      return {};
    m_children[idx] = CreateChildFromWords(
        name.GetString(), {entry.key, entry.value}, m_pair_type);
    return m_children[idx];
  }

  lldb::ChildCacheState Update() override {
    m_contents = MapContents();
    m_children.clear();
    if (std::optional<MapContents> contents =
            ReadMap(m_backend, m_is_dictionary))
      m_contents = *contents;
    m_children.resize(m_contents.entries.size());
    return lldb::ChildCacheState::eRefetch;
  }

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override {
    if (std::optional<size_t> idx = ExtractIndexFromString(name.GetCString()))
      if (*idx < m_contents.entries.size())
        return *idx;
    return llvm::createStringError("Type has no child named '%s'",
                                   name.AsCString(""));
  }

private:
  ValueObjectSP CreateChildFromWords(llvm::StringRef name,
                                     std::initializer_list<addr_t> words,
                                     CompilerType type) {
    WritableDataBufferSP buffer_sp(
        new DataBufferHeap(words.size() * m_ptr_size, 0));
    uint8_t *bytes = buffer_sp->GetBytes();
    for (addr_t word : words) {
      if (m_ptr_size == 8)
        memcpy(bytes, &word, 8);
      else {
        uint32_t narrow = static_cast<uint32_t>(word);
        memcpy(bytes, &narrow, 4);
      }
      bytes += m_ptr_size;
    }
    DataExtractor data(buffer_sp, m_order, m_ptr_size);
    return CreateChildValueObjectFromData(name, data, m_exe_ctx_ref, type);
  }

  bool m_is_dictionary;
  ExecutionContextRef m_exe_ctx_ref;
  uint8_t m_ptr_size = 8;
  lldb::ByteOrder m_order = lldb::eByteOrderLittle;
  CompilerType m_id_type;
  CompilerType m_pair_type;
  MapContents m_contents;
  std::vector<ValueObjectSP> m_children;
};

} // namespace

bool lldb_private::formatters::GNUstepNSDictionarySummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  if (!IsGNUstepObjCRuntime(valobj))
    return false;
  std::optional<uint64_t> count = ReadNodeCount(valobj);
  if (!count)
    return false;
  stream.Printf("%" PRIu64 " key/value pair%s", *count, *count == 1 ? "" : "s");
  return true;
}

bool lldb_private::formatters::GNUstepNSSetSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  if (!IsGNUstepObjCRuntime(valobj))
    return false;
  std::optional<uint64_t> count = ReadNodeCount(valobj);
  if (!count)
    return false;
  stream.Printf("%" PRIu64 " element%s", *count, *count == 1 ? "" : "s");
  return true;
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::GNUstepNSDictionarySyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp || !IsGNUstepObjCRuntime(*valobj_sp))
    return nullptr;
  return new GNUstepMapSyntheticFrontEnd(valobj_sp, /*is_dictionary=*/true);
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::GNUstepNSSetSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp || !IsGNUstepObjCRuntime(*valobj_sp))
    return nullptr;
  return new GNUstepMapSyntheticFrontEnd(valobj_sp, /*is_dictionary=*/false);
}
