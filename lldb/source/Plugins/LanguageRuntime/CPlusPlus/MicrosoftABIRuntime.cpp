//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MicrosoftABIRuntime.h"

#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/Utility/LLDBLog.h"
#include "clang/AST/RecordLayout.h"

#include <deque>
#include <mutex>

using namespace lldb;
using namespace lldb_private;

static const llvm::StringRef s_vtable_demangled_indicator = "::`vftable'";
static const llvm::StringRef s_rtti_complete_object_locator_indicator =
    "::`RTTI Complete Object Locator'";

static llvm::StringRef StripDemangledVTableSuffix(llvm::StringRef demangled) {
  llvm::StringRef name = demangled.split(s_vtable_demangled_indicator).first;
  name.consume_front("const ");
  return name;
}

static bool
SplitVTableComponents(llvm::StringRef demangled,
                      llvm::SmallVectorImpl<llvm::StringRef> &components) {
  llvm::StringRef suffix = demangled.split(s_vtable_demangled_indicator).second;
  if (suffix.empty())
    return true; // e.g. MyClass::`vftable'

  // Otherwise, suffix looks like "{for `MyBase2's `MyBase1'}" or
  // "{for `MyBase'}".
  if (!suffix.consume_front("{for `") || !suffix.consume_back("'}"))
    return false;

  suffix.split(components, "'s `");
  return true;
}

MicrosoftABIRuntime::MicrosoftABIRuntime(Process *process)
    : CommonABIRuntime(process) {}

bool MicrosoftABIRuntime::GetDynamicTypeAndAddress(
    ValueObject &in_value, lldb::DynamicValueType use_dynamic,
    const LanguageRuntime::VTableInfo &vtable_info,
    TypeAndOrName &class_type_or_name, Address &dynamic_address) {
  std::optional<size_t> offset_to_top =
      GetOffsetToTop(in_value, class_type_or_name, vtable_info);
  if (!offset_to_top)
    return false;

  lldb::addr_t dynamic_addr =
      in_value.GetPointerValue().address - *offset_to_top;
  if (!m_process->GetTarget().ResolveLoadAddress(dynamic_addr, dynamic_address))
    dynamic_address.SetRawAddress(dynamic_addr);

  return true;
}

bool MicrosoftABIRuntime::IsVTableSymbol(Mangled &mangled) const {
  return mangled.GetDemangledName().GetStringRef().contains(
      s_vtable_demangled_indicator);
}

TypeAndOrName MicrosoftABIRuntime::GetTypeInfo(
    ValueObject &in_value, const LanguageRuntime::VTableInfo &vtable_info,
    llvm::SmallVectorImpl<llvm::StringRef> &vtable_components) {
  if (!vtable_info.addr.IsSectionOffset() || !vtable_info.symbol)
    return {};

  // See if we have cached info for this type already
  TypeAndOrName type_info = GetDynamicTypeInfo(vtable_info.addr);
  if (type_info)
    return type_info;

  Log *log = GetLog(LLDBLog::Object);
  llvm::StringRef symbol_name =
      vtable_info.symbol->GetMangled().GetDemangledName().GetStringRef();
  if (!SplitVTableComponents(symbol_name, vtable_components)) {
    LLDB_LOG(log, "{0:x16}: '{1}' is not a valid vtable symbol",
             in_value.GetPointerValue().address, symbol_name);
    return {};
  }

  llvm::StringRef class_name = StripDemangledVTableSuffix(symbol_name);

  type_info.SetName(class_name);
  bool any_found = false;
  TypeSP type_sp = LookupTypeByName(
      class_name, vtable_info.symbol->CalculateSymbolContextModule(),
      any_found);
  if (!any_found)
    return TypeAndOrName(); // Type is not dynamic.

  if (type_sp) {
    LLDB_LOG(log,
             "static-type = '{0}' has dynamic type: uid={1:x}, type-name='{2}'",
             in_value.GetTypeName(), type_sp->GetID(), type_sp->GetName());
    type_info.SetTypeSP(std::move(type_sp));
  }

  if (type_info)
    SetDynamicTypeInfo(vtable_info.addr, type_info);
  return type_info;
}

std::optional<int64_t> MicrosoftABIRuntime::GetOffsetToTop(
    ValueObject &in_value, TypeAndOrName &class_type_or_name,
    const LanguageRuntime::VTableInfo &vtable_info) {
  Log *log = GetLog(LLDBLog::Object);

  llvm::SmallVector<llvm::StringRef, 2> vtable_components;
  class_type_or_name = GetTypeInfo(in_value, vtable_info, vtable_components);
  if (!class_type_or_name) {
    LLDB_LOG(log, "{0:x16}: failed to get type info",
             in_value.GetPointerValue().address);
    return std::nullopt;
  }

  {
    std::scoped_lock<std::mutex> g(m_mutex);
    auto offset_to_top_it = m_offset_to_top_map.find(vtable_info.addr);
    if (offset_to_top_it != m_offset_to_top_map.end())
      return offset_to_top_it->second;
  }

  CompilerType type = class_type_or_name.GetCompilerType();
  if (!type) {
    LLDB_LOG(log, "{0:x16}: failed to get type info (no compiler type)",
             in_value.GetPointerValue().address);
    return std::nullopt;
  }

  CompilerType in_type = in_value.GetCompilerType();
  if (in_type.IsPointerOrReferenceType())
    in_type = in_type.GetPointeeType();
  if (!in_type.GetCompleteType()) {
    LLDB_LOG(log, "{0:x16}: {1} is not a complete type",
             in_value.GetPointerValue().address, in_type.GetTypeName());
    return std::nullopt;
  }
  if (!type.GetCompleteType()) {
    LLDB_LOG(log, "{0:x16}: {1} is not a complete type",
             in_value.GetPointerValue().address,
             type.GetTypeName().GetCString());
    return std::nullopt;
  }

  if (TypeSystemClang::AreTypesSame(in_type, type)) {
    LLDB_LOG(log, "{0:x16}: in type is the same as the dynamic type",
             in_value.GetPointerValue().address);

    // The dynamic type we found was the same type, so we don't have a
    // dynamic type here, but if a derived type encounters this vtable symbol,
    // remember that it has offset 0.
    std::scoped_lock<std::mutex> g(m_mutex);
    m_offset_to_top_map[vtable_info.addr] = 0;
    return std::nullopt;
  }

  // Unlike in the Itanium ABI, the VTable itself doesn't have a field telling
  // us the offset to the dynamic type.
  // When compiled with RTTI, we can use the complete object locator to get this
  // offset. This is preferred, because it can resolve types across different
  // modules and handle virtual bases with PDB.
  // If this fails (e.g. no RTTI available), the offset is determined from the
  // AST. This requires the types to be from the same type system. Because PDB
  // doesn't include the offsets of virtual bases, we don't attempt to resolve
  // these. They're technically present in the AST, but they often have
  // incorrect offsets.
  std::optional<int64_t> offset = GetOffsetToTopFromRTTI(vtable_info, type);
  llvm::StringRef strategy = "RTTI";
  if (!offset) {
    offset = GetOffsetToTopFromAST(in_type, in_value, type, vtable_info,
                                   vtable_components);
    strategy = "AST";
  }

  if (offset) {
    LLDB_LOG(GetLog(LLDBLog::Object),
             "{0:x16}: [{1}] '{2}' has offset {3} in '{4}'",
             in_value.GetPointerValue().address, strategy,
             in_type.GetTypeName(), *offset, type.GetTypeName());

    std::scoped_lock<std::mutex> g(m_mutex);
    m_offset_to_top_map[vtable_info.addr] = offset;
  }

  return offset;
}

std::optional<int64_t> MicrosoftABIRuntime::GetOffsetToTopFromRTTI(
    const LanguageRuntime::VTableInfo &vtable_info,
    const CompilerType &dynamic_type) {
  // The pointer to the RTTI complete object locator sits one pointer size above
  // the vftable.
  Target &target = m_process->GetTarget();
  const addr_t vtable_load_addr = vtable_info.addr.GetLoadAddress(&target);
  if (vtable_load_addr == LLDB_INVALID_ADDRESS)
    return std::nullopt;
  const uint32_t addr_byte_size = m_process->GetAddressByteSize();
  const lldb::addr_t complete_object_locator_load_addr =
      vtable_load_addr - addr_byte_size;
  // Check for underflow.
  if (complete_object_locator_load_addr >= vtable_load_addr)
    return std::nullopt;

  Log *log = GetLog(LLDBLog::Object);
  // Pointer to the RTTI complete object locator.
  llvm::Expected<lldb::addr_t> rtti_locator_base_load_addr =
      m_process->ReadPointerFromMemory(complete_object_locator_load_addr);
  if (!rtti_locator_base_load_addr ||
      *rtti_locator_base_load_addr == LLDB_INVALID_ADDRESS) {
    LLDB_LOG_ERROR(log, rtti_locator_base_load_addr.takeError(),
                   "Failed to read address of RTTI complete object locator at "
                   "{1:x16}: {0}",
                   complete_object_locator_load_addr);
    return std::nullopt;
  }

  Status error;
  Address rtti_locator_base_addr;
  if (!m_process->GetTarget().ResolveLoadAddress(*rtti_locator_base_load_addr,
                                                 rtti_locator_base_addr)) {
    LLDB_LOG_ERROR(log, error.takeError(),
                   "Failed to resolve load address of RTTI complete object "
                   "locator at {1:x16}: {0}",
                   *rtti_locator_base_load_addr);
    return std::nullopt;
  }

  // Check if we're actually looking at the RTTI complete object locator.
  Symbol *symbol = rtti_locator_base_addr.CalculateSymbolContextSymbol();
  if (symbol == nullptr)
    return std::nullopt;
  if (!symbol->GetName().GetStringRef().contains(
          s_rtti_complete_object_locator_indicator))
    return std::nullopt;

  LLDB_LOG(log, "Found RTTI complete object locator: {0:x16}",
           *rtti_locator_base_load_addr);

  auto ts = dynamic_type.GetTypeSystem();
  if (!ts)
    return std::nullopt;

  llvm::Expected<uint64_t> int_size =
      ts->GetBasicTypeFromAST(lldb::eBasicTypeInt).GetByteSize(nullptr);
  if (!int_size) {
    LLDB_LOG_ERROR(log, int_size.takeError(),
                   "Failed to get size of int from AST: {0}");
    return std::nullopt;
  }

  // See MSRTTIBuilder::getCompleteObjectLocator in Clang for the layout.
  // We're interested in the offset which is the second int in the struct.
  Address offset_from_top_addr = rtti_locator_base_addr;
  offset_from_top_addr.Slide(*int_size);

  error.Clear();
  const int64_t offset_from_top = target.ReadSignedIntegerFromMemory(
      offset_from_top_addr, *int_size, INT64_MIN, error);
  if (offset_from_top == INT64_MIN || error.Fail()) {
    LLDB_LOG_ERROR(log, error.takeError(),
                   "Failed to read 'offset from top' from RTTI complete object "
                   "locator at {1:x16}: {0}",
                   offset_from_top_addr.GetOffset());
    return std::nullopt;
  }

  return offset_from_top;
}

/// Try to find the offset of `target_base` inside `haystack` while visiting all
/// `checkpoints`.
static std::optional<size_t>
GetOffsetOfBaseFromAST(clang::ASTContext &ast,
                       const clang::CXXRecordDecl *haystack,
                       const clang::CXXRecordDecl *target_base,
                       const llvm::ArrayRef<llvm::StringRef> checkpoints) {
  struct BaseEntry {
    const clang::CXXRecordDecl *decl;
    /// If this is `std::nullopt`, then this entry is on a path where we can't
    /// determine the offset.
    std::optional<size_t> offset;
    /// Remaining checkpoints on this path.
    llvm::ArrayRef<llvm::StringRef> checkpoints;
    /// Was `target_base` found on this path?
    bool found_target = false;
  };

  // This does a breadth-first search over all bases. Starting point is the root
  // (`haystack`).
  std::deque<BaseEntry> bases;
  bases.emplace_back(BaseEntry{
      /*decl=*/haystack,
      /*offset=*/0,
      /*checkpoints=*/checkpoints,
  });

  // Go through virtual bases first.
  std::string base_name;
  for (const clang::CXXBaseSpecifier &base : haystack->vbases()) {
    const clang::CXXRecordDecl *base_decl = llvm::cast<clang::CXXRecordDecl>(
        base.getType()->castAs<clang::RecordType>()->getDecl());
    // Although clang gives us some offset here, this offset is not from the
    // debug info and thus likely wrong. Neither PDB nor DWARF include it.
    if (!checkpoints.empty()) {
      base_name = base_decl->getDeclName().getAsString();
      if (base_name == checkpoints.back())
        return std::nullopt;
    } else if (base_decl == target_base) {
      return std::nullopt;
    }

    // We still need to keep track of this class as `target_base` might be
    // inside this base.
    bases.emplace_back(BaseEntry{
        /*decl=*/base_decl,
        /*offset=*/std::nullopt,
        /*checkpoints=*/checkpoints,
    });
  }

  while (!bases.empty()) {
    BaseEntry current = bases.front();
    assert(
        (!current.found_target || !current.checkpoints.empty()) &&
        "if we found the target we should only need to check the checkpoints");
    bases.pop_front();
    const clang::ASTRecordLayout &record_layout =
        ast.getASTRecordLayout(current.decl);

    for (const clang::CXXBaseSpecifier &base : current.decl->bases()) {
      if (base.isVirtual())
        continue; // We already looked through virtual bases.

      const clang::CXXRecordDecl *base_decl = llvm::cast<clang::CXXRecordDecl>(
          base.getType()->castAs<clang::RecordType>()->getDecl());
      bool found_target = base_decl == target_base || current.found_target;
      std::optional<size_t> offset = current.offset;
      if (offset)
        *offset += record_layout.getBaseClassOffset(base_decl).getQuantity();

      if (!current.checkpoints.empty()) {
        base_name = base_decl->getDeclName().getAsString();
        if (base_name == current.checkpoints.back()) {
          // Last component was the base we were looking for.
          if (found_target && current.checkpoints.size() == 1)
            return offset;

          // Found a checkpoint, so we have to take this path.
          bases.clear();
          bases.emplace_back(BaseEntry{
              /*decl=*/base_decl,
              /*offset=*/offset,
              /*checkpoints=*/current.checkpoints.drop_back(),
              /*found_target=*/found_target,
          });
        }
      } else if (found_target) {
        return offset;
      }

      bases.emplace_back(BaseEntry{
          /*decl=*/base_decl,
          /*offset=*/offset,
          /*checkpoints=*/current.checkpoints,
          /*found_target=*/found_target,
      });
    }
  }

  return std::nullopt;
}

std::optional<int64_t> MicrosoftABIRuntime::GetOffsetToTopFromAST(
    const CompilerType &in_type, ValueObject &in_value,
    const CompilerType &dyn_type,
    const LanguageRuntime::VTableInfo &vtable_info,
    llvm::ArrayRef<llvm::StringRef> vtable_components) {
  Log *log = GetLog(LLDBLog::Object);

  CompilerType in_base_type = in_type;

  clang::CXXRecordDecl *in_decl =
      TypeSystemClang::GetAsCXXRecordDecl(in_base_type.GetOpaqueQualType());
  clang::CXXRecordDecl *target_decl =
      TypeSystemClang::GetAsCXXRecordDecl(dyn_type.GetOpaqueQualType());
  if (!in_decl || !target_decl) {
    LLDB_LOG(log, "{0:x16}: types are not record types (in={1}, target={2})",
             in_value.GetPointerValue().address, in_base_type.GetTypeName(),
             dyn_type.GetTypeName());
    return std::nullopt;
  }

  TypeSystemClangSP ts = dyn_type.GetTypeSystem<TypeSystemClang>();
  if (!ts || ts != in_base_type.GetTypeSystem<TypeSystemClang>()) {
    LLDB_LOG(GetLog(LLDBLog::Object),
             "{0:x16}: '{1}' and '{2}' are from different type systems",
             in_value.GetPointerValue().address,
             in_decl->getDeclName().getAsString(),
             target_decl->getDeclName().getAsString());
    return std::nullopt;
  }

  std::optional<size_t> offset = GetOffsetOfBaseFromAST(
      ts->getASTContext(), target_decl, in_decl, vtable_components);
  if (!offset) {
    LLDB_LOG(GetLog(LLDBLog::Object),
             "{0:x16}: failed to find offset of '{1}' in '{2}'",
             in_value.GetPointerValue().address,
             in_decl->getDeclName().getAsString(),
             target_decl->getDeclName().getAsString());
    return std::nullopt;
  }

  return offset;
}
