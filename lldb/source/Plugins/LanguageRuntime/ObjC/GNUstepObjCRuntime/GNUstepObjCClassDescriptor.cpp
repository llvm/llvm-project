//===-- GNUstepObjCClassDescriptor.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GNUstepObjCClassDescriptor.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"
#include "lldb/ValueObject/ValueObjectVariable.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace lldb;
using namespace lldb_private;

// Flags from libobjc2's `enum objc_class_flags` (class.h).
static constexpr uint64_t g_class_flag_meta = 1ULL << 0;
static constexpr uint64_t g_class_flag_resolved = 1ULL << 9;

// An upper bound for plausible class names. A string that does not terminate
// within this many bytes is not a class name, and stopping there keeps a
// stray pointer from dragging in arbitrary amounts of inferior memory.
static constexpr size_t g_max_class_name_length = 256;

namespace {
/// Offsets of the `struct objc_class` fields this descriptor reads. The first
/// three fields are pointers; the rest are `long`, which is not always the
/// same width (see the class documentation).
struct ClassLayout {
  uint32_t pointer_size;
  uint32_t long_size;
  uint64_t superclass_offset;
  uint64_t name_offset;
  uint64_t info_offset;
  uint64_t instance_size_offset;
};

ClassLayout GetClassLayout(Process &process) {
  ClassLayout layout;
  layout.pointer_size = process.GetAddressByteSize();
  // Windows is LLP64, so `long` stays 32 bits there while pointers are 64.
  const llvm::Triple &triple =
      process.GetTarget().GetArchitecture().GetTriple();
  layout.long_size = (triple.isOSWindows() && layout.pointer_size == 8)
                         ? 4
                         : layout.pointer_size;
  layout.superclass_offset = layout.pointer_size;
  layout.name_offset = 2 * layout.pointer_size;
  layout.info_offset = 3 * layout.pointer_size + layout.long_size;
  layout.instance_size_offset = 3 * layout.pointer_size + 2 * layout.long_size;
  return layout;
}
} // namespace

GNUstepObjCClassDescriptor::GNUstepObjCClassDescriptor(
    ProcessSP process_sp, ObjCLanguageRuntime::ObjCISA isa)
    : m_process_wp(process_sp), m_isa(isa) {
  Read();
}

void GNUstepObjCClassDescriptor::Read() {
  ProcessSP process_sp = m_process_wp.lock();
  if (!process_sp || m_isa == 0 || m_isa == LLDB_INVALID_ADDRESS)
    return;

  const ClassLayout layout = GetClassLayout(*process_sp);
  // Class objects are at least pointer-aligned.
  if (m_isa % layout.pointer_size != 0)
    return;

  Status error;
  auto read_pointer = [&](uint64_t offset) -> addr_t {
    addr_t value = process_sp->ReadPointerFromMemory(m_isa + offset, error);
    return error.Fail() ? LLDB_INVALID_ADDRESS : value;
  };

  const addr_t metaclass = read_pointer(0);
  if (metaclass == 0 || metaclass == LLDB_INVALID_ADDRESS)
    return;
  const addr_t superclass = read_pointer(layout.superclass_offset);
  if (superclass == LLDB_INVALID_ADDRESS)
    return;
  const addr_t name_ptr = read_pointer(layout.name_offset);
  if (name_ptr == 0 || name_ptr == LLDB_INVALID_ADDRESS)
    return;

  char name_buffer[g_max_class_name_length];
  const size_t name_length = process_sp->ReadCStringFromMemory(
      name_ptr, name_buffer, sizeof(name_buffer), error);
  // A string that fills the buffer was truncated, so it is not a class name.
  if (error.Fail() || name_length == 0 ||
      name_length >= sizeof(name_buffer) - 1)
    return;

  const uint64_t info = process_sp->ReadUnsignedIntegerFromMemory(
      m_isa + layout.info_offset, layout.long_size, 0, error);
  if (error.Fail())
    return;

  // A class and its metaclass must disagree about the meta flag. Checking
  // both directions is what keeps an arbitrary readable address from being
  // accepted as a class.
  const bool is_meta = (info & g_class_flag_meta) != 0;
  if (!is_meta) {
    const uint64_t metaclass_info = process_sp->ReadUnsignedIntegerFromMemory(
        metaclass + layout.info_offset, layout.long_size, 0, error);
    if (error.Fail() || (metaclass_info & g_class_flag_meta) == 0)
      return;
  }

  // Only a resolved class has a real superclass pointer and instance size;
  // see the class documentation.
  const bool resolved = (info & g_class_flag_resolved) != 0;
  if (resolved) {
    const int64_t instance_size = process_sp->ReadSignedIntegerFromMemory(
        m_isa + layout.instance_size_offset, layout.long_size, 0, error);
    if (error.Fail())
      return;
    m_instance_size = static_cast<uint64_t>(instance_size < 0 ? -instance_size
                                                              : instance_size);
    m_superclass_isa = superclass;
  }

  m_metaclass_isa = metaclass;
  m_is_meta = is_meta;
  m_name = ConstString(name_buffer);
  m_valid = true;
}

ObjCLanguageRuntime::ClassDescriptorSP
GNUstepObjCClassDescriptor::GetSuperclass() {
  if (!m_valid || m_superclass_isa == 0)
    return ObjCLanguageRuntime::ClassDescriptorSP();
  // A class that is its own superclass would make any walk up the chain spin
  // forever. The chain comes from inferior memory, so refuse the edge here
  // rather than leaving every caller to remember. Longer cycles cannot be
  // seen from a single descriptor and still need the caller's depth bound.
  if (m_superclass_isa == m_isa)
    return ObjCLanguageRuntime::ClassDescriptorSP();
  ProcessSP process_sp = m_process_wp.lock();
  if (!process_sp)
    return ObjCLanguageRuntime::ClassDescriptorSP();
  return std::make_shared<GNUstepObjCClassDescriptor>(process_sp,
                                                      m_superclass_isa);
}

std::unique_ptr<ObjCLanguageRuntime::ClassDescriptor>
GNUstepObjCClassDescriptor::GetMetaclass() const {
  if (!m_valid || m_metaclass_isa == 0)
    return nullptr;
  ProcessSP process_sp = m_process_wp.lock();
  if (!process_sp)
    return nullptr;
  return std::make_unique<GNUstepObjCClassDescriptor>(process_sp,
                                                      m_metaclass_isa);
}

bool GNUstepObjCTaggedPointerClassDescriptor::GetTaggedPointerInfo(
    uint64_t *info_bits, uint64_t *value_bits, uint64_t *payload) {
  if (info_bits)
    *info_bits = m_tag;
  if (value_bits)
    *value_bits = m_pointer_value >> m_payload_shift;
  if (payload)
    *payload = m_pointer_value;
  return true;
}

bool GNUstepObjCTaggedPointerClassDescriptor::GetTaggedPointerInfoSigned(
    uint64_t *info_bits, int64_t *value_bits, uint64_t *payload) {
  if (info_bits)
    *info_bits = m_tag;
  if (value_bits) {
    // Sign-extend from the target's pointer width before shifting, so that a
    // negative payload in a 32-bit pointer is not read as a large positive.
    // Done through SignExtend64 rather than by hand: shifting a signed value
    // into its own sign bit is undefined.
    const uint32_t pointer_bits = m_pointer_size * 8;
    *value_bits =
        llvm::SignExtend64(m_pointer_value, pointer_bits) >> m_payload_shift;
  }
  if (payload)
    *payload = m_pointer_value;
  return true;
}

bool GNUstepTaggedPointerVendor::IsPossibleTaggedPointer(lldb::addr_t ptr) {
  const uint64_t mask = m_process.GetAddressByteSize() == 8 ? 7 : 1;
  return (ptr & mask) != 0;
}

std::unique_ptr<ObjCLanguageRuntime::ClassDescriptor>
GNUstepTaggedPointerVendor::GetClassDescriptor(lldb::addr_t ptr) {
  const uint32_t pointer_size = m_process.GetAddressByteSize();
  const bool is_64_bit = pointer_size == 8;
  const uint64_t mask = is_64_bit ? 7 : 1;
  const uint32_t payload_shift = is_64_bit ? 3 : 1;
  const uint64_t tag = ptr & mask;
  if (tag == 0)
    return nullptr;

  // Mirror libobjc2's classForObject(): 32-bit targets have a single small
  // object class at index 0; 64-bit targets index the table by the tag.
  // `SmallObjectClasses` has 7 entries (class_table.c), so a tag of 7 has no
  // corresponding class.
  const uint64_t index = is_64_bit ? tag : 0;
  if (index > 6)
    return nullptr;

  if (!m_table_addr) {
    m_table_addr = LLDB_INVALID_ADDRESS;
    Target &target = m_process.GetTarget();
    SymbolContextList sc_list;
    target.GetImages().FindSymbolsWithNameAndType(
        ConstString("SmallObjectClasses"), eSymbolTypeAny, sc_list);
    for (const SymbolContext &sc : sc_list) {
      if (!sc.symbol)
        continue;
      const addr_t table = sc.symbol->GetAddress().GetLoadAddress(&target);
      if (table != LLDB_INVALID_ADDRESS) {
        m_table_addr = table;
        break;
      }
    }
    // The table has hidden visibility, so a linked image carries no symbol
    // for it unless a PDB or an unstripped symtab is around; the debug info
    // still describes it as a global, which is enough to find its address.
    if (*m_table_addr == LLDB_INVALID_ADDRESS) {
      VariableList variables;
      target.GetImages().FindGlobalVariables(ConstString("SmallObjectClasses"),
                                             1, variables);
      if (VariableSP variable_sp = variables.GetVariableAtIndex(0)) {
        ValueObjectSP valobj_sp =
            ValueObjectVariable::Create(&target, variable_sp);
        if (valobj_sp) {
          const addr_t table = valobj_sp->GetAddressOf(false).address;
          if (table != 0 && table != LLDB_INVALID_ADDRESS)
            m_table_addr = table;
        }
      }
    }
    if (*m_table_addr == LLDB_INVALID_ADDRESS)
      LLDB_LOG(GetLog(LLDBLog::Language),
               "GNUstepTaggedPointerVendor: SmallObjectClasses not found in "
               "any symbol table or debug info (stripped libobjc?); tagged "
               "pointer classes unknown");
  }
  if (*m_table_addr == LLDB_INVALID_ADDRESS)
    return nullptr;

  Status error;
  const addr_t isa = m_process.ReadPointerFromMemory(
      *m_table_addr + index * pointer_size, error);
  if (error.Fail() || isa == 0 || isa == LLDB_INVALID_ADDRESS)
    return nullptr;

  auto descriptor_up =
      std::make_unique<GNUstepObjCTaggedPointerClassDescriptor>(
          m_process.shared_from_this(), isa, ptr, tag, payload_shift,
          pointer_size);
  if (!descriptor_up->IsValid())
    return nullptr;
  return descriptor_up;
}
