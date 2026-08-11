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
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"

using namespace lldb;
using namespace lldb_private;

// Field indices into libobjc2's `struct objc_class` (see class documentation
// in the header).
static constexpr uint64_t kClassFieldIsa = 0;
static constexpr uint64_t kClassFieldSuperclass = 1;
static constexpr uint64_t kClassFieldName = 2;
static constexpr uint64_t kClassFieldInstanceSize = 5;

// An upper bound for plausible class names; longer strings indicate that the
// name pointer does not actually point at a class name.
static constexpr size_t kMaxClassNameLength = 512;

GNUstepObjCClassDescriptor::GNUstepObjCClassDescriptor(
    ProcessSP process_sp, ObjCLanguageRuntime::ObjCISA isa)
    : m_process_wp(process_sp), m_isa(isa) {
  Read();
}

void GNUstepObjCClassDescriptor::Read() {
  ProcessSP process_sp = m_process_wp.lock();
  if (!process_sp || m_isa == 0 || m_isa == LLDB_INVALID_ADDRESS)
    return;

  const uint32_t addr_size = process_sp->GetAddressByteSize();
  // Class objects are at least pointer-aligned.
  if (m_isa % addr_size != 0)
    return;

  Status error;
  auto read_field = [&](uint64_t index) -> addr_t {
    addr_t value = process_sp->ReadPointerFromMemory(
        m_isa + index * addr_size, error);
    return error.Fail() ? LLDB_INVALID_ADDRESS : value;
  };

  const addr_t metaclass = read_field(kClassFieldIsa);
  if (metaclass == LLDB_INVALID_ADDRESS)
    return;
  const addr_t superclass = read_field(kClassFieldSuperclass);
  if (superclass == LLDB_INVALID_ADDRESS)
    return;
  const addr_t name_ptr = read_field(kClassFieldName);
  if (name_ptr == LLDB_INVALID_ADDRESS || name_ptr == 0)
    return;

  std::string name;
  process_sp->ReadCStringFromMemory(name_ptr, name, error);
  if (error.Fail() || name.empty() || name.size() >= kMaxClassNameLength)
    return;

  // `instance_size` is a signed `long`. With the non-fragile ABI it is
  // negative until the runtime registers the class; take the magnitude so a
  // not-yet-registered class still yields a usable size.
  const int64_t instance_size = process_sp->ReadSignedIntegerFromMemory(
      m_isa + kClassFieldInstanceSize * addr_size, addr_size, 0, error);
  if (error.Fail())
    return;

  m_metaclass_isa = metaclass;
  m_superclass_isa = superclass;
  m_name = ConstString(name);
  m_instance_size = static_cast<uint64_t>(
      instance_size < 0 ? -instance_size : instance_size);
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
  if (value_bits)
    *value_bits =
        static_cast<int64_t>(m_pointer_value) >> m_payload_shift;
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
  const bool is_64_bit = m_process.GetAddressByteSize() == 8;
  const uint64_t mask = is_64_bit ? 7 : 1;
  const uint32_t payload_shift = is_64_bit ? 3 : 1;
  const uint64_t tag = ptr & mask;
  if (tag == 0)
    return nullptr;

  // Mirror libobjc2's classForObject(): 32-bit targets have a single small
  // object class at index 0; 64-bit targets index the table by the tag. The
  // table has 7 entries, so reject out-of-range tags.
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
    if (*m_table_addr == LLDB_INVALID_ADDRESS)
      LLDB_LOG(GetLog(LLDBLog::Language),
               "GNUstepTaggedPointerVendor: SmallObjectClasses symbol not "
               "found (stripped libobjc?); tagged pointer classes unknown");
  }
  if (*m_table_addr == LLDB_INVALID_ADDRESS)
    return nullptr;

  Status error;
  const addr_t isa = m_process.ReadPointerFromMemory(
      *m_table_addr + index * m_process.GetAddressByteSize(), error);
  if (error.Fail() || isa == 0 || isa == LLDB_INVALID_ADDRESS)
    return nullptr;

  auto descriptor_up =
      std::make_unique<GNUstepObjCTaggedPointerClassDescriptor>(
          m_process.shared_from_this(), isa, ptr, tag, payload_shift);
  if (!descriptor_up->IsValid())
    return nullptr;
  return descriptor_up;
}
