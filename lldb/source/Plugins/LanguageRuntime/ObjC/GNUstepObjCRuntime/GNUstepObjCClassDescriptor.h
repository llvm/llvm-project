//===-- GNUstepObjCClassDescriptor.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_OBJC_GNUSTEPOBJCRUNTIME_GNUSTEPOBJCCLASSDESCRIPTOR_H
#define LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_OBJC_GNUSTEPOBJCRUNTIME_GNUSTEPOBJCCLASSDESCRIPTOR_H

#include "Plugins/LanguageRuntime/ObjC/ObjCLanguageRuntime.h"

#include "lldb/lldb-forward.h"
#include "lldb/lldb-types.h"

#include <optional>

namespace lldb_private {

/// A class descriptor for classes of the GNUstep libobjc2 runtime, backed
/// entirely by reads of the inferior's memory - no code is ever executed in
/// the inferior.
///
/// The layout parsed here is libobjc2's `struct objc_class` (class.h), whose
/// leading fields have been stable across the gnustep-2.x ABI:
///
///   Class isa;              // metaclass          [index 0]
///   Class super_class;      //                    [index 1]
///   const char *name;       //                    [index 2]
///   long version;           //                    [index 3]
///   unsigned long info;     // flag bits          [index 4]
///   long instance_size;     //                    [index 5]
///
/// Note: with the non-fragile ABI the compiler emits a negative
/// instance_size; the runtime replaces it with the real size when the class
/// is registered, so debug-time reads of loaded classes see the real value.
class GNUstepObjCClassDescriptor : public ObjCLanguageRuntime::ClassDescriptor {
public:
  GNUstepObjCClassDescriptor(lldb::ProcessSP process_sp,
                             ObjCLanguageRuntime::ObjCISA isa);

  ~GNUstepObjCClassDescriptor() override = default;

  ConstString GetClassName() override { return m_name; }

  ObjCLanguageRuntime::ClassDescriptorSP GetSuperclass() override;

  std::unique_ptr<ObjCLanguageRuntime::ClassDescriptor>
  GetMetaclass() const override;

  bool IsValid() override { return m_valid; }

  bool GetTaggedPointerInfo(uint64_t *info_bits = nullptr,
                            uint64_t *value_bits = nullptr,
                            uint64_t *payload = nullptr) override {
    return false;
  }

  bool GetTaggedPointerInfoSigned(uint64_t *info_bits = nullptr,
                                  int64_t *value_bits = nullptr,
                                  uint64_t *payload = nullptr) override {
    return false;
  }

  uint64_t GetInstanceSize() override { return m_instance_size; }

  ObjCLanguageRuntime::ObjCISA GetISA() override { return m_isa; }

protected:
  /// Parse `struct objc_class` at m_isa. Called from the constructor;
  /// sets m_valid on success.
  void Read();

  lldb::ProcessWP m_process_wp;
  ObjCLanguageRuntime::ObjCISA m_isa = 0;
  ConstString m_name;
  ObjCLanguageRuntime::ObjCISA m_superclass_isa = 0;
  ObjCLanguageRuntime::ObjCISA m_metaclass_isa = 0;
  uint64_t m_instance_size = 0;
  bool m_valid = false;
};

/// Class descriptor for libobjc2 "small objects" (tagged pointers). The
/// pointed-to class is the entry of the runtime's `SmallObjectClasses` table
/// selected by the low tag bits; the descriptor additionally exposes the
/// payload via GetTaggedPointerInfo.
class GNUstepObjCTaggedPointerClassDescriptor
    : public GNUstepObjCClassDescriptor {
public:
  GNUstepObjCTaggedPointerClassDescriptor(lldb::ProcessSP process_sp,
                                          ObjCLanguageRuntime::ObjCISA isa,
                                          lldb::addr_t pointer_value,
                                          uint64_t tag, uint32_t payload_shift)
      : GNUstepObjCClassDescriptor(std::move(process_sp), isa),
        m_pointer_value(pointer_value), m_tag(tag),
        m_payload_shift(payload_shift) {}

  bool GetTaggedPointerInfo(uint64_t *info_bits = nullptr,
                            uint64_t *value_bits = nullptr,
                            uint64_t *payload = nullptr) override;

  bool GetTaggedPointerInfoSigned(uint64_t *info_bits = nullptr,
                                  int64_t *value_bits = nullptr,
                                  uint64_t *payload = nullptr) override;

private:
  lldb::addr_t m_pointer_value;
  uint64_t m_tag;
  uint32_t m_payload_shift;
};

/// Resolves tagged pointers by mirroring libobjc2's `classForObject()`
/// (class.h): a pointer with any of the low tag bits set (3 bits on 64-bit
/// targets, 1 bit on 32-bit targets) is a small object whose class is
/// `SmallObjectClasses[tag]` (index 0 on 32-bit targets).
///
/// `SmallObjectClasses` has hidden visibility, so resolving it requires the
/// library's .symtab (present in unstripped builds). If it cannot be
/// resolved, tagged pointers are still detected but their class is unknown.
class GNUstepTaggedPointerVendor
    : public ObjCLanguageRuntime::TaggedPointerVendor {
public:
  explicit GNUstepTaggedPointerVendor(Process &process) : m_process(process) {}

  ~GNUstepTaggedPointerVendor() override = default;

  bool IsPossibleTaggedPointer(lldb::addr_t ptr) override;

  std::unique_ptr<ObjCLanguageRuntime::ClassDescriptor>
  GetClassDescriptor(lldb::addr_t ptr) override;

private:
  /// Load address of libobjc2's `SmallObjectClasses` table, resolved lazily
  /// and cached. LLDB_INVALID_ADDRESS inside the optional means resolution
  /// was attempted and failed.
  std::optional<lldb::addr_t> m_table_addr;

  Process &m_process;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_OBJC_GNUSTEPOBJCRUNTIME_GNUSTEPOBJCCLASSDESCRIPTOR_H
