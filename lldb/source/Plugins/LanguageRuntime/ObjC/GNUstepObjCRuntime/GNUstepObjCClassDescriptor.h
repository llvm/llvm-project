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
///   Class isa;              // metaclass
///   Class super_class;
///   const char *name;
///   long version;
///   unsigned long info;     // enum objc_class_flags
///   long instance_size;
///
/// Note that the last three fields are `long`, which is 32 bits on Windows
/// (LLP64) and pointer-sized on the LP64 and ILP32 targets libobjc2 supports,
/// so field offsets are computed from the target's data model rather than from
/// the pointer size alone.
///
/// Classes emitted by the compiler are only fully formed once the runtime has
/// resolved them (`objc_class_flag_resolved`): before that, `super_class` may
/// still hold the superclass *name* rather than a Class, and `instance_size`
/// holds the negated size of just this class's own ivars. Both are therefore
/// only reported for resolved classes.
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

  /// True if this descriptor describes a metaclass, i.e. the ISA it was
  /// built from is itself the class pointer of a class object rather than of
  /// an instance. Instances never have a metaclass as their ISA, so a value
  /// that resolves to one is a Class, not an object.
  bool IsMetaclass() const { return m_is_meta; }

protected:
  /// Parse `struct objc_class` at m_isa. Called from the constructor; sets
  /// m_valid only if the structure passes the consistency checks that keep a
  /// stray pointer into readable memory from being reported as a class.
  void Read();

  lldb::ProcessWP m_process_wp;
  ObjCLanguageRuntime::ObjCISA m_isa = 0;
  ConstString m_name;
  ObjCLanguageRuntime::ObjCISA m_superclass_isa = 0;
  ObjCLanguageRuntime::ObjCISA m_metaclass_isa = 0;
  uint64_t m_instance_size = 0;
  bool m_is_meta = false;
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
                                          uint64_t tag, uint32_t payload_shift,
                                          uint32_t pointer_size)
      : GNUstepObjCClassDescriptor(std::move(process_sp), isa),
        m_pointer_value(pointer_value), m_tag(tag),
        m_payload_shift(payload_shift), m_pointer_size(pointer_size) {}

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
  uint32_t m_pointer_size;
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

  /// Forget where (or whether) the small object class table was found, so a
  /// newly loaded runtime is picked up and a negative result is not cached
  /// for the lifetime of the process.
  void ModulesDidLoad() { m_table_addr.reset(); }

private:
  /// Load address of libobjc2's `SmallObjectClasses` table, resolved lazily
  /// and cached. LLDB_INVALID_ADDRESS inside the optional means resolution
  /// was attempted and failed.
  std::optional<lldb::addr_t> m_table_addr;

  Process &m_process;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_OBJC_GNUSTEPOBJCRUNTIME_GNUSTEPOBJCCLASSDESCRIPTOR_H
