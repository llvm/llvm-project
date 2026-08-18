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

#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

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

  /// True once libobjc2 has resolved the class: before that its superclass
  /// pointer may still hold a name, its instance size is the negated size of
  /// its own ivars, and its ivar offsets are not yet absolute. A descriptor
  /// built in that state is a snapshot of it, so callers that cache
  /// descriptors must not cache an unresolved one.
  bool IsResolved() const { return m_resolved; }

  /// libobjc2's analogue of an Apple KVO subclass: a class the runtime hides
  /// from object_getClass(), which walks past it to the first visible
  /// superclass (runtime.c). Associated-object classes are the in-tree
  /// producer (associate.m).
  bool IsKVO() override { return m_is_hidden; }

  /// Reports this class's superclass, methods and ivars to the matching
  /// callbacks. Any callback may be null, and a method or ivar callback
  /// returning true stops the iteration.
  ///
  /// Class methods are collected from the metaclass, which is where libobjc2
  /// keeps them, as its instance methods.
  ///
  /// A method whose name cannot be recovered is skipped rather than reported
  /// under a placeholder: the name lives only in the symbol clang emitted for
  /// the selector, because __objc_load overwrites the name field in memory
  /// with a numeric dispatch index (selector_table.cc).
  ///
  /// Returns true if the class could be described at all.
  bool Describe(
      std::function<void(ObjCLanguageRuntime::ObjCISA)> const &superclass_func,
      std::function<bool(const char *, const char *)> const
          &instance_method_func,
      std::function<bool(const char *, const char *)> const &class_method_func,
      std::function<bool(const char *, const char *, lldb::addr_t,
                         uint64_t)> const &ivar_func) const override;

  size_t GetNumIVars() override;

  iVarDescriptor GetIVarAtIndex(size_t idx) override;

protected:
  /// One entry of libobjc2's `objc_ivar_list`, as read from memory. The type
  /// is kept as its encoding here; turning it into a CompilerType needs the
  /// runtime, which this class deliberately does not depend on.
  struct RawIvar {
    ConstString name;
    std::string type_encoding;
    int32_t offset = 0;
    uint32_t size = 0;
  };

  /// One entry of libobjc2's `objc_method_list`. The selector is kept as its
  /// address: after __objc_load its name field holds a dispatch index rather
  /// than a string, so the name has to come from the symbol emitted for it.
  struct RawMethod {
    lldb::addr_t selector = 0;
    std::string types;
  };

  /// Reads the methods this class implements, walking the whole list chain
  /// (categories are prepended to it). Pure memory reads.
  std::vector<RawMethod> ReadMethodList() const;

  /// Reads this class's own ivars, or an empty list if the class has none or
  /// its metadata is not yet trustworthy. Pure memory reads.
  ///
  /// The offsets libobjc2 stores are only meaningful once the runtime has
  /// computed them, and `objc_class_flag_resolved` alone does not say that:
  /// objc_resolve_class sets the flag *before* calling
  /// objc_compute_ivar_offsets (class_table.c), so a stop anywhere in that
  /// window - including a +load breakpoint while a sibling class is being
  /// resolved - would otherwise yield offsets relative to the start of this
  /// class's own ivars rather than to the object. objc_compute_ivar_offsets
  /// only runs while instance_size is non-positive and leaves it positive
  /// (ivar.c), so requiring a positive instance_size closes that window.
  std::vector<RawIvar> ReadIvarList() const;

  /// Realized ivars, filled on first use. Not cached while the class is
  /// unresolved: a descriptor lives in the runtime's ISA map for the life of
  /// the process, so caching an unresolved class's ivars would pin
  /// class-relative offsets forever.
  void GetIVarInformation();

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
  bool m_is_hidden = false;
  bool m_resolved = false;
  bool m_valid = false;

  std::vector<iVarDescriptor> m_ivars;
  bool m_ivars_filled = false;
  /// Guards the two above while they are filled, as
  /// AppleObjCClassDescriptorV2::iVarsStorage does. Recursive for the same
  /// reason: realizing an ivar's type can re-enter this runtime. Note that
  /// neither implementation protects a re-entrant *reader* - both return
  /// early on the filled flag - so this is about concurrent access.
  std::recursive_mutex m_ivars_mutex;
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
