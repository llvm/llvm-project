//===-- AppleObjCRuntimeV2.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_OBJC_APPLEOBJCRUNTIME_APPLEOBJCRUNTIMEV2_H
#define LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_OBJC_APPLEOBJCRUNTIME_APPLEOBJCRUNTIMEV2_H

#include <map>
#include <memory>
#include <mutex>

#include "AppleObjCRuntime.h"
#include "lldb/lldb-private.h"

#include "Plugins/LanguageRuntime/ObjC/ObjCLanguageRuntime.h"

class RemoteNXMapTable;

namespace lldb_private {

class AppleObjCRuntimeV2 : public AppleObjCRuntime {
public:
  ~AppleObjCRuntimeV2() override = default;

  static void Initialize();

  static void Terminate();

  static lldb_private::LanguageRuntime *
  CreateInstance(Process *process, lldb::LanguageType language);

  static lldb_private::ConstString GetPluginNameStatic();

  static char ID;

  bool isA(const void *ClassID) const override {
    return ClassID == &ID || AppleObjCRuntime::isA(ClassID);
  }

  static bool classof(const LanguageRuntime *runtime) {
    return runtime->isA(&ID);
  }

  bool GetDynamicTypeAndAddress(ValueObject &in_value,
                                lldb::DynamicValueType use_dynamic,
                                TypeAndOrName &class_type_or_name,
                                Address &address,
                                Value::ValueType &value_type) override;

  llvm::Expected<std::unique_ptr<UtilityFunction>>
  CreateObjectChecker(std::string name, ExecutionContext &exe_ctx) override;

  ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

  ObjCRuntimeVersions GetRuntimeVersion() const override {
    return ObjCRuntimeVersions::eAppleObjC_V2;
  }

  size_t GetByteOffsetForIvar(CompilerType &parent_qual_type,
                              const char *ivar_name) override;

  void UpdateISAToDescriptorMapIfNeeded() override;

  ClassDescriptorSP GetClassDescriptor(ValueObject &in_value) override;

  ClassDescriptorSP GetClassDescriptorFromISA(ObjCISA isa) override;

  DeclVendor *GetDeclVendor() override;

  lldb::addr_t LookupRuntimeSymbol(ConstString name) override;

  EncodingToTypeSP GetEncodingToType() override;

  bool IsTaggedPointer(lldb::addr_t ptr) override;

  TaggedPointerVendor *GetTaggedPointerVendor() override {
    return m_tagged_pointer_vendor_up.get();
  }

  lldb::addr_t GetTaggedPointerObfuscator();

  void GetValuesForGlobalCFBooleans(lldb::addr_t &cf_true,
                                    lldb::addr_t &cf_false) override;

  // none of these are valid ISAs - we use them to infer the type
  // of tagged pointers - if we have something meaningful to say
  // we report an actual type - otherwise, we just say tagged
  // there is no connection between the values here and the tagged pointers map
  static const ObjCLanguageRuntime::ObjCISA g_objc_Tagged_ISA = 1;
  static const ObjCLanguageRuntime::ObjCISA g_objc_Tagged_ISA_NSAtom = 2;
  static const ObjCLanguageRuntime::ObjCISA g_objc_Tagged_ISA_NSNumber = 3;
  static const ObjCLanguageRuntime::ObjCISA g_objc_Tagged_ISA_NSDateTS = 4;
  static const ObjCLanguageRuntime::ObjCISA g_objc_Tagged_ISA_NSManagedObject =
      5;
  static const ObjCLanguageRuntime::ObjCISA g_objc_Tagged_ISA_NSDate = 6;

protected:
  lldb::BreakpointResolverSP
  CreateExceptionResolver(const lldb::BreakpointSP &bkpt, bool catch_bp,
                          bool throw_bp) override;

private:
  class HashTableSignature {
  public:
    HashTableSignature();

    bool NeedsUpdate(Process *process, AppleObjCRuntimeV2 *runtime,
                     RemoteNXMapTable &hash_table);

    void UpdateSignature(const RemoteNXMapTable &hash_table);

  protected:
    uint32_t m_count;
    uint32_t m_num_buckets;
    lldb::addr_t m_buckets_ptr;
  };

  class NonPointerISACache {
  public:
    static NonPointerISACache *
    CreateInstance(AppleObjCRuntimeV2 &runtime,
                   const lldb::ModuleSP &objc_module_sp);

    ObjCLanguageRuntime::ClassDescriptorSP GetClassDescriptor(ObjCISA isa);

  private:
    NonPointerISACache(AppleObjCRuntimeV2 &runtime,
                       const lldb::ModuleSP &objc_module_sp,
                       uint64_t objc_debug_isa_class_mask,
                       uint64_t objc_debug_isa_magic_mask,
                       uint64_t objc_debug_isa_magic_value,
                       uint64_t objc_debug_indexed_isa_magic_mask,
                       uint64_t objc_debug_indexed_isa_magic_value,
                       uint64_t objc_debug_indexed_isa_index_mask,
                       uint64_t objc_debug_indexed_isa_index_shift,
                       lldb::addr_t objc_indexed_classes);

    bool EvaluateNonPointerISA(ObjCISA isa, ObjCISA &ret_isa);

    AppleObjCRuntimeV2 &m_runtime;
    std::map<ObjCISA, ObjCLanguageRuntime::ClassDescriptorSP> m_cache;
    lldb::ModuleWP m_objc_module_wp;
    uint64_t m_objc_debug_isa_class_mask;
    uint64_t m_objc_debug_isa_magic_mask;
    uint64_t m_objc_debug_isa_magic_value;

    uint64_t m_objc_debug_indexed_isa_magic_mask;
    uint64_t m_objc_debug_indexed_isa_magic_value;
    uint64_t m_objc_debug_indexed_isa_index_mask;
    uint64_t m_objc_debug_indexed_isa_index_shift;
    lldb::addr_t m_objc_indexed_classes;

    std::vector<lldb::addr_t> m_indexed_isa_cache;

    friend class AppleObjCRuntimeV2;

    NonPointerISACache(const NonPointerISACache &) = delete;
    const NonPointerISACache &operator=(const NonPointerISACache &) = delete;
  };

  class TaggedPointerVendorV2
      : public ObjCLanguageRuntime::TaggedPointerVendor {
  public:
    ~TaggedPointerVendorV2() override = default;

    static TaggedPointerVendorV2 *
    CreateInstance(AppleObjCRuntimeV2 &runtime,
                   const lldb::ModuleSP &objc_module_sp);

  protected:
    AppleObjCRuntimeV2 &m_runtime;

    TaggedPointerVendorV2(AppleObjCRuntimeV2 &runtime)
        : TaggedPointerVendor(), m_runtime(runtime) {}

  private:
    TaggedPointerVendorV2(const TaggedPointerVendorV2 &) = delete;
    const TaggedPointerVendorV2 &
    operator=(const TaggedPointerVendorV2 &) = delete;
  };

  class TaggedPointerVendorRuntimeAssisted : public TaggedPointerVendorV2 {
  public:
    bool IsPossibleTaggedPointer(lldb::addr_t ptr) override;

    ObjCLanguageRuntime::ClassDescriptorSP
    GetClassDescriptor(lldb::addr_t ptr) override;

  protected:
    TaggedPointerVendorRuntimeAssisted(
        AppleObjCRuntimeV2 &runtime, uint64_t objc_debug_taggedpointer_mask,
        uint32_t objc_debug_taggedpointer_slot_shift,
        uint32_t objc_debug_taggedpointer_slot_mask,
        uint32_t objc_debug_taggedpointer_payload_lshift,
        uint32_t objc_debug_taggedpointer_payload_rshift,
        lldb::addr_t objc_debug_taggedpointer_classes);

    typedef std::map<uint8_t, ObjCLanguageRuntime::ClassDescriptorSP> Cache;
    typedef Cache::iterator CacheIterator;
    Cache m_cache;
    uint64_t m_objc_debug_taggedpointer_mask;
    uint32_t m_objc_debug_taggedpointer_slot_shift;
    uint32_t m_objc_debug_taggedpointer_slot_mask;
    uint32_t m_objc_debug_taggedpointer_payload_lshift;
    uint32_t m_objc_debug_taggedpointer_payload_rshift;
    lldb::addr_t m_objc_debug_taggedpointer_classes;

    friend class AppleObjCRuntimeV2::TaggedPointerVendorV2;

    TaggedPointerVendorRuntimeAssisted(
        const TaggedPointerVendorRuntimeAssisted &) = delete;
    const TaggedPointerVendorRuntimeAssisted &
    operator=(const TaggedPointerVendorRuntimeAssisted &) = delete;
  };

  class TaggedPointerVendorExtended
      : public TaggedPointerVendorRuntimeAssisted {
  public:
    ObjCLanguageRuntime::ClassDescriptorSP
    GetClassDescriptor(lldb::addr_t ptr) override;

  protected:
    TaggedPointerVendorExtended(
        AppleObjCRuntimeV2 &runtime, uint64_t objc_debug_taggedpointer_mask,
        uint64_t objc_debug_taggedpointer_ext_mask,
        uint32_t objc_debug_taggedpointer_slot_shift,
        uint32_t objc_debug_taggedpointer_ext_slot_shift,
        uint32_t objc_debug_taggedpointer_slot_mask,
        uint32_t objc_debug_taggedpointer_ext_slot_mask,
        uint32_t objc_debug_taggedpointer_payload_lshift,
        uint32_t objc_debug_taggedpointer_payload_rshift,
        uint32_t objc_debug_taggedpointer_ext_payload_lshift,
        uint32_t objc_debug_taggedpointer_ext_payload_rshift,
        lldb::addr_t objc_debug_taggedpointer_classes,
        lldb::addr_t objc_debug_taggedpointer_ext_classes);

    bool IsPossibleExtendedTaggedPointer(lldb::addr_t ptr);

    typedef std::map<uint8_t, ObjCLanguageRuntime::ClassDescriptorSP> Cache;
    typedef Cache::iterator CacheIterator;
    Cache m_ext_cache;
    uint64_t m_objc_debug_taggedpointer_ext_mask;
    uint32_t m_objc_debug_taggedpointer_ext_slot_shift;
    uint32_t m_objc_debug_taggedpointer_ext_slot_mask;
    uint32_t m_objc_debug_taggedpointer_ext_payload_lshift;
    uint32_t m_objc_debug_taggedpointer_ext_payload_rshift;
    lldb::addr_t m_objc_debug_taggedpointer_ext_classes;

    friend class AppleObjCRuntimeV2::TaggedPointerVendorV2;

    TaggedPointerVendorExtended(const TaggedPointerVendorExtended &) = delete;
    const TaggedPointerVendorExtended &
    operator=(const TaggedPointerVendorExtended &) = delete;
  };

  class TaggedPointerVendorLegacy : public TaggedPointerVendorV2 {
  public:
    bool IsPossibleTaggedPointer(lldb::addr_t ptr) override;

    ObjCLanguageRuntime::ClassDescriptorSP
    GetClassDescriptor(lldb::addr_t ptr) override;

  protected:
    TaggedPointerVendorLegacy(AppleObjCRuntimeV2 &runtime)
        : TaggedPointerVendorV2(runtime) {}

    friend class AppleObjCRuntimeV2::TaggedPointerVendorV2;

    TaggedPointerVendorLegacy(const TaggedPointerVendorLegacy &) = delete;
    const TaggedPointerVendorLegacy &
    operator=(const TaggedPointerVendorLegacy &) = delete;
  };

  struct DescriptorMapUpdateResult {
    bool m_update_ran;
    uint32_t m_num_found;

    DescriptorMapUpdateResult(bool ran, uint32_t found) {
      m_update_ran = ran;
      m_num_found = found;
    }

    static DescriptorMapUpdateResult Fail() { return {false, 0}; }

    static DescriptorMapUpdateResult Success(uint32_t found) {
      return {true, found};
    }
  };

  /// Abstraction to read the Objective-C class info.
  class ClassInfoExtractor {
  public:
    ClassInfoExtractor(AppleObjCRuntimeV2 &runtime) : m_runtime(runtime) {}
    std::mutex &GetMutex() { return m_mutex; }

  protected:
    /// The lifetime of this object is tied to that of the runtime.
    AppleObjCRuntimeV2 &m_runtime;
    std::mutex m_mutex;
  };

  /// We can read the class info from the Objective-C runtime using
  /// gdb_objc_realized_classes or objc_copyRealizedClassList. The latter is
  /// preferred because it includes lazily named classes, but it's not always
  /// available or safe to call.
  ///
  /// We potentially need both for the same process, because we may need to use
  /// gdb_objc_realized_classes until dyld is initialized and then switch over
  /// to objc_copyRealizedClassList for lazily named classes.
  class DynamicClassInfoExtractor : public ClassInfoExtractor {
  public:
    DynamicClassInfoExtractor(AppleObjCRuntimeV2 &runtime)
        : ClassInfoExtractor(runtime) {}

    DescriptorMapUpdateResult
    UpdateISAToDescriptorMap(RemoteNXMapTable &hash_table);

  private:
    enum Helper { gdb_objc_realized_classes, objc_copyRealizedClassList };

    /// Compute which helper to use. Prefer objc_copyRealizedClassList if it's
    /// available and it's safe to call (i.e. dyld is fully initialized). Use
    /// gdb_objc_realized_classes otherwise.
    Helper ComputeHelper() const;

    UtilityFunction *GetClassInfoUtilityFunction(ExecutionContext &exe_ctx,
                                                 Helper helper);
    lldb::addr_t &GetClassInfoArgs(Helper helper);

    std::unique_ptr<UtilityFunction>
    GetClassInfoUtilityFunctionImpl(ExecutionContext &exe_ctx, std::string code,
                                    std::string name);

    /// Helper to read class info using the gdb_objc_realized_classes.
    struct gdb_objc_realized_classes_helper {
      std::unique_ptr<UtilityFunction> utility_function;
      lldb::addr_t args = LLDB_INVALID_ADDRESS;
    };

    /// Helper to read class info using objc_copyRealizedClassList.
    struct objc_copyRealizedClassList_helper {
      std::unique_ptr<UtilityFunction> utility_function;
      lldb::addr_t args = LLDB_INVALID_ADDRESS;
    };

    gdb_objc_realized_classes_helper m_gdb_objc_realized_classes_helper;
    objc_copyRealizedClassList_helper m_objc_copyRealizedClassList_helper;
  };

  /// Abstraction to read the Objective-C class info from the shared cache.
  class SharedCacheClassInfoExtractor : public ClassInfoExtractor {
  public:
    SharedCacheClassInfoExtractor(AppleObjCRuntimeV2 &runtime)
        : ClassInfoExtractor(runtime) {}

    DescriptorMapUpdateResult UpdateISAToDescriptorMap();

  private:
    UtilityFunction *GetClassInfoUtilityFunction(ExecutionContext &exe_ctx);

    std::unique_ptr<UtilityFunction>
    GetClassInfoUtilityFunctionImpl(ExecutionContext &exe_ctx);

    std::unique_ptr<UtilityFunction> m_utility_function;
    lldb::addr_t m_args = LLDB_INVALID_ADDRESS;
  };

  AppleObjCRuntimeV2(Process *process, const lldb::ModuleSP &objc_module_sp);

  ObjCISA GetPointerISA(ObjCISA isa);

  lldb::addr_t GetISAHashTablePointer();

  /// Update the generation count of realized classes. This is not an exact
  /// count but rather a value that is incremented when new classes are realized
  /// or destroyed. Unlike the count in gdb_objc_realized_classes, it will
  /// change when lazily named classes get realized.
  bool RealizedClassGenerationCountChanged();

  uint32_t ParseClassInfoArray(const lldb_private::DataExtractor &data,
                               uint32_t num_class_infos);

  enum class SharedCacheWarningReason {
    eExpressionExecutionFailure,
    eNotEnoughClassesRead
  };

  void WarnIfNoClassesCached(SharedCacheWarningReason reason);

  lldb::addr_t GetSharedCacheReadOnlyAddress();

  bool GetCFBooleanValuesIfNeeded();

  bool HasSymbol(ConstString Name);

  NonPointerISACache *GetNonPointerIsaCache() {
    if (!m_non_pointer_isa_cache_up)
      m_non_pointer_isa_cache_up.reset(
          NonPointerISACache::CreateInstance(*this, m_objc_module_sp));
    return m_non_pointer_isa_cache_up.get();
  }

  friend class ClassDescriptorV2;

  lldb::ModuleSP m_objc_module_sp;

  DynamicClassInfoExtractor m_dynamic_class_info_extractor;
  SharedCacheClassInfoExtractor m_shared_cache_class_info_extractor;

  std::unique_ptr<DeclVendor> m_decl_vendor_up;
  lldb::addr_t m_tagged_pointer_obfuscator;
  lldb::addr_t m_isa_hash_table_ptr;
  HashTableSignature m_hash_signature;
  bool m_has_object_getClass;
  bool m_has_objc_copyRealizedClassList;
  bool m_loaded_objc_opt;
  std::unique_ptr<NonPointerISACache> m_non_pointer_isa_cache_up;
  std::unique_ptr<TaggedPointerVendor> m_tagged_pointer_vendor_up;
  EncodingToTypeSP m_encoding_to_type_sp;
  bool m_noclasses_warning_emitted;
  llvm::Optional<std::pair<lldb::addr_t, lldb::addr_t>> m_CFBoolean_values;
  uint64_t m_realized_class_generation_count;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_OBJC_APPLEOBJCRUNTIME_APPLEOBJCRUNTIMEV2_H
