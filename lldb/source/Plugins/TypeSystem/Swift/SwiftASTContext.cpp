//===-- SwiftASTContext.cpp -----------------------------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2020 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "Plugins/TypeSystem/Swift/SwiftASTContext.h"
#include "Plugins/TypeSystem/Swift/SwiftDWARFImporterForClangTypes.h"
#include "Plugins/TypeSystem/Swift/StoringDiagnosticConsumer.h"
#include "Plugins/ExpressionParser/Swift/SwiftPersistentExpressionState.h"

#include "SwiftASTContext.h"
#include "SwiftDemangle.h"
#include "TypeSystemSwift.h"
#include "TypeSystemSwiftTypeRef.h"
#include "lldb/Utility/Log.h"
#include "swift/AST/ASTContext.h"
#include "swift/AST/ASTDemangler.h"
#include "swift/AST/ASTMangler.h"
#include "swift/AST/ASTWalker.h"
#include "swift/AST/DebuggerClient.h"
#include "swift/AST/Decl.h"
#include "swift/AST/ExistentialLayout.h"
#include "swift/AST/GenericParamList.h"
#include "swift/AST/GenericSignature.h"
#include "swift/AST/IRGenOptions.h"
#include "swift/AST/ImportCache.h"
#include "swift/AST/ModuleLoader.h"
#include "swift/AST/NameLookup.h"
#include "swift/AST/OperatorNameLookup.h"
#include "swift/AST/PluginLoader.h"
#include "swift/AST/SearchPathOptions.h"
#include "swift/AST/SubstitutionMap.h"
#include "swift/AST/Type.h"
#include "swift/AST/Types.h"
#include "swift/ASTSectionImporter/ASTSectionImporter.h"
#include "swift/Basic/LLVM.h"
#include "swift/Basic/LangOptions.h"
#include "swift/Basic/Located.h"
#include "swift/Basic/Platform.h"
#include "swift/Basic/PrimarySpecificPaths.h"
#include "swift/ClangImporter/ClangImporter.h"
#include "swift/Demangling/Demangle.h"
#include "swift/Demangling/ManglingFlavor.h"
#include "swift/Demangling/ManglingMacros.h"
#include "swift/Frontend/Frontend.h"
#include "swift/Frontend/ModuleInterfaceLoader.h"
#include "swift/IRGen/Linking.h"
#include "swift/SIL/SILModule.h"
#include "swift/Sema/IDETypeChecking.h"
#include "swift/Serialization/SerializationOptions.h"
#include "swift/Serialization/Validation.h"
#include "swift/SymbolGraphGen/SymbolGraphOptions.h"

#include "clang/AST/ASTContext.h"
#include "clang/Basic/DarwinSDKInfo.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Driver/Driver.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include "swift/../../lib/IRGen/FixedTypeInfo.h"
#include "swift/../../lib/IRGen/GenEnum.h"
#include "swift/../../lib/IRGen/GenHeap.h"
#include "swift/../../lib/IRGen/IRGenModule.h"
#include "swift/../../lib/IRGen/TypeInfo.h"

#include "swift/Serialization/SerializedModuleLoader.h"
#include "swift/Strings.h"

#include "Plugins/ExpressionParser/Clang/ClangHost.h"
#include "Plugins/ExpressionParser/Swift/SwiftUserExpression.h"
#include "Plugins/Platform/MacOSX/PlatformDarwin.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/DumpDataExtractor.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/Progress.h"
#include "lldb/Core/Section.h"
#include "lldb/Expression/DiagnosticManager.h"
#include "lldb/Expression/IRExecutionUnit.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/StreamFile.h"
#include "lldb/Host/XML.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SourceModule.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/ThreadSafeDenseMap.h"
#include "lldb/Utility/Timer.h"
#include "lldb/Utility/XcodeSDK.h"

#include "llvm/ADT/ScopeExit.h"

#include "Plugins/Language/Swift/LogChannelSwift.h"
#include "Plugins/LanguageRuntime/Swift/SwiftLanguageRuntime.h"
#include "Plugins/SymbolFile/DWARF/DWARFASTParserClang.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"

#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>

namespace {

/// Used to sort the log output.
std::recursive_mutex g_log_mutex;

} // namespace

/// Similar to LLDB_LOG, but with richer contextual information.
#define LOG_PRINTF(CHANNEL, FMT, ...)                                          \
  LOG_PRINTF_IMPL(CHANNEL, false, FMT, ##__VA_ARGS__)

#define LOG_VERBOSE_PRINTF(CHANNEL, FMT, ...)                                  \
  LOG_PRINTF_IMPL(CHANNEL, true, FMT, ##__VA_ARGS__)

#define LOG_PRINTF_IMPL(LOG_CHANNEL, VERBOSE, FMT, ...)                        \
  do {                                                                         \
    if (Log *log = LOG_CHANNEL)                                                \
      if (!(VERBOSE) || log->GetVerbose()) {                                   \
        std::lock_guard<std::recursive_mutex> locker(g_log_mutex);             \
        /* The format string is optimized for code size, not speed. */         \
        log->Printf("%s::%s%s" FMT, m_description.c_str(), __FUNCTION__,       \
                    (FMT && FMT[0] == '(') ? "" : "() -- ", ##__VA_ARGS__);    \
      }                                                                        \
  } while (0)

#define HEALTH_LOG_PRINTF(FMT, ...)                                            \
  do {                                                                         \
    LOG_PRINTF(GetLog(LLDBLog::Types), FMT, ##__VA_ARGS__);                    \
    LOG_PRINTF_IMPL(lldb_private::GetSwiftHealthLog(), false, FMT,             \
                    ##__VA_ARGS__);                                            \
  } while (0)

#define VALID_OR_RETURN(value)                                                 \
  do {                                                                         \
    if (HasFatalErrors()) {                                                    \
      LogFatalErrors();                                                        \
      return value;                                                            \
    }                                                                          \
  } while (0)
#define VALID_OR_RETURN_CHECK_TYPE(type, value)                                \
  do {                                                                         \
    if (HasFatalErrors()) {                                                    \
      LogFatalErrors();                                                        \
      return (value);                                                          \
    }                                                                          \
    if (!type) {                                                               \
      LOG_PRINTF(GetLog(LLDBLog::Types),                                       \
                 "Input type is nullptr, bailing out.");                       \
      return (value);                                                          \
    }                                                                          \
  } while (0)

using namespace lldb;
using namespace lldb_private;

char SwiftASTContext::ID;
char SwiftASTContextForModule::ID;
char SwiftASTContextForExpressions::ID;

CompilerType lldb_private::ToCompilerType(swift::Type qual_type) {
  assert(*reinterpret_cast<const char *>(qual_type.getPointer()) != '$' &&
         "wrong type system");
  if (!qual_type)
    return {};
  auto *ast_ctx = SwiftASTContext::GetSwiftASTContext(&qual_type->getASTContext());
  if (!ast_ctx) {
    std::string m_description("::");
    LOG_PRINTF(GetLog(LLDBLog::Types), "dropped orphaned AST type");
    return {};
  }
  return {ast_ctx->weak_from_this(), qual_type.getPointer()};
}

TypePayloadSwift::TypePayloadSwift(bool is_fixed_value_buffer) {
  SetIsFixedValueBuffer(is_fixed_value_buffer);
}

CompilerType SwiftASTContext::GetCompilerType(ConstString mangled_name) {
  if (auto ts = GetTypeSystemSwiftTypeRef())
    return ts->GetTypeFromMangledTypename(mangled_name);
  return {};
}

CompilerType SwiftASTContext::GetCompilerType(swift::TypeBase *swift_type) {
  return {weak_from_this(), swift_type};
}

llvm::Expected<swift::Type>
SwiftASTContext::GetSwiftType(CompilerType compiler_type) {
  if (compiler_type.GetTypeSystem().GetSharedPointer().get() == this)
    return reinterpret_cast<swift::TypeBase *>(
        compiler_type.GetOpaqueQualType());
  return ReconstructType(compiler_type.GetMangledTypeName());
}

swift::Type SwiftASTContext::GetSwiftType(opaque_compiler_type_t opaque_type) {
  assert(opaque_type && *reinterpret_cast<const char *>(opaque_type) != '$' &&
         "wrong type system");
  return GetSwiftTypeIgnoringErrors(
      CompilerType(weak_from_this(), opaque_type));
}

swift::Type
SwiftASTContext::GetSwiftTypeIgnoringErrors(CompilerType compiler_type) {
  return llvm::expectedToStdOptional(GetSwiftType(compiler_type))
      .value_or(swift::Type());
}

swift::CanType
SwiftASTContext::GetCanonicalSwiftType(CompilerType compiler_type) {
  swift::Type swift_type = GetSwiftTypeIgnoringErrors(compiler_type);
  return swift_type ? swift_type->getCanonicalType() : swift::CanType();
}

swift::CanType
SwiftASTContext::GetCanonicalSwiftType(opaque_compiler_type_t opaque_type) {
  assert(!opaque_type || *reinterpret_cast<const char *>(opaque_type) != '$' &&
         "wrong type system");
  return GetCanonicalSwiftType(CompilerType(weak_from_this(), opaque_type));
}

ConstString SwiftASTContext::GetMangledTypeName(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(
      type, ConstString("<invalid Swift context or opaque type>"));
  return GetMangledTypeName(GetSwiftType(type).getPointer());
}

typedef lldb_private::ThreadSafeDenseMap<swift::ASTContext *, SwiftASTContext *>
    ThreadSafeSwiftASTMap;

static ThreadSafeSwiftASTMap &GetASTMap() {
  // The global destructor list will tear down all of the modules when
  // the LLDB shared library is being unloaded and this needs to live
  // beyond all of those and not be destructed before they have all
  // gone away. So we will leak this list intentionally so we can
  // avoid global destructor problems.
  static ThreadSafeSwiftASTMap *g_map_ptr = NULL;
  static std::once_flag g_once_flag;
  std::call_once(g_once_flag, []() {
    // Intentional leak.
    g_map_ptr = new ThreadSafeSwiftASTMap();
  });
  return *g_map_ptr;
}

class SwiftEnumDescriptor;

typedef std::shared_ptr<SwiftEnumDescriptor> SwiftEnumDescriptorSP;
typedef llvm::DenseMap<opaque_compiler_type_t, SwiftEnumDescriptorSP>
    EnumInfoCache;
typedef std::shared_ptr<EnumInfoCache> EnumInfoCacheSP;
typedef llvm::DenseMap<const SwiftASTContext *, EnumInfoCacheSP>
    ASTEnumInfoCacheMap;

static EnumInfoCache *GetEnumInfoCache(const SwiftASTContext *swift_ast_ctx) {
  static ASTEnumInfoCacheMap g_cache;
  static std::mutex g_mutex;
  std::lock_guard<std::mutex> locker(g_mutex);
  ASTEnumInfoCacheMap::iterator pos = g_cache.find(swift_ast_ctx);
  if (pos == g_cache.end()) {
    g_cache.insert(std::make_pair(
        swift_ast_ctx, std::shared_ptr<EnumInfoCache>(new EnumInfoCache())));
    return g_cache.find(swift_ast_ctx)->second.get();
  }
  return pos->second.get();
}

llvm::LLVMContext &SwiftASTContext::GetGlobalLLVMContext() {
  static llvm::LLVMContext s_global_context;
  return s_global_context;
}

llvm::ArrayRef<swift::VarDecl *>
SwiftASTContext::GetStoredProperties(swift::NominalTypeDecl *nominal) {
  VALID_OR_RETURN(llvm::ArrayRef<swift::VarDecl *>());

  // Check whether we already have the stored properties for this
  // nominal type.
  auto known = m_stored_properties.find(nominal);
  if (known != m_stored_properties.end())
    return known->second;

  // Collect the stored properties from the AST and put them in the
  // cache.
  auto stored_properties = nominal->getStoredProperties();
  auto &stored = m_stored_properties[nominal];
  stored = std::vector<swift::VarDecl *>(stored_properties.begin(),
                                         stored_properties.end());
  return stored;
}

class SwiftEnumDescriptor {
public:
  enum class Kind {
    Empty,      ///< No cases in this enum.
    CStyle,     ///< No cases have payloads.
    AllPayload, ///< All cases have payloads.
    Mixed,      ///< Some cases have payloads.
    Resilient   ///< A resilient enum.
  };

  struct ElementInfo {
    lldb_private::ConstString name;
    CompilerType payload_type;
    bool has_payload : 1;
    bool is_indirect : 1;
  };

  Kind GetKind() const { return m_kind; }

  ConstString GetTypeName() { return m_type_name; }

  virtual ElementInfo *
  GetElementFromData(const lldb_private::DataExtractor &data,
                     bool no_payload) = 0;

  virtual size_t GetNumElements() {
    return GetNumElementsWithPayload() + GetNumCStyleElements();
  }

  virtual size_t GetNumElementsWithPayload() = 0;

  virtual size_t GetNumCStyleElements() = 0;

  virtual ElementInfo *GetElementWithPayloadAtIndex(size_t idx) = 0;

  virtual ElementInfo *GetElementWithNoPayloadAtIndex(size_t idx) = 0;

  virtual ~SwiftEnumDescriptor() = default;

  static SwiftEnumDescriptor *CreateDescriptor(SwiftASTContext *swift_ast_ctx,
                                               swift::CanType swift_can_type,
                                               swift::EnumDecl *enum_decl);

protected:
  SwiftEnumDescriptor(swift::CanType swift_can_type,
                      swift::EnumDecl *enum_decl, SwiftEnumDescriptor::Kind k)
      : m_kind(k), m_type_name() {
    if (swift_can_type.getPointer()) {
      if (auto nominal = swift_can_type->getAnyNominal()) {
        swift::Identifier name(nominal->getName());
        if (name.get())
          m_type_name.SetCString(name.get());
      }
    }
  }

private:
  Kind m_kind;
  ConstString m_type_name;
};

class SwiftEmptyEnumDescriptor : public SwiftEnumDescriptor {
public:
  SwiftEmptyEnumDescriptor(swift::CanType swift_can_type,
                           swift::EnumDecl *enum_decl)
      : SwiftEnumDescriptor(swift_can_type, enum_decl,
                            SwiftEnumDescriptor::Kind::Empty) {}

  ElementInfo *GetElementFromData(const lldb_private::DataExtractor &data,
                                  bool no_payload) override {
    return nullptr;
  }

  size_t GetNumElementsWithPayload() override { return 0; }
  size_t GetNumCStyleElements() override { return 0; }
  ElementInfo *GetElementWithPayloadAtIndex(size_t idx) override {
    return nullptr;
  }

  ElementInfo *GetElementWithNoPayloadAtIndex(size_t idx) override {
    return nullptr;
  }

  static bool classof(const SwiftEnumDescriptor *S) {
    return S->GetKind() == SwiftEnumDescriptor::Kind::Empty;
  }
};

namespace std {
template <> struct less<swift::ClusteredBitVector> {
  bool operator()(const swift::ClusteredBitVector &lhs,
                  const swift::ClusteredBitVector &rhs) const {
    int iL = lhs.size() - 1;
    int iR = rhs.size() - 1;
    for (; iL >= 0 && iR >= 0; --iL, --iR) {
      bool bL = lhs[iL];
      bool bR = rhs[iR];
      if (bL and not bR)
        return false;
      if (bR and not bL)
        return true;
    }
    return false;
  }
};
} // namespace std

static std::string Dump(const swift::ClusteredBitVector &bit_vector) {
  std::string buffer;
  llvm::raw_string_ostream ostream(buffer);
  for (size_t i = 0; i < bit_vector.size(); i++) {
    if (bit_vector[i])
      ostream << '1';
    else
      ostream << '0';
    if ((i % 4) == 3)
      ostream << ' ';
  }
  ostream.flush();
  return buffer;
}

class SwiftCStyleEnumDescriptor : public SwiftEnumDescriptor {
  llvm::SmallString<32> m_description = {"SwiftCStyleEnumDescriptor"};

public:
  SwiftCStyleEnumDescriptor(SwiftASTContext *swift_ast_ctx,
                            swift::CanType swift_can_type,
                            swift::EnumDecl *enum_decl)
      : SwiftEnumDescriptor(swift_can_type, enum_decl,
                            SwiftEnumDescriptor::Kind::CStyle),
        m_nopayload_elems_bitmask(), m_elements(), m_element_indexes() {
    LOG_PRINTF(GetLog(LLDBLog::Types), "doing C-style enum layout for %s",
               GetTypeName().AsCString());

    swift::irgen::IRGenModule &irgen_module = swift_ast_ctx->GetIRGenModule();
    const swift::irgen::EnumImplStrategy &enum_impl_strategy =
        swift::irgen::getEnumImplStrategy(irgen_module, swift_can_type);
    llvm::ArrayRef<swift::irgen::EnumImplStrategy::Element>
        elements_with_no_payload =
            enum_impl_strategy.getElementsWithNoPayload();
    const bool has_payload = false;
    const bool is_indirect = false;
    uint64_t case_counter = 0;
    m_nopayload_elems_bitmask =
        enum_impl_strategy.getBitMaskForNoPayloadElements();

    if (enum_decl->isObjC())
      m_is_objc_enum = true;

    LOG_PRINTF(GetLog(LLDBLog::Types), "m_nopayload_elems_bitmask = %s",
               Dump(m_nopayload_elems_bitmask).c_str());

    for (auto enum_case : elements_with_no_payload) {
      ConstString case_name(enum_case.decl->getBaseIdentifier().str());
      swift::ClusteredBitVector case_value =
          enum_impl_strategy.getBitPatternForNoPayloadElement(enum_case.decl);

      LOG_PRINTF(GetLog(LLDBLog::Types), "case_name = %s, unmasked value = %s",
                 case_name.AsCString(), Dump(case_value).c_str());

      case_value &= m_nopayload_elems_bitmask;

      LOG_PRINTF(GetLog(LLDBLog::Types), "case_name = %s, masked value = %s",
                 case_name.AsCString(), Dump(case_value).c_str());

      std::unique_ptr<ElementInfo> elem_info(
          new ElementInfo{case_name, CompilerType(), has_payload, is_indirect});
      m_element_indexes.emplace(case_counter, elem_info.get());
      case_counter++;
      m_elements.emplace(case_value, std::move(elem_info));
    }
  }

  ElementInfo *GetElementFromData(const lldb_private::DataExtractor &data,
                                  bool no_payload) override {
    LOG_PRINTF(GetLog(LLDBLog::Types),
               "C-style enum - inspecting data to find enum case for type %s",
               GetTypeName().AsCString());

    swift::ClusteredBitVector current_payload;
    lldb::offset_t offset = 0;
    for (size_t idx = 0; idx < data.GetByteSize(); idx++) {
      uint64_t byte = data.GetU8(&offset);
      current_payload.add(8, byte);
    }

    LOG_PRINTF(GetLog(LLDBLog::Types), "m_nopayload_elems_bitmask        = %s",
               Dump(m_nopayload_elems_bitmask).c_str());
    LOG_PRINTF(GetLog(LLDBLog::Types), "current_payload                  = %s",
               Dump(current_payload).c_str());

    if (current_payload.size() != m_nopayload_elems_bitmask.size()) {
      LOG_PRINTF(GetLog(LLDBLog::Types),
                 "sizes don't match; getting out with an error");
      return nullptr;
    }

    // A C-Like Enum  is laid out as an integer tag with the minimal number of
    // bits to contain all of the cases. The cases are assigned tag values in
    // declaration order. e.g.
    // enum Patatino { // => LLVM i1
    // case X         // => i1 0
    // case Y         // => i1 1
    // }
    // From this we can find out the number of bits really used for the payload.
    if (!m_is_objc_enum) {
      current_payload &= m_nopayload_elems_bitmask;
      auto elem_mask =
          swift::ClusteredBitVector::getConstant(current_payload.size(), false);
      int64_t bit_count = m_elements.size() - 1;
      if (bit_count > 0 && no_payload) {
        uint64_t bit_set = 0;
        while (bit_count > 0) {
          elem_mask.setBit(bit_set);
          bit_set += 1;
          bit_count /= 2;
        }
        current_payload &= elem_mask;
      }
    }

    LOG_PRINTF(GetLog(LLDBLog::Types), "masked current_payload           = %s",
               Dump(current_payload).c_str());

    auto iter = m_elements.find(current_payload), end = m_elements.end();
    if (iter == end) {
      LOG_PRINTF(GetLog(LLDBLog::Types), "bitmask search failed");
      return nullptr;
    }
    LOG_PRINTF(GetLog(LLDBLog::Types), "bitmask search success - found case %s",
               iter->second.get()->name.AsCString());
    return iter->second.get();
  }

  size_t GetNumElementsWithPayload() override { return 0; }
  size_t GetNumCStyleElements() override { return m_elements.size(); }

  ElementInfo *GetElementWithPayloadAtIndex(size_t idx) override {
    return nullptr;
  }

  ElementInfo *GetElementWithNoPayloadAtIndex(size_t idx) override {
    if (idx >= m_element_indexes.size())
      return nullptr;
    return m_element_indexes[idx];
  }

  static bool classof(const SwiftEnumDescriptor *S) {
    return S->GetKind() == SwiftEnumDescriptor::Kind::CStyle;
  }

  virtual ~SwiftCStyleEnumDescriptor() = default;

private:
  swift::ClusteredBitVector m_nopayload_elems_bitmask;
  std::map<swift::ClusteredBitVector, std::unique_ptr<ElementInfo>> m_elements;
  std::map<uint64_t, ElementInfo *> m_element_indexes;
  bool m_is_objc_enum = false;
};

class SwiftAllPayloadEnumDescriptor : public SwiftEnumDescriptor {
  llvm::SmallString<32> m_description = {"SwiftAllPayloadEnumDescriptor"};

public:
  SwiftAllPayloadEnumDescriptor(SwiftASTContext *swift_ast_ctx,
                                swift::CanType swift_can_type,
                                swift::EnumDecl *enum_decl)
      : SwiftEnumDescriptor(swift_can_type, enum_decl,
                            SwiftEnumDescriptor::Kind::AllPayload) {
    LOG_PRINTF(GetLog(LLDBLog::Types), "doing ADT-style enum layout for %s",
               GetTypeName().AsCString());

    swift::irgen::IRGenModule &irgen_module = swift_ast_ctx->GetIRGenModule();
    const swift::irgen::EnumImplStrategy &enum_impl_strategy =
        swift::irgen::getEnumImplStrategy(irgen_module, swift_can_type);
    llvm::ArrayRef<swift::irgen::EnumImplStrategy::Element>
        elements_with_payload = enum_impl_strategy.getElementsWithPayload();
    m_tag_bits = enum_impl_strategy.getTagBitsForPayloads();

    LOG_PRINTF(GetLog(LLDBLog::Types), "tag_bits = %s",
               Dump(m_tag_bits).c_str());

    const bool has_payload = true;
    for (auto enum_case : elements_with_payload) {
      ConstString case_name(enum_case.decl->getBaseIdentifier().str());

      swift::EnumElementDecl *case_decl = enum_case.decl;
      assert(case_decl);
      auto arg_type = case_decl->getPayloadInterfaceType();
      CompilerType case_type;
      if (arg_type) {
        case_type = ToCompilerType(
            {swift_can_type->getTypeOfMember(case_decl, arg_type)
                 ->getCanonicalType()
                 .getPointer()});
      }

      const bool is_indirect =
          case_decl->isIndirect() || case_decl->getParentEnum()->isIndirect();

      LOG_PRINTF(GetLog(LLDBLog::Types),
                 "case_name = %s, type = %s, is_indirect = %s",
                 case_name.AsCString(), case_type.GetTypeName().AsCString(),
                 is_indirect ? "yes" : "no");

      std::unique_ptr<ElementInfo> elem_info(
          new ElementInfo{case_name, case_type, has_payload, is_indirect});
      m_elements.push_back(std::move(elem_info));
    }
  }

  ElementInfo *GetElementFromData(const lldb_private::DataExtractor &data,
                                  bool no_payload) override {
    LOG_PRINTF(GetLog(LLDBLog::Types),
               "ADT-style enum - inspecting data to find enum case for type %s",
               GetTypeName().AsCString());

    // No elements, just fail.
    if (m_elements.size() == 0) {
      LOG_PRINTF(GetLog(LLDBLog::Types), "enum with no cases. getting out");
      return nullptr;
    }
    // One element, so it's got to be it.
    if (m_elements.size() == 1) {
      LOG_PRINTF(GetLog(LLDBLog::Types),
                 "enum with one case. getting out easy with %s",
                 m_elements.front().get()->name.AsCString());

      return m_elements.front().get();
    }

    swift::ClusteredBitVector current_payload;
    lldb::offset_t offset = 0;
    for (size_t idx = 0; idx < data.GetByteSize(); idx++) {
      uint64_t byte = data.GetU8(&offset);
      current_payload.add(8, byte);
    }
    LOG_PRINTF(GetLog(LLDBLog::Types), "tag_bits        = %s",
               Dump(m_tag_bits).c_str());
    LOG_PRINTF(GetLog(LLDBLog::Types), "current_payload = %s",
               Dump(current_payload).c_str());

    if (current_payload.size() != m_tag_bits.size()) {
      LOG_PRINTF(GetLog(LLDBLog::Types),
                 "sizes don't match; getting out with an error");
      return nullptr;
    }

    // FIXME: this assumes the tag bits should be gathered in
    // little-endian byte order.
    size_t discriminator = 0;
    size_t power_of_2 = 1;
    for (size_t i = 0; i < m_tag_bits.size(); ++i) {
      if (m_tag_bits[i]) {
        discriminator |= current_payload[i] ? power_of_2 : 0;
        power_of_2 <<= 1;
      }
    }

    // The discriminator is too large?
    if (discriminator >= m_elements.size()) {
      LOG_PRINTF(GetLog(LLDBLog::Types),
                 "discriminator value of %" PRIu64 " too large, getting out",
                 (uint64_t)discriminator);
      return nullptr;
    } else {
      auto ptr = m_elements[discriminator].get();
      if (!ptr)
        LOG_PRINTF(GetLog(LLDBLog::Types),
                   "discriminator value of %" PRIu64
                   " acceptable, but null case matched - that's bad",
                   (uint64_t)discriminator);
      else
        LOG_PRINTF(GetLog(LLDBLog::Types),
                   "discriminator value of %" PRIu64
                   " acceptable, case %s matched",
                   (uint64_t)discriminator, ptr->name.AsCString());
      return ptr;
    }
  }

  size_t GetNumElementsWithPayload() override { return m_elements.size(); }
  size_t GetNumCStyleElements() override { return 0; }

  ElementInfo *GetElementWithPayloadAtIndex(size_t idx) override {
    if (idx >= m_elements.size())
      return nullptr;
    return m_elements[idx].get();
  }

  ElementInfo *GetElementWithNoPayloadAtIndex(size_t idx) override {
    return nullptr;
  }

  static bool classof(const SwiftEnumDescriptor *S) {
    return S->GetKind() == SwiftEnumDescriptor::Kind::AllPayload;
  }

private:
  swift::ClusteredBitVector m_tag_bits;
  std::vector<std::unique_ptr<ElementInfo>> m_elements;
};

class SwiftMixedEnumDescriptor : public SwiftEnumDescriptor {
public:
  SwiftMixedEnumDescriptor(SwiftASTContext *swift_ast_ctx,
                           swift::CanType swift_can_type,
                           swift::EnumDecl *enum_decl)
      : SwiftEnumDescriptor(swift_can_type, enum_decl,
                            SwiftEnumDescriptor::Kind::Mixed),
        m_non_payload_cases(swift_ast_ctx, swift_can_type, enum_decl),
        m_payload_cases(swift_ast_ctx, swift_can_type, enum_decl) {}

  ElementInfo *GetElementFromData(const lldb_private::DataExtractor &data,
                                  bool no_payload) override {
    ElementInfo *elem_info =
        m_non_payload_cases.GetElementFromData(data, false);
    return elem_info ? elem_info
                     : m_payload_cases.GetElementFromData(data, false);
  }

  static bool classof(const SwiftEnumDescriptor *S) {
    return S->GetKind() == SwiftEnumDescriptor::Kind::Mixed;
  }

  size_t GetNumElementsWithPayload() override {
    return m_payload_cases.GetNumElementsWithPayload();
  }

  size_t GetNumCStyleElements() override {
    return m_non_payload_cases.GetNumCStyleElements();
  }

  ElementInfo *GetElementWithPayloadAtIndex(size_t idx) override {
    return m_payload_cases.GetElementWithPayloadAtIndex(idx);
  }

  ElementInfo *GetElementWithNoPayloadAtIndex(size_t idx) override {
    return m_non_payload_cases.GetElementWithNoPayloadAtIndex(idx);
  }

private:
  SwiftCStyleEnumDescriptor m_non_payload_cases;
  SwiftAllPayloadEnumDescriptor m_payload_cases;
};

class SwiftResilientEnumDescriptor : public SwiftEnumDescriptor {
  llvm::SmallString<32> m_description = {"SwiftResilientEnumDescriptor"};

public:
  SwiftResilientEnumDescriptor(swift::CanType swift_can_type,
                               swift::EnumDecl *enum_decl)
      : SwiftEnumDescriptor(swift_can_type, enum_decl,
                            SwiftEnumDescriptor::Kind::Resilient) {
    LOG_PRINTF(GetLog(LLDBLog::Types), "doing resilient enum layout for %s",
               GetTypeName().AsCString());
  }

  ElementInfo *GetElementFromData(const lldb_private::DataExtractor &data,
                                  bool no_payload) override {
    // Not yet supported by LLDB.
    return nullptr;
  }
  size_t GetNumElementsWithPayload() override { return 0; }
  size_t GetNumCStyleElements() override { return 0; }
  ElementInfo *GetElementWithPayloadAtIndex(size_t idx) override {
    return nullptr;
  }
  ElementInfo *GetElementWithNoPayloadAtIndex(size_t idx) override {
    return nullptr;
  }
  static bool classof(const SwiftEnumDescriptor *S) {
    return S->GetKind() == SwiftEnumDescriptor::Kind::Resilient;
  }
};

SwiftEnumDescriptor *
SwiftEnumDescriptor::CreateDescriptor(SwiftASTContext *swift_ast_ctx,
                                      swift::CanType swift_can_type,
                                      swift::EnumDecl *enum_decl) {
  assert(swift_ast_ctx);
  assert(enum_decl);
  assert(swift_can_type.getPointer());
  assert(swift_ast_ctx);
  swift::irgen::IRGenModule &irgen_module = swift_ast_ctx->GetIRGenModule();
  const swift::irgen::EnumImplStrategy &enum_impl_strategy =
      swift::irgen::getEnumImplStrategy(irgen_module, swift_can_type);
  llvm::ArrayRef<swift::irgen::EnumImplStrategy::Element>
      elements_with_payload = enum_impl_strategy.getElementsWithPayload();
  llvm::ArrayRef<swift::irgen::EnumImplStrategy::Element>
      elements_with_no_payload = enum_impl_strategy.getElementsWithNoPayload();
  swift::SILType swift_sil_type = irgen_module.getLoweredType(swift_can_type);
  if (!irgen_module.getTypeInfo(swift_sil_type).isFixedSize())
    return new SwiftResilientEnumDescriptor(swift_can_type, enum_decl);
  if (elements_with_no_payload.size() == 0) {
    // Nothing with no payload.. empty or all payloads?
    if (elements_with_payload.size() == 0)
      return new SwiftEmptyEnumDescriptor(swift_can_type, enum_decl);
    return new SwiftAllPayloadEnumDescriptor(swift_ast_ctx, swift_can_type,
                                             enum_decl);
  }

  // Something with no payload.. mixed or C-style?
  if (elements_with_payload.size() == 0)
    return new SwiftCStyleEnumDescriptor(swift_ast_ctx, swift_can_type,
                                         enum_decl);
  return new SwiftMixedEnumDescriptor(swift_ast_ctx, swift_can_type, enum_decl);
}

static SwiftEnumDescriptor *
GetEnumInfoFromEnumDecl(SwiftASTContext *swift_ast_ctx,
                        swift::CanType swift_can_type,
                        swift::EnumDecl *enum_decl) {
  return SwiftEnumDescriptor::CreateDescriptor(swift_ast_ctx, swift_can_type,
                                               enum_decl);
}

SwiftEnumDescriptor *
SwiftASTContext::GetCachedEnumInfo(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, nullptr);

  EnumInfoCache *enum_info_cache = GetEnumInfoCache(this);
  EnumInfoCache::const_iterator pos = enum_info_cache->find(type);
  if (pos != enum_info_cache->end())
    return pos->second.get();

  if (IsMeaninglessWithoutDynamicResolution(type))
    return nullptr;

  SwiftEnumDescriptorSP enum_info_sp;
  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  if (auto *enum_type = swift_can_type->getAs<swift::EnumType>()) {
    enum_info_sp.reset(GetEnumInfoFromEnumDecl(this, swift_can_type,
                                               enum_type->getDecl()));
  } else if (auto *bound_enum_type =
                 swift_can_type->getAs<swift::BoundGenericEnumType>()) {
    enum_info_sp.reset(GetEnumInfoFromEnumDecl(this, swift_can_type,
                                               bound_enum_type->getDecl()));
  }

  if (enum_info_sp.get())
    enum_info_cache->insert(std::make_pair(type, enum_info_sp));
  return enum_info_sp.get();
}

namespace {
static inline bool
SwiftASTContextSupportsLanguage(lldb::LanguageType language) {
  return language == eLanguageTypeSwift;
}

static bool IsDeviceSupport(StringRef path) {
  // The old-style check, which we preserve for safety.
  if (path.contains("iOS DeviceSupport"))
    return true;

  // The new-style check, which should cover more devices.
  StringRef Developer = path.substr(path.find("Developer"));
  StringRef DeviceSupport =
    Developer.substr(Developer.find("DeviceSupport"));
  if (DeviceSupport.contains("Symbols"))
    return true;

  // Don't look in the simulator runtime frameworks either.  They
  // either duplicate what the SDK has, or for older simulators
  // conflict with them.
  if (path.contains(".simruntime/Contents/Resources/"))
    return true;

  return false;
}
} // namespace

static std::string GetClangModulesCacheProperty() {
  llvm::SmallString<128> path;
  const auto &props = ModuleList::GetGlobalModuleListProperties();
  props.GetClangModulesCachePath().GetPath(path);
  return std::string(path);
}

SwiftASTContext::ScopedDiagnostics::ScopedDiagnostics(
    swift::DiagnosticConsumer &consumer)
    : m_consumer(consumer),
      m_cursor({static_cast<StoringDiagnosticConsumer *>(&m_consumer)
                    ->m_raw_swift_diagnostics.size(),
                static_cast<StoringDiagnosticConsumer *>(&m_consumer)
                    ->m_raw_clang_diagnostics.size(),
                static_cast<StoringDiagnosticConsumer *>(&m_consumer)
                    ->m_diagnostics.size()}) {}

void SwiftASTContext::ScopedDiagnostics::PrintDiagnostics(
    DiagnosticManager &diagnostic_manager, uint32_t bufferID,
    uint32_t first_line, uint32_t last_line) const {
  auto consumer = static_cast<StoringDiagnosticConsumer *>(&m_consumer);
  consumer->PrintDiagnostics(diagnostic_manager, m_cursor, bufferID, first_line,
                             last_line);
}

std::optional<SwiftASTContext::ScopedDiagnostics::ErrorKind>
SwiftASTContext::ScopedDiagnostics::GetOptionalErrorKind() const {
  using ErrorKind = SwiftASTContext::ScopedDiagnostics::ErrorKind;
  auto &consumer = *static_cast<StoringDiagnosticConsumer *>(&m_consumer);

  if (consumer.m_raw_clang_diagnostics.size() > m_cursor.clang)
    return ErrorKind::clang;
  if (consumer.m_num_swift_errors > m_cursor.m_num_swift_errors)
    return ErrorKind::swift;

  for (size_t i = m_cursor.lldb; i < consumer.m_diagnostics.size(); ++i)
    if (consumer.m_diagnostics[i]->GetSeverity() == eSeverityError)
      return ErrorKind::swift;

  return {};
}

bool SwiftASTContext::ScopedDiagnostics::HasErrors() const {
  return GetOptionalErrorKind() != std::nullopt;
}

llvm::Error SwiftASTContext::ScopedDiagnostics::GetAllErrors() const {
  // Retrieve the error message from the DiagnosticConsumer.
  DiagnosticManager diagnostic_manager;
  PrintDiagnostics(diagnostic_manager);
  return llvm::createStringError(diagnostic_manager.GetString());
}

llvm::Error SwiftASTContext::ScopedDiagnostics::GetAsExpressionError(
    lldb::ExpressionResults result) const {
  DiagnosticManager diagnostic_manager;
  PrintDiagnostics(diagnostic_manager);
  return diagnostic_manager.GetAsError(result);
}

SwiftASTContext::ScopedDiagnostics::~ScopedDiagnostics() {
  auto &consumer = *static_cast<StoringDiagnosticConsumer *>(&m_consumer);
  auto &lldb_diags = consumer.m_diagnostics;
  auto &swift_diags = consumer.m_raw_swift_diagnostics;

  // Intentionally don't reset clang diagnostics. Fatal clang
  // diagnostics cannot be recovered from and they may need be
  // surfaced as a root cause later on.
  lldb_diags.resize(std::min(lldb_diags.size(), m_cursor.lldb));
  swift_diags.resize(std::min(swift_diags.size(), m_cursor.swift));

  assert(consumer.m_num_swift_errors >= m_cursor.m_num_swift_errors);
  consumer.m_num_swift_errors = m_cursor.m_num_swift_errors;

  consumer.m_ast_context.GetDiagnosticEngine().resetHadAnyError();
}

std::unique_ptr<SwiftASTContext::ScopedDiagnostics>
SwiftASTContext::getScopedDiagnosticConsumer() {
  auto &consumer =
      *static_cast<StoringDiagnosticConsumer *>(m_diagnostic_consumer_ap.get());
  return std::make_unique<SwiftASTContext::ScopedDiagnostics>(consumer);
}

#ifndef NDEBUG
SwiftASTContext::SwiftASTContext()
    : m_diagnostic_consumer_ap(new StoringDiagnosticConsumer(*this)) {
  llvm::dbgs() << "Initialized mock SwiftASTContext\n";
}
#endif

SwiftASTContext::SwiftASTContext(std::string description, ModuleSP module_sp,
                                 TypeSystemSwiftTypeRefSP typeref_typesystem)
    : TypeSystemSwift(), m_typeref_typesystem(typeref_typesystem),
      m_compiler_invocation_ap(new swift::CompilerInvocation()),
      m_diagnostic_consumer_ap(new StoringDiagnosticConsumer(*this)) {
  assert(
      ModuleList::GetGlobalModuleListProperties().GetSwiftEnableASTContext() &&
      "Swift AST context instantiation is disabled!");

  m_description = description;
  m_module = module_sp.get();

  // Set the clang modules cache path.
  m_compiler_invocation_ap->setClangModuleCachePath(
      GetClangModulesCacheProperty());
}

static std::string DerivePlatformPluginPath(StringRef sdk_path) {
  llvm::StringRef path = sdk_path;
  path = llvm::sys::path::parent_path(path);
  if (llvm::sys::path::filename(path) != "SDKs")
    return {};
  path = llvm::sys::path::parent_path(path);
  if (llvm::sys::path::filename(path) != "Developer")
    return {};
  path = llvm::sys::path::parent_path(path);
  if (!path.ends_with(".platform"))
    return {};
  return std::string(path) + "/usr/local/lib/swift/host/plugins";
}

void SwiftASTContext::SetCompilerInvocationLLDBOverrides() {
  swift::IRGenOptions &ir_gen_opts =
      m_compiler_invocation_ap->getIRGenOptions();
  ir_gen_opts.OutputKind = swift::IRGenOutputKind::Module;
  ir_gen_opts.UseJIT = true;
  // In the JIT we don't benefit from the indexed indirections in DWARF 5.
  ir_gen_opts.DWARFVersion = 4;
  ir_gen_opts.DebugInfoFormat = swift::IRGenDebugInfoFormat::DWARF;
  // Allow deserializing @_implementationOnly dependencies
  // to avoid crashing due to module recovery issues.
  swift::LangOptions &lang_opts = m_compiler_invocation_ap->getLangOptions();
  lang_opts.AllowDeserializingImplementationOnly = true;
  lang_opts.DebuggerSupport = true;

  // ModuleFileSharedCore::getTransitiveLoadingBehavior() has a
  // best-effort mode that is enabled when debugger support is turned
  // on that will try to import implementation-only imports of Swift
  // modules, but won't treat import failures as errors. When explicit
  // modules are on, this has the unwanted side-effect of potentially
  // triggering an implicit Clang module build if one of the internal
  // dependencies of a library was not used to build the target. It
  // can also lead to additional Swift modules being pulled in that
  // through their dependencies can lead to dependency cycles that
  // were not a problem at build time.
  lang_opts.ImportNonPublicDependencies = false;
  // When loading Swift types that conform to ObjC protocols that have
  // been renamed with NS_SWIFT_NAME the DwarfImporterDelegate will crash
  // during protocol conformance checks as the underlying type cannot be
  // found. Allowing module compilation to proceed with compiler
  // errors will prevent crashing, instead we will have empty type info
  // for the protocol conforming types.
  lang_opts.AllowModuleWithCompilerErrors = true;
  lang_opts.EnableTargetOSChecking = false;
  lang_opts.EnableModuleLoadingRemarks = true;

  // Bypass deserialization safety to allow deserializing internal details from
  // swiftmodule files.
  lang_opts.EnableDeserializationSafety = false;

  // Platform plugin path (macOS hosts only).
  swift::PluginSearchOption::ExternalPluginPath platform_plugins;
  platform_plugins.SearchPath =
      DerivePlatformPluginPath(m_compiler_invocation_ap->getSDKPath());
  if (!platform_plugins.SearchPath.empty()) {
    platform_plugins.ServerPath = GetPluginServer(platform_plugins.SearchPath);
    if (!platform_plugins.ServerPath.empty()) {
      if (FileSystem::Instance().Exists(platform_plugins.SearchPath) &&
          FileSystem::Instance().Exists(platform_plugins.ServerPath))
        m_compiler_invocation_ap->getSearchPathOptions()
            .PluginSearchOpts.push_back(platform_plugins);
    }
  }
}

SwiftASTContext::~SwiftASTContext() {
#ifndef NDEBUG
  m_ast_context_mutex.lock();
  if (swift::ASTContext *ctx = m_ast_context_ap.get())
    // A RemoteASTContext associated with this swift::ASTContext has
    // to be destroyed before the swift::ASTContext is destroyed.
    assert(!GetASTMap().Lookup(ctx) && "ast context still in global map");
  m_ast_context_mutex.unlock();
#endif
}

SwiftASTContextForModule::~SwiftASTContextForModule() {
  if (auto ctx = GetASTContext())
    GetASTMap().Erase(*ctx);
}

/// This code comes from CompilerInvocation.cpp (setRuntimeResourcePath).
static void ConfigureResourceDirs(swift::CompilerInvocation &invocation,
                                  StringRef resource_dir, llvm::Triple triple) {
  // Make sure the triple is right:
  invocation.setTargetTriple(triple.str());
  invocation.setRuntimeResourcePath(resource_dir);
}

static void ConfigureModuleCachePath(SwiftASTContext &swift_ast_sp) {
  std::string moduleCachePath =
      swift_ast_sp.GetCompilerInvocation().getClangModuleCachePath().str();
  if (!moduleCachePath.empty())
    return;

  // If the moduleCachePath is not configured, setup a default path location.
  llvm::SmallString<0> path;
  std::error_code ec =
      llvm::sys::fs::createUniqueDirectory("ModuleCache", path);
  if (!ec)
    moduleCachePath = std::string(path);
  else
    moduleCachePath = "/tmp/lldb-ModuleCache";
  swift_ast_sp.GetCompilerInvocation().setClangModuleCachePath(moduleCachePath);
}

static const char *getImportFailureString(swift::serialization::Status status) {
  switch (status) {
  case swift::serialization::Status::Valid:
    return "The module is valid.";
  case swift::serialization::Status::FormatTooOld:
    return "The module file format is too old to be used by this version of "
           "the debugger.";
  case swift::serialization::Status::FormatTooNew:
    return "The module file format is too new to be used by this version of "
           "the debugger.";
  case swift::serialization::Status::MissingDependency:
    return "The module file depends on another module that can't be loaded.";
  case swift::serialization::Status::MissingUnderlyingModule:
    return "The module file is an overlay for a Clang module, which can't be "
           "found.";
  case swift::serialization::Status::CircularDependency:
    return "The module file depends on a module that is still being loaded, "
           "i.e. there is a circular dependency.";
  case swift::serialization::Status::FailedToLoadBridgingHeader:
    return "The module file depends on a bridging header that can't be loaded.";
  case swift::serialization::Status::Malformed:
    return "The module file is malformed in some way.";
  case swift::serialization::Status::MalformedDocumentation:
    return "The module documentation file is malformed in some way.";
  case swift::serialization::Status::NameMismatch:
    return "The module file's name does not match the module it is being "
           "loaded into.";
  case swift::serialization::Status::TargetIncompatible:
    return "The module file was built for a different target platform.";
  case swift::serialization::Status::TargetTooNew:
    return "The module file was built for a target newer than the current "
           "target.";
  case swift::serialization::Status::RevisionIncompatible:
    return "The module file was built with library evolution enabled by a "
           "different version of the compiler.";
  case swift::serialization::Status::SDKMismatch:
    return "The module file was built with a different SDK version.";
  case swift::serialization::Status::ChannelIncompatible:
    return "The distribution channel doesn't match.";
  }
}

static void printASTValidationError(
    llvm::raw_ostream &errs,
    const swift::serialization::ValidationInfo &ast_info,
    const swift::serialization::ExtendedValidationInfo &ext_ast_info,
    StringRef module_name, StringRef module_buf, bool invalid_name,
    bool invalid_size) {
  const char *error = getImportFailureString(ast_info.status);
  errs << "AST validation error";
  if (!invalid_name)
    errs << " in \"" << ast_info.name << '"';
  errs << ": ";
  // Instead of printing the generic Status::Malformed error, be specific.
  if (invalid_size)
    errs << "The serialized module is corrupted.";
  else if (invalid_name)
    errs << "The serialized module has an invalid name.";
  else
    errs << error;

  llvm::SmallString<1> m_description;
  Log *log(GetLog(LLDBLog::Types));
  LLDB_LOG(log, R"(Unable to load Swift AST for module "{0}" from library "{1}".
  {2}
  - targetTriple: {3}
  - shortVersion: {4}
  - bytes: {5} (module_buf bytes: {6})
  - SDK path: {7}
  - Clang Importer Options:
)",
           ast_info.name, module_name, error, ast_info.targetTriple,
           ast_info.shortVersion, ast_info.bytes, module_buf.size(),
           ext_ast_info.getSDKPath());
  for (StringRef ExtraOpt : ext_ast_info.getExtraClangImporterOptions())
    LLDB_LOG(log, "  -- {0}", ExtraOpt);
}

/// Locate the swift-plugin-server for a plugin library,
/// by converting  ${toolchain}/usr/(local)?/lib/swift/host/plugins
/// into           ${toolchain}/usr/bin/swift-plugin-server
/// FIXME: move this to Host, it may be platform-specific.
std::string
SwiftASTContext::GetPluginServer(llvm::StringRef plugin_library_path) {
  llvm::StringRef path = plugin_library_path;
  if (llvm::sys::path::filename(path) != "plugins") {
    // Strip off a plugin name.
    path = llvm::sys::path::parent_path(plugin_library_path);
    if (llvm::sys::path::filename(path) != "plugins")
      return {};
  }
  path = llvm::sys::path::parent_path(path);
  if (llvm::sys::path::filename(path) != "host")
    return {};
  path = llvm::sys::path::parent_path(path);
  if (llvm::sys::path::filename(path) != "swift")
    return {};
  path = llvm::sys::path::parent_path(path);
  if (llvm::sys::path::filename(path) != "lib")
    return {};
  path = llvm::sys::path::parent_path(path);
  if (llvm::sys::path::filename(path) == "local")
    path = llvm::sys::path::parent_path(path);
  llvm::SmallString<256> server(path);
  llvm::sys::path::append(server, "bin", "swift-plugin-server");
  return std::string(server);
}

static std::string GetPluginServerForSDK(llvm::StringRef sdk_path) {
  XcodeSDK sdk(std::string(llvm::sys::path::filename(sdk_path)));
  auto server_or_err = HostInfo::FindSDKTool(sdk, "swift-plugin-server");
  if (!server_or_err)
    return "";
  return server_or_err->str();
}

namespace {
  constexpr std::array<std::string_view, 4> g_known_eplicit_module_prefixes =
       {"-fmodule-map-file=",
        "-fmodule-file=",
        "-fno-implicit-modules",
        "-fno-implicit-module-maps"};
}

/// Retrieve the serialized AST data blobs and initialize the compiler
/// invocation with the concatenated search paths from the blobs.
/// \returns true if an error was encountered.
static bool DeserializeAllCompilerFlags(swift::CompilerInvocation &invocation,
                                        llvm::StringRef module_name,
                                        llvm::StringRef module_filter,
                                        llvm::ArrayRef<StringRef> buffers,
                                        const PathMappingList &path_map,
                                        bool discover_implicit_search_paths,
                                        const std::string &m_description,
                                        llvm::raw_ostream &error,
                                        bool &got_serialized_options,
                                        bool &found_swift_modules,
                                        bool search_paths_only = false) {
  bool found_validation_errors = false;
  got_serialized_options = false;

  auto &search_path_options = invocation.getSearchPathOptions();
  auto get_override_server = [&](llvm::StringRef plugin_path) -> std::string {
    // If the user manually specified an override plugin server for a
    // specific path prefix, return it.
    Args plugin_servers =
        Target::GetGlobalProperties().GetSwiftPluginServerForPath();
    for (auto &arg: plugin_servers) {
      auto key_value = arg.ref().split('=');
      llvm::SmallString<0> ignore(plugin_path);
      if (llvm::sys::path::replace_path_prefix(ignore, key_value.first, {}))
        return key_value.second.str();
    }
    return {};
  };
  auto get_plugin_server = [&](llvm::StringRef plugin,
                               std::function<std::string(void)> fallback) {
    // Search for a manual override first, then try fallback.
    std::string server = get_override_server(plugin);
    if (server.empty())
      server = fallback();
    if (server.empty()) {
      LOG_VERBOSE_PRINTF(GetLog(LLDBLog::Types),
                         "Could not find swift-plugin-server for %s",
                         plugin.str().c_str());
      return std::string();
    }
    if (!FileSystem::Instance().Exists(server)) {
      HEALTH_LOG_PRINTF("Swift plugin server does not exist: %s",
                        server.c_str());
      server.clear();
    }
    return server; 
  };

#define INIT_SEARCH_PATH_SET(TYPE, ACCESSOR, NAME, KEY)                        \
  std::vector<TYPE> NAME;                                                      \
  llvm::StringSet<> known_##NAME;                                              \
  for (auto &path : search_path_options.ACCESSOR) {                            \
    NAME.push_back(path);                                                      \
    known_##NAME.insert(path KEY);                                             \
  }

  INIT_SEARCH_PATH_SET(swift::SearchPathOptions::SearchPath,
                       getImportSearchPaths(), import_search_paths,
                       .Path);
  INIT_SEARCH_PATH_SET(swift::SearchPathOptions::SearchPath,
                       getFrameworkSearchPaths(), framework_search_paths,
                       .Path);

  std::vector<swift::PluginSearchOption> plugin_search_options;
  llvm::StringSet<> known_plugin_search_paths;
  llvm::StringSet<> known_external_plugin_search_paths;
  llvm::StringSet<> known_compiler_plugin_library_paths;
  llvm::StringSet<> known_compiler_plugin_executable_paths;
  llvm::StringSet<> known_resolved_plugin_configs;
  for (auto &elem : search_path_options.PluginSearchOpts) {
    plugin_search_options.push_back(elem);

#define INSERT_PATH_SET(SET, TYPE, KEY)                                        \
  if (auto *opt = elem.dyn_cast<swift::PluginSearchOption::TYPE>()) {          \
    known_plugin_search_paths.insert(opt->KEY);                                \
    continue;                                                                  \
  }
    INSERT_PATH_SET(known_plugin_search_paths, PluginPath, SearchPath);
    INSERT_PATH_SET(known_external_plugin_search_paths, ExternalPluginPath,
                    SearchPath);
    INSERT_PATH_SET(known_compiler_plugin_library_paths, LoadPluginLibrary,
                    LibraryPath);
    INSERT_PATH_SET(known_compiler_plugin_executable_paths,
                    LoadPluginExecutable, ExecutablePath);
  }

  // An AST section consists of one or more AST modules, optionally
  // with headers. Iterate over all AST modules.
  for (auto buf : buffers) {
    swift::serialization::ValidationInfo info;
    for (; !buf.empty(); buf = buf.substr(info.bytes)) {
      llvm::SmallVector<swift::serialization::SearchPath> searchPaths;
      swift::serialization::ExtendedValidationInfo extended_validation_info;
      info = swift::serialization::validateSerializedAST(
          buf,
          /*requiredSDK*/ StringRef(), &extended_validation_info,
          /*dependencies*/ nullptr, &searchPaths);
      bool invalid_ast = info.status != swift::serialization::Status::Valid;
      bool invalid_size = (info.bytes == 0) || (info.bytes > buf.size());
      bool invalid_name = info.name.empty();
      if (invalid_ast || invalid_size || invalid_name) {
        // Validation errors are diagnosed, but not fatal for the context.
        found_validation_errors = true;
        printASTValidationError(error, info, extended_validation_info,
                                module_name, buf, invalid_name, invalid_size);
        // If there's a size error, quit the loop early, otherwise try the next.
        if (invalid_size)
          break;
        continue;
      }

      // Skip over this module if filtering and the filter matches.
      if (!module_filter.empty() && module_filter != info.name) {
        LOG_PRINTF(GetLog(LLDBLog::Types),
                   "Skipping compiler options from %s, filter=%s",
                   extended_validation_info.getModuleABIName().str().c_str(),
                   module_filter.str().c_str());
        continue;
      }

      found_swift_modules = true;
      StringRef moduleData = buf.substr(0, info.bytes);

      auto remap = [&](const std::string &path) {
        ConstString remapped;
        if (path_map.RemapPath(ConstString(path), remapped))
          return remapped.GetStringRef().str();
        return path;
      };

      /// Initialize the compiler invocation with it the search paths from a
      /// serialized AST.
      auto deserializeCompilerFlags = [&](swift::CompilerInvocation &invocation) {
        auto result = invocation.loadFromSerializedAST(moduleData);
        if (result != swift::serialization::Status::Valid) {
          error << "Could not deserialize " << info.name << ":\n"
                << getImportFailureString(result) << "\n";
          return false;
        }
        if (discover_implicit_search_paths) {
          for (auto &searchPath : searchPaths) {
            std::string path = remap(searchPath.Path);
            if (!searchPath.IsFramework) {
              swift::SearchPathOptions::SearchPath
                  import_search_path(path, searchPath.IsSystem);
              if (known_import_search_paths.insert(path).second)
                import_search_paths.push_back(import_search_path);
            } else {
              swift::SearchPathOptions::SearchPath
                  framework_search_path(path, searchPath.IsSystem);
              if (known_framework_search_paths.insert(path).second)
                framework_search_paths.push_back(framework_search_path);
            }
          }
        }
        auto exists = [&](llvm::StringRef path) {
          if (FileSystem::Instance().Exists(path))
            return true;
          HEALTH_LOG_PRINTF("Ignoring missing Swift plugin at path: %s",
                            path.str().c_str());
          return false;
        };

        for (auto &opt : extended_validation_info.getPluginSearchOptions()) {
          switch (opt.first) {
          case swift::PluginSearchOption::Kind::PluginPath: {
            StringRef path = opt.second;
            // System plugins shipping with the compiler.
            // Rewrite them to go through an ABI-compatible swift-plugin-server.
            if (known_plugin_search_paths.insert(path).second) {
              if (known_external_plugin_search_paths.insert(path).second) {
                std::string server = get_plugin_server(path, [&]() {
                  std::string server = SwiftASTContext::GetPluginServer(path);
                  if (!server.empty() && !FileSystem::Instance().Exists(server))
                    server.clear();
                  return server;
                });
                if (server.empty())
                  continue;
                if (exists(path))
                  plugin_search_options.emplace_back(
                      swift::PluginSearchOption::ExternalPluginPath{path.str(),
                                                                    server});
              }
            }
            continue;
          }
          case swift::PluginSearchOption::Kind::ExternalPluginPath: {
            // Sandboxed system plugins shipping with some compiler.
            // Keep the original plugin server path, it needs to be ABI
            // compatible with the version of SwiftSyntax used by the plugin.
            auto plugin_server = opt.second.split('#');
            llvm::StringRef plugin = plugin_server.first;
            std::string server = get_plugin_server(
                plugin, [&]() { return plugin_server.second.str(); });
            if (server.empty())
              continue;
            if (known_external_plugin_search_paths.insert(plugin).second)
              if (exists(plugin))
                plugin_search_options.emplace_back(
                    swift::PluginSearchOption::ExternalPluginPath{plugin.str(),
                                                                  server});
            continue;
          }
          case swift::PluginSearchOption::Kind::LoadPluginLibrary: {
            // Compiler plugin libraries.
            StringRef dylib = opt.second;
            if (known_compiler_plugin_library_paths.insert(dylib).second)
              if (exists(dylib)) {
                // We never want to directly load any plugins, since a crash in
                // the plugin would bring down LLDB. Here, we assume that the
                // correct plugin server for a direct compiler plugin is the one
                // from the SDK the compiler was building for. This is just a
                // heuristic.
                // This works because the Swift compiler enforces
                // '-load-plugin-library' dylibs to be named
                // libModuleName.[dylib|so|dll] just like
                // '-external-plugin-path'.
                llvm::SmallString<0> dir(dylib);
                llvm::sys::path::remove_filename(dir);
                std::string server = get_plugin_server(dir, [&]() {
                  return GetPluginServerForSDK(invocation.getSDKPath());
                });
                if (server.empty())
                  continue;

                plugin_search_options.emplace_back(
                    swift::PluginSearchOption::ExternalPluginPath{
                        dir.str().str(), server});
              }
            continue;
          }
          case swift::PluginSearchOption::Kind::LoadPluginExecutable: {
            // Compiler plugin executables.
            auto plugin_modules = opt.second.split('#');
            llvm::StringRef plugin = plugin_modules.first;
            llvm::StringRef modules_list = plugin_modules.second;
            llvm::SmallVector<llvm::StringRef, 0> modules;
            modules_list.split(modules, ",");
            std::vector<std::string> modules_vec;
            for (auto m : modules)
              modules_vec.push_back(m.str());
            if (known_compiler_plugin_executable_paths.insert(opt.second)
                    .second)
              if (exists(plugin))
                plugin_search_options.emplace_back(
                    swift::PluginSearchOption::LoadPluginExecutable{
                        plugin.str(), modules_vec});
            continue;
          }
          case swift::PluginSearchOption::Kind::ResolvedPluginConfig: {
            // Resolved plugin config.
            StringRef lib_path;
            StringRef exe_path;
            StringRef modules_list;
            std::tie(lib_path, exe_path) = opt.second.split('#');
            std::tie(exe_path, modules_list) = exe_path.split('#');
            std::vector<std::string> modules_vec;
            for (auto name : llvm::split(modules_list, ','))
              modules_vec.emplace_back(name);
            if (known_resolved_plugin_configs.insert(opt.second).second)
              if ((lib_path.empty() || exists(lib_path)) &&
                  (exe_path.empty() || exists(exe_path)))
                plugin_search_options.emplace_back(
                    swift::PluginSearchOption::ResolvedPluginConfig{
                        lib_path.str(), exe_path.str(),
                        std::move(modules_vec)});
            continue;
          }
          }
          llvm_unreachable("unhandled plugin search option kind");
        }

        return true;
      };

      if (search_paths_only) {
        swift::CompilerInvocation fresh_invocation;
        got_serialized_options |= deserializeCompilerFlags(fresh_invocation);
      } else {
        got_serialized_options |= deserializeCompilerFlags(invocation);
        LOG_PRINTF(
            GetLog(LLDBLog::Types), "SDK path from module \"%s\" was \"%s\".",
            info.name.str().c_str(), invocation.getSDKPath().str().c_str());
        // We will deduce a matching SDK path from DWARF later.
        invocation.setSDKPath("");
      }
    }
  }

  search_path_options.setImportSearchPaths(std::move(import_search_paths));
  search_path_options.setFrameworkSearchPaths(
      std::move(framework_search_paths));
  // (All PluginSearchPaths were rewritten to be external.)
  search_path_options.PluginSearchOpts = plugin_search_options;
  return found_validation_errors;
}

/// Return whether this module contains any serialized Swift ASTs.
bool HasSwiftModules(Module &module) {
  auto ast_file_datas = module.GetASTData(eLanguageTypeSwift);
  return !ast_file_datas.empty();
}

namespace {

/// Calls arg.consume_front(<options>) and returns true on success.
/// \param prefix contains the consumed prefix.
bool ConsumeIncludeOption(StringRef &arg, StringRef &prefix) {
  static StringRef options[] = {"-I",
                                "-F",
                                "-fmodule-map-file=",
                                "-iquote",
                                "-idirafter",
                                "-iframeworkwithsysroot",
                                "-iframework",
                                "-iprefix",
                                "-iwithprefixbefore",
                                "-iwithprefix",
                                "-isystemafter",
                                "-isystem",
                                "-isysroot",
                                "-ivfsoverlay"};

  for (StringRef &option : options)
    if (arg.consume_front(option)) {
      prefix = option;
      return true;
    }

  // We need special handling for -fmodule-file as we need to
  // split on the final = after the module name.
  if (arg.starts_with("-fmodule-file=")) {
    prefix = arg.substr(0, arg.rfind("=") + 1);
    arg.consume_front(prefix);
    return true;
  }

  return false;
}

std::array<StringRef, 2> macro_flags = { "-D", "-U" };
std::array<StringRef, 6> multi_arg_flags = {
    "-D", "-U", "-I", "-F", "-working-directory", "--target="};
std::array<StringRef, 6> args_to_unique = {
    "-D", "-U", "-I", "-F", "-fmodule-file=", "-fmodule-map-file="};

bool IsMultiArgClangFlag(StringRef arg) {
  for (auto &flag : multi_arg_flags)
    if (flag == arg)
      return true;
  return false;
}

bool IsMacroDefinition(StringRef arg) {
  for (auto &flag : macro_flags)
    if (arg.starts_with(flag))
      return true;
  return false;
}

bool ShouldUnique(StringRef arg) {
  for (auto &flag : args_to_unique)
    if (arg.starts_with(flag))
      return true;
  return false;
}
} // namespace

// static
void SwiftASTContext::AddExtraClangArgs(const std::vector<std::string> &source,
                                        std::vector<std::string> &dest) {
  llvm::StringSet<> unique_flags;
  for (auto &arg : dest)
    unique_flags.insert(arg);

  llvm::SmallString<128> cur_working_dir;
  llvm::SmallString<128> clang_argument;

  auto match_explicit_build_option = [](StringRef arg) {
    for (const auto &option : g_known_eplicit_module_prefixes)
      if (arg.starts_with(option))
        return true;
    return false;
  };
  bool has_explicit_builds_enabled =
      llvm::find(source, "-fno-implicit-modules") != source.end();

  for (const std::string &arg : source) {
    if (arg == "-triple") {
      // Clang extra args are for driver flags. Replace the frontend `-triple`
      // flag with a `--target=` driver flag.
      clang_argument = "--target=";
    } else {
      // Join multi-arg options for uniquing.
      clang_argument += arg;
    }

    if (IsMultiArgClangFlag(clang_argument))
      continue;

    auto clear_arg = llvm::make_scope_exit([&] { clang_argument.clear(); });

    // Consume any -working-directory arguments.
    StringRef cwd(clang_argument);
    if (cwd.consume_front("-working-directory")) {
      cur_working_dir = cwd;
      continue;
    }
    // Drop -Werror; it would only cause trouble in the debugger.
    if (clang_argument.starts_with("-Werror"))
      continue;

    // Drop `--`. This might be coming from the user-provided setting
    // swift-extra-clang-flags (where users sometimes think a -- is necessary
    // to separate the flags from the settings name). `--` indicates to Clang
    // that all following arguments are file names instead of flags, so this
    // should never be passed to Clang (which would otherwise either crash or
    // cause Clang to look for files with the name '-Wflag-name`).
    if (clang_argument == "--")
      continue;

    if (clang_argument.empty())
      continue;

    // In case of explicit modules, for now fallback to implicit
    // module loading.
    // TODO: Incorporate loading explicit module dependencies to
    // speedup dependency resolution.
    if (!Target::GetGlobalProperties().GetSwiftAllowExplicitModules())
      if (has_explicit_builds_enabled &&
          match_explicit_build_option(clang_argument))
        continue;

    // Otherwise add the argument to the list.
    if (!IsMacroDefinition(clang_argument))
      ApplyWorkingDir(clang_argument, cur_working_dir);

    std::string clang_arg_str = clang_argument.str().str();
    if (!ShouldUnique(clang_argument) || !unique_flags.count(clang_arg_str)) {
      dest.push_back(clang_arg_str);
      unique_flags.insert(clang_arg_str);
    }
  }
}

namespace {

bool HasNonexistentExplicitModule(const std::vector<std::string> &args) {
  for (const std::string &arg : args) {
    StringRef value = arg;
    if (!value.consume_front("-fmodule-file="))
      continue;
    StringRef path = value;
    size_t eq = value.find('=');
    // The value that follows is in one of two formats:
    //   1. ModuleName=ModulePath
    //   2. ModulePath
    if (eq != std::string::npos)
      // The value appears to be in ModuleName=ModulePath forat.
      path = value.drop_front(eq + 1);
    // Check both path and value. This is to handle paths containing '='.
    if (!llvm::sys::fs::exists(path) && !llvm::sys::fs::exists(value)) {
      std::string m_description;
      HEALTH_LOG_PRINTF("Nonexistent explicit module file %s", arg.data());
      return true;
    }
  }
  return false;
}

void RemoveExplicitModules(std::vector<std::string> &args) {
  llvm::erase_if(args, [](const std::string &arg) {
    if (arg == "-fno-implicit-modules" || arg == "-fno-implicit-module-maps")
      return true;
    StringRef s = arg;
    if (s.starts_with("-fmodule-file=") || s.starts_with("-fmodule-map-file="))
      return true;

    return false;
  });
}

} // namespace

/// LLDB wrapper for `clang::driver::applyOverrideOptions` (which implements
/// CCC_OVERRIDE_OPTIONS behavior).
static void applyOverrideOptions(std::vector<std::string> &args,
                                 llvm::StringRef overrideOpts) {
  if (overrideOpts.empty())
    return;

  // Convert input args to the type required by applyOverrideOptions.
  llvm::SmallVector<const char *, 64> raw_args;
  // Add placeholder clang executable, which applyOverrideOptions expects to be
  // the first argument.
  raw_args.push_back("clang");
  for (const std::string &arg : args)
    raw_args.push_back(arg.data());

  /// LLVM stream backed by a callback. This is used to redirect
  /// applyOverrideOptions logging to LLDB.
  struct CallbackStream : public llvm::raw_ostream {
    using callback_t = std::function<void(const char *, size_t)>;
    callback_t m_callback;
    uint64_t m_pos = 0;

    CallbackStream(callback_t callback) : m_callback(callback) {}
    ~CallbackStream() override { flush(); }

    void write_impl(const char *Ptr, size_t Size) override {
      m_callback(Ptr, Size);
      m_pos += Size;
    }

    uint64_t current_pos() const override { return m_pos; }
  };

  // Perform the override operations.
  llvm::StringSet<> savedStrings;
  auto *log = GetLog(LLDBLog::Types);
  CallbackStream log_stream{[log](const char *Ptr, size_t Size) {
    if (!log)
      return;
    if (Ptr[Size] == '\n')
      // Skip the newline because LLDB logging writes a newline.
      Size--;
    log->PutString({Ptr, Size});
  }};

  clang::driver::applyOverrideOptions(raw_args, overrideOpts.data(),
                                      savedStrings, "CCC_OVERRIDE_OPTIONS",
                                      &log_stream);

  // Delete the placeholder "clang" executable argument.
  raw_args.erase(raw_args.begin());

  // Copy `raw_args` into a new args vector.
  std::vector<std::string> new_args;
  for (const char *arg : raw_args)
    new_args.emplace_back(arg);

  // Only now that `raw_args` has been copied into `new_args`, can `args` be
  // overwritten. This is because `args` owns the data pointed to by `raw_args`.
  args = new_args;
}

/// Set PCM validation. This needs to happen before ClangImporter is created but
/// after m_has_explicit_modules has been initialized.
void SwiftASTContext::ConfigureModuleValidation(
    std::vector<std::string> &extra_args) {
  // Read the setting.
  AutoBool validate_pcm_setting = AutoBool::Auto;
  TargetSP target_sp = GetTargetWP().lock();
  if (target_sp)
    validate_pcm_setting = target_sp->GetSwiftPCMValidation();

  // If validation is explicitly enabled, honor it.
  bool validate_pcm = validate_pcm_setting == AutoBool::True;
  if (validate_pcm_setting == AutoBool::Auto) {
    // Disable validation for explicit modules.
    validate_pcm = m_has_explicit_modules ? false : true;
    // Enable validation in asserts builds.
#ifndef NDEBUG
    validate_pcm = true;
#endif
  }

  if (!validate_pcm)
    extra_args.push_back("-fno-modules-check-relocated");
  LOG_PRINTF(GetLog(LLDBLog::Types), "PCM validation is %s",
             validate_pcm ? "disabled" : "enabled");
}

void SwiftASTContext::AddExtraClangArgs(
    const std::vector<std::string> &ExtraArgs,
    const std::vector<std::pair<std::string, bool>> module_search_paths,
    const std::vector<std::pair<std::string, bool>> framework_search_paths,
    StringRef overrideOpts) {
  swift::ClangImporterOptions &importer_options = GetClangImporterOptions();
  auto defer = llvm::make_scope_exit([&]() {
    // Detect explicitly-built modules.
    m_has_explicit_modules =
        llvm::any_of(importer_options.ExtraArgs, [](const std::string &arg) {
          return StringRef(arg).starts_with("-fmodule-file=");
        });
    ConfigureModuleValidation(importer_options.ExtraArgs);
  });

  if (ExtraArgs.empty())
    return;

  // Detect cc1 flags.  When DirectClangCC1ModuleBuild is on then the
  // clang arguments in the serialized invocation are clang cc1 flags,
  // which are very specific to one compiler version and cannot
  // be merged with driver options.
  bool fresh_invocation = importer_options.ExtraArgs.empty();
  bool invocation_direct_cc1 = ExtraArgs.front() == "-cc1";

  // If it is not a fresh invocation, make sure the cc1 option matches.
  if (!fresh_invocation &&
      (importer_options.DirectClangCC1ModuleBuild != invocation_direct_cc1))
    AddDiagnostic(
        eSeverityWarning,
        "Mixing and matching of driver and cc1 Clang options detected");

  importer_options.DirectClangCC1ModuleBuild = invocation_direct_cc1;

  // If using direct cc1 flags, compute the arguments and return.
  if (importer_options.DirectClangCC1ModuleBuild) {
    if (!fresh_invocation)
      importer_options.ExtraArgs.clear();
    AddExtraClangCC1Args(ExtraArgs, module_search_paths, framework_search_paths,
                         importer_options.ExtraArgs);
    applyOverrideOptions(importer_options.ExtraArgs, overrideOpts);
    return;
  }

  AddExtraClangArgs(ExtraArgs, importer_options.ExtraArgs);
  applyOverrideOptions(importer_options.ExtraArgs, overrideOpts);
  if (HasNonexistentExplicitModule(importer_options.ExtraArgs))
    RemoveExplicitModules(importer_options.ExtraArgs);
}

void SwiftASTContext::AddExtraClangCC1Args(
    const std::vector<std::string> &source,
    const std::vector<std::pair<std::string, bool>> module_search_paths,
    const std::vector<std::pair<std::string, bool>> framework_search_paths,
    std::vector<std::string> &dest) {
  clang::CompilerInvocation invocation;
  std::vector<std::string> default_paths = {"/usr/include",
                                            "/user/local/include"};
  llvm::SmallVector<const char *> clangArgs;
  clangArgs.reserve(source.size() + module_search_paths.size() * 2 +
                    framework_search_paths.size() * 2 +
                    default_paths.size() * 2);
  llvm::for_each(source, [&](const std::string &Arg) {
    // Workaround for the extra driver argument embedded in the swiftmodule by
    // some swift compiler version. It always starts with `--target=` and it is
    // not a valid cc1 option.
    if (!StringRef(Arg).starts_with("--target="))
      clangArgs.push_back(Arg.c_str());
  });
  // Append some search paths from swift invocation so lldb can import
  // additional clang modules when doing type reconstruction.
  for (auto &path : module_search_paths) {
    clangArgs.push_back("-I");
    clangArgs.push_back(path.first.c_str());
  }
  for (auto &path : default_paths) {
    llvm::SmallString<128> search_path(GetPlatformSDKPath());
    llvm::sys::path::append(search_path, path);
    path = std::string(search_path);
  }
  for (auto &path : default_paths) {
    clangArgs.push_back("-I");
    clangArgs.push_back(path.c_str());
  }
  for (auto &path : framework_search_paths) {
    clangArgs.push_back("-F");
    clangArgs.push_back(path.first.c_str());
  }

  std::string diags;
  llvm::raw_string_ostream os(diags);
  clang::DiagnosticOptions diagOpts;
  clang::DiagnosticsEngine clangDiags(
      new clang::DiagnosticIDs(), diagOpts,
      new clang::TextDiagnosticPrinter(os, diagOpts));

  if (!clang::CompilerInvocation::CreateFromArgs(invocation, clangArgs,
                                                 clangDiags)) {
    // If cc1 arguments failed to parse, report diagnostics and return
    // immediately.
    AddDiagnostic(eSeverityError, diags);
    // Disable direct-cc1 build as fallback.
    GetClangImporterOptions().DirectClangCC1ModuleBuild = false;
    return;
  }

  // Clear module cache key and other CAS options to load modules from disk
  // directly.
  invocation.getFrontendOpts().ModuleCacheKeys.clear();
  invocation.getCASOpts() = clang::CASOptions();

  // Ignore CAS info inside modules when loading.
  invocation.getFrontendOpts().ModuleLoadIgnoreCAS = true;

  // Add options to allow clang importer to do implicit module build.
  invocation.getLangOpts().ImplicitModules = true;
  invocation.getHeaderSearchOpts().ImplicitModuleMaps = true;
  invocation.getHeaderSearchOpts().ModuleCachePath =
      GetCompilerInvocation().getClangModuleCachePath().str();

  // Remove non-existing modules in a systematic way.
  auto CheckFileExists = [&](const std::string &file) -> bool {
    if (llvm::sys::fs::exists(file))
      return true;
    std::string warn;
    llvm::raw_string_ostream(warn)
        << "Nonexistent explicit module file " << file;
    AddDiagnostic(eSeverityWarning, warn);
    return false;
  };
  for (auto it = invocation.getHeaderSearchOpts().PrebuiltModuleFiles.begin();
       it != invocation.getHeaderSearchOpts().PrebuiltModuleFiles.end();) {
    if (!CheckFileExists(it->second))
      it = invocation.getHeaderSearchOpts().PrebuiltModuleFiles.erase(it);
    else
      ++it;
  }
  invocation.getFrontendOpts().ModuleFiles.erase(
      llvm::remove_if(invocation.getFrontendOpts().ModuleFiles,
                      [&](const auto &mod) { return !CheckFileExists(mod); }),
      invocation.getFrontendOpts().ModuleFiles.end());

  invocation.generateCC1CommandLine(
      [&](const llvm::Twine &arg) { dest.push_back(arg.str()); });

  // If cc1 arguments are parsed and generated correctly, set explicitly-built
  // module since only explicit module build can use direct cc1 mode.
  m_has_explicit_modules = true;
  ConfigureModuleValidation(dest);
  return;
}

void SwiftASTContext::AddUserClangArgs(TargetProperties &props) {
  Args args(props.GetSwiftExtraClangFlags());
  if (args.empty())
    return;

  std::vector<std::string> user_clang_flags;
  for (const auto &arg : args.entries())  {
    if (arg.ref() == "--")
      continue;
    user_clang_flags.push_back(arg.ref().str());
  }
  if (GetClangImporterOptions().DirectClangCC1ModuleBuild) {
    llvm::append_range(GetClangImporterOptions().ExtraArgs, user_clang_flags);
  } else {
    AddExtraClangArgs(user_clang_flags, {}, {});
  }
}

/// Turn relative paths in clang options into absolute paths based on
/// \c cur_working_dir.
void SwiftASTContext::ApplyWorkingDir(
    llvm::SmallVectorImpl<char> &clang_argument, StringRef cur_working_dir) {
  StringRef arg = StringRef(clang_argument.data(), clang_argument.size());
  StringRef prefix;
  if (ConsumeIncludeOption(arg, prefix)) {
    // Ignore the option part of a double-arg include option.
    if (arg.empty())
      return;
  } else if (arg.starts_with("-")) {
    // Assume this is a compiler arg and not a path starting with "-".
    return;
  }
  // There is most probably a path in arg now.
  if (!llvm::sys::path::is_relative(arg))
    return;

  llvm::SmallString<128> joined_path;
  llvm::sys::path::append(joined_path, cur_working_dir, arg);
  llvm::sys::path::remove_dots(joined_path);
  // remove_dots can return an empty string if given a . or chain of ./.
  if (joined_path.empty())
    joined_path = ".";

  clang_argument.resize(prefix.size());
  clang_argument.append(joined_path.begin(), joined_path.end());
}

void SwiftASTContext::ApplyDiagnosticOptions() {
  const auto &opts = GetCompilerInvocation().getDiagnosticOptions();
  GetDiagnosticEngine().setPrintDiagnosticNamesMode(opts.PrintDiagnosticNames);

  if (!opts.DiagnosticDocumentationPath.empty())
    GetDiagnosticEngine().setDiagnosticDocumentationPath(
        opts.DiagnosticDocumentationPath);

  if (!opts.LocalizationCode.empty() && !opts.LocalizationPath.empty())
    GetDiagnosticEngine().setLocalization(opts.LocalizationCode,
                                          opts.LocalizationPath);
}

void SwiftASTContext::RemapClangImporterOptions(
    const PathMappingList &path_map) {
  auto &options = GetClangImporterOptions();
  ConstString remapped;
  if (path_map.RemapPath(ConstString(options.BridgingHeader), remapped)) {
    LOG_PRINTF(GetLog(LLDBLog::Types), "remapped %s -> %s",
               options.BridgingHeader.c_str(), remapped.GetCString());
    options.BridgingHeader = remapped.GetCString();
  }

  // Previous argument was the dash-option of an option pair.
  bool remap_next = false;
  for (auto &arg_string : options.ExtraArgs) {
    StringRef prefix;
    StringRef arg = arg_string;

    if (remap_next)
      remap_next = false;
    else if (ConsumeIncludeOption(arg, prefix)) {
      if (arg.empty()) {
        // Option pair.
        remap_next = true;
        continue;
      }
      // Single-arg include option with prefix.
    } else {
      // Not a recognized option.
      continue;
    }

    if (path_map.RemapPath(ConstString(arg), remapped)) {
      LOG_PRINTF(GetLog(LLDBLog::Types), "remapped %s -> %s%s",
                 arg.str().c_str(), prefix.str().c_str(),
                 remapped.GetCString());
      arg_string = prefix.str() + remapped.GetCString();
    }
  }
}

void SwiftASTContext::FilterClangImporterOptions(
    std::vector<std::string> &extra_args, SwiftASTContext *ctx) {
  // The direct cc1 mode do not need any extra audit.
  if (ctx && ctx->GetClangImporterOptions().DirectClangCC1ModuleBuild)
    return;

  std::string ivfs_arg;
  // Copy back a filtered version of ExtraArgs.
  std::vector<std::string> orig_args(std::move(extra_args));
  for (auto &arg : orig_args) {
    StringRef arg_sr(arg);
    // LLDB wants to control implicit/explicit modules build flags.
    if (arg_sr == "-fno-implicit-modules" ||
        arg_sr == "-fno-implicit-module-maps")
      continue;

    // The VFS options turn into fatal errors when the referenced file
    // is not found. Since the Xcode build system tends to create a
    // lot of VFS overlays by default, stat them and emit a warning if
    // the yaml file couldn't be found.
    if (arg_sr.starts_with("-ivfs")) {
      // Stash the argument.
      ivfs_arg = arg;
      continue;
    }
    if (!ivfs_arg.empty()) {
      auto clear_ivfs_arg = llvm::make_scope_exit([&] { ivfs_arg.clear(); });
      if (!FileSystem::Instance().Exists(arg)) {
        if (ctx) {
          std::string error;
          llvm::raw_string_ostream(error)
              << "Ignoring missing VFS file: " << arg
              << "\nThis is the likely root cause for any subsequent compiler "
                 "errors.";
          ctx->AddDiagnostic(eSeverityWarning, error);
        }
        continue;
      }
      // Keep it.
      extra_args.push_back(ivfs_arg);
    }

    extra_args.push_back(std::move(arg));
  }
}

/// Retrieve the .dSYM bundle for \p module.
static std::optional<StringRef> GetDSYMBundle(Module &module) {
  auto sym_file = module.GetSymbolFile();
  if (!sym_file)
    return {};

  auto obj_file = sym_file->GetObjectFile();
  if (!obj_file)
    return {};

  StringRef dir = obj_file->GetFileSpec().GetDirectory().GetStringRef();
  auto it = llvm::sys::path::rbegin(dir);
  auto end = llvm::sys::path::rend(dir);
  if (it == end)
    return {};
  if (*it != "DWARF")
    return {};
  if (++it == end)
    return {};
  if (*it != "Resources")
    return {};
  if (++it == end)
    return {};
  if (*it != "Contents")
    return {};
  StringRef sep = llvm::sys::path::get_separator();
  StringRef dsym = dir.take_front(it - end - sep.size());
  if (llvm::sys::path::extension(dsym) != ".dSYM")
    return {};
  return dsym;
}

static std::string GetSDKPath(std::string m_description, XcodeSDK sdk) {
  auto sdk_path_or_err = HostInfo::GetSDKRoot(HostInfo::SDKOptions{sdk});
  if (!sdk_path_or_err) {
    Debugger::ReportError("Error while searching for SDK: " +
                          toString(sdk_path_or_err.takeError()));
    HEALTH_LOG_PRINTF("Error while searching for SDK (XcodeSDK: %s)",
                      sdk.GetString().str().c_str());
    return {};
  }

  std::string sdk_path = sdk_path_or_err->str();
  // GetSDKRoot reports no SDK as an empty string.
  if (sdk_path.empty()) {
    std::string sdk_spec = sdk.GetString().str();
    Debugger::ReportError("LLDB couldn't find an SDK for \"" + sdk_spec + "\".");
    HEALTH_LOG_PRINTF("Could not find an SDK for \"%s\". Try to verify that "
                      "\"xcrun --show-sdk-path --sdk %s\" works.",
                      sdk_spec.c_str(), sdk_spec.c_str());
    return {};
  }
  LOG_PRINTF(GetLog(LLDBLog::Types), "Host SDK path: \"%s\" (XcodeSDK: %s)",
             sdk_path.c_str(), sdk.GetString().str().c_str());
  return sdk_path;
}

/// Force parsing of the CUs to extract the SDK info.
static std::string GetSDKPathFromDebugInfo(std::string m_description,
                                           Module &module) {
  auto platform_sp = Platform::GetHostPlatform();
  if (!platform_sp)
    return {};
  auto sdk_or_err = platform_sp->GetSDKPathFromDebugInfo(module);
  if (!sdk_or_err) {
    Debugger::ReportError("Error while parsing SDK path from debug info: " +
                          toString(sdk_or_err.takeError()));
    return {};
  }

  auto [sdk, found_mismatch] = std::move(*sdk_or_err);

  if (found_mismatch)
    HEALTH_LOG_PRINTF(
        "Unsupported mixing of public and internal SDKs in "
        "'%s'. Mixed use of SDKs indicates use of different "
        "toolchains, which is not supported.",
        module.GetFileSpec().GetFilename().AsCString("<unknown module>"));

  return GetSDKPath(m_description, std::move(sdk));
}

static std::vector<llvm::StringRef>
GetASTBuffersFromModule(const std::string &m_description,
                        llvm::ArrayRef<lldb::DataBufferSP> ast_file_datas,
                        std::string &module_name) {
  LOG_PRINTF(GetLog(LLDBLog::Types), "Found %d AST file data entries in %s.",
             (int)ast_file_datas.size(), module_name.c_str());
  std::vector<llvm::StringRef> buffers;
  for (auto &data : ast_file_datas)
    if (data)
      buffers.push_back(
          StringRef((const char *)data->GetBytes(), data->GetByteSize()));
  return buffers;
}

/// Detect whether a Swift module was "imported" by DWARFImporter.
/// All this *really* means is that it couldn't be loaded through any
/// other mechanism.
static bool IsDWARFImported(const swift::ModuleDecl &module) {
  return llvm::any_of(module.getFiles(), [](const swift::FileUnit *file_unit) {
    return (file_unit->getKind() == swift::FileUnitKind::DWARFModule);
  });
}

/// Detect whether this is a proper Swift module.
static bool IsSerializedAST(const swift::ModuleDecl &module) {
  return llvm::any_of(module.getFiles(), [](const swift::FileUnit *file_unit) {
    return (file_unit->getKind() == swift::FileUnitKind::SerializedAST);
  });
}

/// Scan a newly added lldb::Module for Swift modules and report any errors in
/// its module SwiftASTContext to Target.
static void
ProcessModule(Module &module, std::string m_description,
              bool discover_implicit_search_paths, bool use_all_compiler_flags,
              bool is_main_executable, StringRef module_filter,
              llvm::Triple triple,
              std::vector<swift::PluginSearchOption> &plugin_search_options,
              std::vector<std::pair<std::string, bool>> &module_search_paths,
              std::vector<std::pair<std::string, bool>> &framework_search_paths,
              std::vector<std::string> &extra_clang_args,
              std::string &error) {
  {
    llvm::raw_string_ostream ss(m_description);
    ss << "::ProcessModule(" << '"';
    module.GetDescription(ss, eDescriptionLevelBrief);
    ss << '"' << ')';
  }

  const FileSpec &module_file = module.GetFileSpec();
  std::string module_path = module_file.GetPath();

  // Add the containing framework to the framework search path.
  // Don't do that if this is the executable module, since it
  // might be buried in some framework that we don't care about.
  if (use_all_compiler_flags && !is_main_executable) {
    size_t framework_offset = module_path.rfind(".framework/");

    if (framework_offset != std::string::npos) {
      // Sometimes the version of the framework that got loaded has been
      // stripped and in that case, adding it to the framework search
      // path will just short-cut a clang search that might otherwise
      // find the needed headers. So don't add these paths.
      std::string framework_path = module_path.substr(0, framework_offset);
      framework_path.append(".framework");
      FileSpec path_spec(framework_path);
      FileSystem::Instance().Resolve(path_spec);
      FileSpec headers_spec = path_spec.CopyByAppendingPathComponent("Headers");
      bool add_it = false;
      if (FileSystem::Instance().Exists(headers_spec))
        add_it = true;
      if (!add_it) {
        FileSpec module_spec =
            path_spec.CopyByAppendingPathComponent("Modules");
        if (FileSystem::Instance().Exists(module_spec))
          add_it = true;
      }

      if (!add_it) {
        LOG_PRINTF(GetLog(LLDBLog::Types),
                   "rejecting framework path \"%s\" as it has no \"Headers\" "
                   "or \"Modules\" subdirectories.",
                   framework_path.c_str());
      }

      if (add_it) {
        while (framework_offset && (module_path[framework_offset] != '/'))
          framework_offset--;

        if (module_path[framework_offset] == '/') {
          // framework_offset now points to the '/';

          std::string parent_path = module_path.substr(0, framework_offset);
          StringRef p(parent_path);

          // Never add framework paths pointing into the system. These
          // modules must be imported from the SDK instead.
          if (!p.starts_with("/System/Library") && !IsDeviceSupport(p) &&
              !p.starts_with(
                  "/Library/Apple/System/Library/PrivateFrameworks") &&
              !p.starts_with("/System/iOSSupport/System/Library/Frameworks")) {
            LOG_PRINTF(GetLog(LLDBLog::Types),
                       "adding framework path \"%s\"/.. .",
                       framework_path.c_str());
            framework_search_paths.push_back(
                {std::move(parent_path), /*system*/ false});
          }
        }
      }
    }
  }

  // Skip images without a serialized Swift AST.
  if (!HasSwiftModules(module))
    return;

  // Load search path options from the module.
  if (!use_all_compiler_flags && !is_main_executable)
    return;

  // Add Swift interfaces in the .dSYM at the end of the search paths.
  // .swiftmodules win over .swiftinterfaces, when they are loaded
  // directly from the .swift_ast section.
  //
  // FIXME: Since these paths end up in the scratch context, we would
  //        need a mechanism to ensure that and newer versions (in the
  //        library evolution sense, not the date on disk) win over
  //        older versions of the same .swiftinterface.
  if (auto dsym = GetDSYMBundle(module)) {
    llvm::SmallString<256> path(*dsym);
    StringRef arch = llvm::Triple::getArchTypeName(triple.getArch());
    llvm::sys::path::append(path, "Contents", "Resources", "Swift", arch);
    bool exists = false;
    llvm::sys::fs::is_directory(path, exists);
    if (exists)
      module_search_paths.push_back({std::string(path), /*system*/ false});
  }

  // Create a one-off CompilerInvocation as a place to load the
  // deserialized search path options into.
  SymbolFile *sym_file = module.GetSymbolFile();
  if (!sym_file)
    return;
  bool found_swift_modules = false;
  bool got_serialized_options = false;
  llvm::raw_string_ostream errs(error);
  swift::CompilerInvocation invocation;
  auto ast_file_datas = module.GetASTData(eLanguageTypeSwift);
  std::string module_name = module.GetSpecificationDescription();
  std::vector<llvm::StringRef> buffers =
      GetASTBuffersFromModule(m_description, ast_file_datas, module_name);

  // If no N_AST symbols exist, this is not an error.
  if (!buffers.empty())
    if (DeserializeAllCompilerFlags(
            invocation, module_name, module_filter, buffers,
            module.GetSourceMappingList(), discover_implicit_search_paths,
            m_description, errs, got_serialized_options, found_swift_modules)) {
      // TODO: After removing DeserializeAllCompilerFlags from
      //       CreateInstance(per-Module), errs will need to be
      //       collected here and surfaced.
    }

  // Copy the interesting deserialized flags to the out parameters.
  const auto &opts = invocation.getSearchPathOptions();
  plugin_search_options.insert(plugin_search_options.end(),
                               opts.PluginSearchOpts.begin(),
                               opts.PluginSearchOpts.end());
  for (auto path : opts.getImportSearchPaths())
    module_search_paths.push_back({path.Path, path.IsSystem});
  for (auto path : opts.getFrameworkSearchPaths())
    framework_search_paths.push_back({path.Path, path.IsSystem});
  auto &clang_opts = invocation.getClangImporterOptions().ExtraArgs;
  // If the args embedded are cc1 args, they are not compatible with existing
  // setting. Clear the previous args.
  if (!clang_opts.empty() && clang_opts.front() == "-cc1")
    extra_clang_args.clear();

  for (const std::string &arg : clang_opts) {
    extra_clang_args.push_back(arg);
    LOG_VERBOSE_PRINTF(GetLog(LLDBLog::Types), "adding Clang argument \"%s\".",
                       arg.c_str());
  }
}

lldb::TypeSystemSP
SwiftASTContext::CreateInstance(lldb::LanguageType language, Module &module,
                                TypeSystemSwiftTypeRef &typeref_typesystem) {
  TargetSP target = typeref_typesystem.GetTargetWP().lock();
  if (!SwiftASTContextSupportsLanguage(language))
    return lldb::TypeSystemSP();

  if (!ModuleList::GetGlobalModuleListProperties()
           .GetSwiftEnableASTContext())
    return lldb::TypeSystemSP();

  std::string m_description;
  {
    llvm::raw_string_ostream ss(m_description);
    ss << "SwiftASTContext";
    ss << "ForModule";
    ss << '(' << '"';
    module.GetDescription(ss, eDescriptionLevelBrief);
    ss << '"' << ')';
  }
  LLDB_SCOPED_TIMERF("%s::CreateInstance", m_description.c_str());
  std::vector<std::pair<std::string, bool>> module_search_paths;
  std::vector<std::pair<std::string, bool>> framework_search_paths;

  LOG_PRINTF(GetLog(LLDBLog::Types), "(Module)");

  auto logError = [&](const char *message) {
    HEALTH_LOG_PRINTF("Failed to create module context - %s", message);
  };

  ArchSpec arch = module.GetArchitecture();
  if (!arch.IsValid()) {
    logError("invalid module architecture");
    return {};
  }

  ObjectFile *objfile = module.GetObjectFile();
  if (!objfile) {
    logError("no object file for module");
    return {};
  }

  ArchSpec object_arch = objfile->GetArchitecture();
  if (!object_arch.IsValid()) {
    logError("invalid objfile architecture");
    return {};
  }

  llvm::Triple triple = GetSwiftFriendlyTriple(arch.GetTriple());
  if (triple.getOS() == llvm::Triple::UnknownOS) {
    // cl_kernels are the only binaries that don't have an
    // LC_MIN_VERSION_xxx load command. This avoids a Swift assertion.

#if defined(__APPLE__)
    switch (triple.getArch()) {
    default:
      triple.setOS(llvm::Triple::MacOSX);
      break;
    case llvm::Triple::arm:
    case llvm::Triple::armeb:
    case llvm::Triple::aarch64:
    case llvm::Triple::aarch64_be:
      triple.setOS(llvm::Triple::IOS);
      break;
    }

#else
    // Not an elegant hack on OS X, not an elegant hack elsewhere.
    // But we shouldn't be claiming things are Mac binaries when they
    // are not.
    triple.setOS(HostInfo::GetArchitecture().GetTriple().getOS());
#endif
  }

  // If there is a target this may be a fallback scratch context.
  std::shared_ptr<SwiftASTContext> swift_ast_sp(
      static_cast<SwiftASTContext *>(new SwiftASTContextForModule(
          m_description, module.shared_from_this(),
          typeref_typesystem.GetTypeSystemSwiftTypeRef())));
  bool suppress_config_log = false;
  auto defer_log =
      llvm::make_scope_exit([swift_ast_sp, &suppress_config_log] {
        // To avoid spamming the log with useless info, we don't log the
        // configuration if everything went fine and the current module
        // doesn't have any Swift contents (i.e., the shared cache dylibs).
        if (!suppress_config_log)
          swift_ast_sp->LogConfiguration();
      });

  // This is a module AST context, mark it as such.
  swift_ast_sp->m_is_scratch_context = false;
  swift_ast_sp->GetLanguageOptions().EnableAccessControl = false;
  swift_ast_sp->GetLanguageOptions().EnableCXXInterop =
      module.IsSwiftCxxInteropEnabled();

  if (module.IsEmbeddedSwift())
    swift_ast_sp->GetLanguageOptions().enableFeature(swift::Feature::Embedded);

  bool found_swift_modules = false;
  SymbolFile *sym_file = module.GetSymbolFile();

  if (sym_file) {
    bool got_serialized_options = false;
    llvm::SmallString<0> error;
    llvm::raw_svector_ostream errs(error);
    // Implicit search paths will be discovered by ValidateSectionModules().
    bool discover_implicit_search_paths = false;
    auto ast_file_datas = module.GetASTData(eLanguageTypeSwift);
    std::string module_name = module.GetSpecificationDescription();
    StringRef module_filter;
    std::vector<llvm::StringRef> buffers =
        GetASTBuffersFromModule(m_description, ast_file_datas, module_name);

    // If no N_AST symbols exist, this is not an error.
    if (!buffers.empty())
      if (DeserializeAllCompilerFlags(
              swift_ast_sp->GetCompilerInvocation(), module_name, module_filter,
              buffers, module.GetSourceMappingList(),
              discover_implicit_search_paths, m_description, errs,
              got_serialized_options, found_swift_modules)) {
        // Validation errors are not fatal for the context.
        swift_ast_sp->AddDiagnostic(eSeverityError, errs.str());
      }

    llvm::StringRef serialized_triple =
        swift_ast_sp->GetCompilerInvocation().getTargetTriple();
    if (!serialized_triple.empty()) {
      LOG_PRINTF(GetLog(LLDBLog::Types),
                 "Serialized/default triple would have been %s.",
                 serialized_triple.str().c_str());
    }

    // SDK path setup.
    //
    // This step is skipped for modules that don't have any Swift
    // debug info. (We assume that a module without a .swift_ast
    // section has no debuggable Swift code). This skips looking
    // through all the shared cache dylibs when they don't have debug
    // info.
    if (found_swift_modules) {
      llvm::StringRef serialized_sdk_path =
          swift_ast_sp->GetCompilerInvocation().getSDKPath();
      if (serialized_sdk_path.empty())
        LOG_PRINTF(GetLog(LLDBLog::Types), "No serialized SDK path.");
      else
        LOG_PRINTF(GetLog(LLDBLog::Types), "Serialized SDK path is %s.",
                   serialized_sdk_path.str().c_str());

      std::string sdk_path = GetSDKPathFromDebugInfo(m_description, module);
      if (FileSystem::Instance().Exists(sdk_path)) {
        // Note that this is not final. InitializeSearchPathOptions()
        // will set the SDK path based on the triple if this fails.
        swift_ast_sp->SetPlatformSDKPath(sdk_path);
        swift_ast_sp->GetCompilerInvocation().setSDKPath(sdk_path);
      }
    }
  }

  // The serialized triple is the triple of the last binary
  // __swiftast section that was processed. Instead of relying on
  // the section contents order, we overwrite the triple in the
  // CompilerInvocation with the triple recovered from the binary.
  swift_ast_sp->SetTriple(triple, &module);

  std::string resource_dir =
      HostInfo::GetSwiftResourceDir(triple, swift_ast_sp->GetPlatformSDKPath());
  ConfigureResourceDirs(swift_ast_sp->GetCompilerInvocation(), resource_dir,
                        triple);
  ConfigureModuleCachePath(*swift_ast_sp);

  swift_ast_sp->SetCompilerInvocationLLDBOverrides();

  // Collect search paths before importing modules.
  const bool discover_implicit_search_paths = false;
  const bool use_all_compiler_flags = false;
  const bool is_target_module = true;

  std::string error;
  StringRef module_filter;
  std::vector<swift::PluginSearchOption> plugin_search_options;
  std::vector<std::string> extra_clang_args = swift_ast_sp->GetClangArguments();
  swift_ast_sp->GetClangImporterOptions().ExtraArgs.clear();

  ProcessModule(module, m_description, discover_implicit_search_paths,
                use_all_compiler_flags, is_target_module, module_filter, triple,
                plugin_search_options, module_search_paths,
                framework_search_paths, extra_clang_args, error);
  if (!error.empty())
    swift_ast_sp->AddDiagnostic(eSeverityError, error);

  // Apply the working directory to all relative paths.
  StringRef overrideOpts = target ? target->GetSwiftClangOverrideOptions() : "";
  swift_ast_sp->AddExtraClangArgs(extra_clang_args, module_search_paths,
                                  framework_search_paths, overrideOpts);
  if (target)
    swift_ast_sp->AddUserClangArgs(*target);
  else
    swift_ast_sp->AddUserClangArgs(Target::GetGlobalProperties());

  // Apply source path remappings found in the module's dSYM.
  swift_ast_sp->RemapClangImporterOptions(module.GetSourceMappingList());

  // Apply source path remappings found in the target settings.
  if (target)
    swift_ast_sp->RemapClangImporterOptions(target->GetSourcePathMap());
  swift_ast_sp->FilterClangImporterOptions(
      swift_ast_sp->GetClangImporterOptions().ExtraArgs, swift_ast_sp.get());

  swift_ast_sp->InitializeSearchPathOptions(module_search_paths,
                                            framework_search_paths);
  if (!swift_ast_sp->GetClangImporter()) {
    LOG_PRINTF(
        GetLog(LLDBLog::Types),
        "(\"%s\") returning NULL - couldn't create a ClangImporter",
        module.GetFileSpec().GetFilename().AsCString("<unknown module>"));
    return {};
  }

  if (swift_ast_sp->HasFatalErrors()) {
    logError(swift_ast_sp->GetFatalErrors().AsCString());
    return {};
  }

  {
    const bool can_create = true;

    // Report progress on module importing by using a callback function in
    // swift::ASTContext
    auto ast_context = swift_ast_sp->GetASTContext();
    Progress progress("Importing Swift standard library");
    ast_context->SetPreModuleImportCallback(
        [&progress](llvm::StringRef module_name,
                    swift::ASTContext::ModuleImportKind kind) {
          progress.Increment(1, module_name.str());
        });

    // Clear the callback function on scope exit to prevent an out-of-scope
    // access of the progress local variable
    auto on_exit = llvm::make_scope_exit([&]() {
      ast_context->SetPreModuleImportCallback(
          [](llvm::StringRef module_name,
             swift::ASTContext::ModuleImportKind kind) {});
    });

    swift::ModuleDecl *stdlib =
        ast_context->getStdlibModule(can_create);
    swift_ast_sp->m_post_first_import = true;
    if (!stdlib || IsDWARFImported(*stdlib)) {
      logError("couldn't load the Swift stdlib");
      return {};
    }
  }

  // Import serialized modules.
  std::vector<std::string> module_names;
  swift_ast_sp->RegisterSectionModules(module, module_names);
  if (!module_names.size()) {
    // This dylib has no Swift contents; logging the configuration is pointless.
    suppress_config_log = true;
  } else {
    swift_ast_sp->ImportSectionModules(module, module_names);
    if (GetLog(LLDBLog::Types)) {
      std::lock_guard<std::recursive_mutex> locker(g_log_mutex);
      LOG_PRINTF(
          GetLog(LLDBLog::Types), "((Module*)%p, \"%s\") = %p",
          static_cast<void *>(&module),
          module.GetFileSpec().GetFilename().AsCString("<unknown module>"),
          static_cast<void *>(swift_ast_sp.get()));
    }
  }

  if (swift_ast_sp->HasFatalErrors()) {
    logError(swift_ast_sp->GetFatalErrors().AsCString());
    return {};
  }

  return swift_ast_sp;
}

static bool IsUnitTestExecutable(lldb_private::Module &module) {
  static ConstString s_xctest("xctest");
  static ConstString s_XCTRunner("XCTRunner");
  ConstString executable_name = module.GetFileSpec().GetFilename();
  return (executable_name == s_xctest || executable_name == s_XCTRunner);
}

static lldb::ModuleSP GetUnitTestModule(lldb_private::ModuleList &modules) {
  ConstString test_bundle_executable;

  for (size_t mi = 0, num_images = modules.GetSize(); mi != num_images; ++mi) {
    ModuleSP module_sp = modules.GetModuleAtIndex(mi);

    std::string module_path = module_sp->GetFileSpec().GetPath();

    const char deep_substr[] = ".xctest/Contents/";
    size_t pos = module_path.rfind(deep_substr);
    if (pos == std::string::npos) {
      const char flat_substr[] = ".xctest/";
      pos = module_path.rfind(flat_substr);

      if (pos == std::string::npos) {
        continue;
      } else {
        module_path.erase(pos + strlen(flat_substr));
      }
    } else {
      module_path.erase(pos + strlen(deep_substr));
    }

    if (!test_bundle_executable) {
      module_path.append("Info.plist");

      ApplePropertyList info_plist(module_path.c_str());

      std::string cf_bundle_executable;
      if (info_plist.GetValueAsString("CFBundleExecutable",
                                      cf_bundle_executable)) {
        test_bundle_executable = ConstString(cf_bundle_executable);
      } else {
        return ModuleSP();
      }
    }

    if (test_bundle_executable &&
        module_sp->GetFileSpec().GetFilename() == test_bundle_executable) {
      return module_sp;
    }
  }

  return ModuleSP();
}

lldb::TypeSystemSP SwiftASTContext::CreateInstance(
    const SymbolContext &sc, TypeSystemSwiftTypeRef &typeref_typesystem,
    bool repl, bool playground, const char *extra_options) {
  bool for_expressions =
      llvm::isa<TypeSystemSwiftTypeRefForExpressions>(&typeref_typesystem);
  // REPL requires an expression type system.
  assert(!repl || for_expressions);
  if (repl && !for_expressions)
    return {};

  if (!ModuleList::GetGlobalModuleListProperties()
           .GetSwiftEnableASTContext())
    return lldb::TypeSystemSP();

  CompileUnit *cu = sc.comp_unit;
  const char *key = TypeSystemSwiftTypeRef::DeriveKeyFor(sc);
  bool swift_context = cu && cu->GetLanguage() == eLanguageTypeSwift;
  std::string m_description;
  {
    StreamString ss;
    ss << "SwiftASTContext";
    if (for_expressions)
      ss << "ForExpressions";
    ss << "(module: " << '"' << key << "\", " << "cu: " << '"';
    if (cu && swift_context)
      ss << cu->GetPrimaryFile().GetFilename();
    else
      ss << "*";
    ss << '"' << ')';
    m_description = ss.GetString();
  }

  LLDB_SCOPED_TIMERF("%s::CreateInstance", m_description.c_str());

  // This function can either create an expression/scratch/repl context,
  // or a SwiftAST fallback context for a TypeSystemSwiftTyperef.
  // - SwiftASTContexForExpressions: target=non-null, module=null.
  // -              SwiftASTContext: target=null,     module=non-null.
  ModuleSP module_sp = sc.module_sp;
  TargetSP target_sp = typeref_typesystem.GetTargetWP().lock();

  // Make an AST but don't set the triple yet. We need to
  // try and detect if we have a iOS simulator.
  std::shared_ptr<SwiftASTContext> swift_ast_sp;
  if (for_expressions) {
    // Expression context.
    if (!target_sp) {
      LOG_PRINTF(GetLog(LLDBLog::Types), "No target for expression typesystem");
      return {};
    }
    swift_ast_sp.reset(new SwiftASTContextForExpressions(
        m_description, module_sp,
        typeref_typesystem.GetTypeSystemSwiftTypeRef()));
    // This is a scratch AST context, mark it as such.
    swift_ast_sp->m_is_scratch_context = true;
    auto &lang_opts = swift_ast_sp->GetLanguageOptions();
    lang_opts.EnableCXXInterop = ShouldEnableCXXInterop(cu);
    if (ShouldEnableEmbeddedSwift(cu))
      lang_opts.enableFeature(swift::Feature::Embedded);
  } else {
    // Typesystem fallback context.
    if (!module_sp) {
      LOG_PRINTF(GetLog(LLDBLog::Types), "No module for fallback typesystem");
      return {};
    }
    swift_ast_sp.reset(
        static_cast<SwiftASTContext *>(new SwiftASTContextForModule(
            m_description, module_sp,
            typeref_typesystem.GetTypeSystemSwiftTypeRef())));
    // This is a module AST context, mark it as such.
    swift_ast_sp->m_is_scratch_context = false;
    auto &lang_opts = swift_ast_sp->GetLanguageOptions();
    lang_opts.EnableAccessControl = false;
    lang_opts.EnableCXXInterop = ShouldEnableCXXInterop(cu);
    if (ShouldEnableEmbeddedSwift(cu))
      lang_opts.enableFeature(swift::Feature::Embedded);
  }
  auto defer_log = llvm::make_scope_exit([swift_ast_sp, repl, playground] {
    swift_ast_sp->LogConfiguration(repl, playground);
  });

  LOG_PRINTF(GetLog(LLDBLog::Types), "(Target)");
  auto logError = [&](const char *message) {
    LOG_PRINTF(GetLog(LLDBLog::Types), "Failed to create scratch context - %s",
               message);
    if (target_sp)
      if (StreamSP errs_sp = target_sp->GetDebugger().GetAsyncErrorStream())
        errs_sp->Printf("Cannot create Swift scratch context (%s)", message);
  };

  ArchSpec arch;
  if (target_sp)
    arch = target_sp->GetArchitecture();
  else if (module_sp)
    arch = module_sp->GetArchitecture();

  if (!arch.IsValid()) {
    logError("invalid target architecture");
    return {};
  }

  bool sdk_path_override = false;
  ModuleList module_module;
  if (!target_sp)
    module_module.Append(module_sp);
  // Leave modules empty if not in a Swift context to avoid a fragile
  // and expensive scan through all images. Unless this is a Playground, which
  // has a non-Swift executable, and user code in a framework.
  ModuleList &modules = (target_sp && (swift_context || playground))
                            ? target_sp->GetImages()
                            : module_module;
  const size_t num_images = modules.GetSize();

  // Set the SDK path prior to doing search paths.  Otherwise when we
  // create search path options we put in the wrong SDK path.
  FileSpec target_sdk_spec = target_sp ? target_sp->GetSDKPath() : FileSpec();
  if (target_sdk_spec && FileSystem::Instance().Exists(target_sdk_spec)) {
    swift_ast_sp->SetPlatformSDKPath(target_sdk_spec.GetPath());
    LOG_PRINTF(GetLog(LLDBLog::Types), "Using target SDK override: %s",
               target_sdk_spec.GetPath().c_str());
    sdk_path_override = true;
  }

  // Get the precise SDK from the symbol context.
  std::optional<XcodeSDK> sdk;
  if (cu)
    if (auto platform_sp = Platform::GetHostPlatform()) {
      auto sdk_or_err = platform_sp->GetSDKPathFromDebugInfo(*cu);
      if (!sdk_or_err)
        Debugger::ReportError("Error while parsing SDK path from debug info: " +
                              toString(sdk_or_err.takeError()));
      else {
        sdk = *sdk_or_err;
        LOG_PRINTF(GetLog(LLDBLog::Types), "Using precise SDK: %s",
                   sdk->GetString().str().c_str());
      }
    }

  // Derive the triple next.

  // First, prime the compiler with the options from the main executable:
  bool got_serialized_options = false;
  ModuleSP exe_module_sp =
      target_sp ? target_sp->GetExecutableModule() : ModuleSP();

  // If we're debugging a testsuite, then treat the main test bundle
  // as the executable.
  if (exe_module_sp && IsUnitTestExecutable(*exe_module_sp)) {
    ModuleSP unit_test_module = GetUnitTestModule(modules);

    if (unit_test_module) {
      exe_module_sp = unit_test_module;
    }
  }

  {
    auto get_executable_triple = [&]() -> llvm::Triple {
      if (!exe_module_sp)
        return {};
      return exe_module_sp->GetArchitecture().GetTriple();
    };

    ArchSpec module_arch;
    llvm::Triple module_triple;
    if (sc.module_sp) {
      module_arch = sc.module_sp->GetArchitecture();
      module_triple = module_arch.GetTriple();
      LOG_PRINTF(GetLog(LLDBLog::Types), "Module triple: \"%s\"",
                 module_triple.str().c_str());
    }

    ArchSpec target_arch;
    llvm::Triple target_triple;
    if (target_sp) {
      target_arch = target_sp->GetArchitecture();
      target_triple = target_arch.GetTriple();
      LOG_PRINTF(GetLog(LLDBLog::Types), "Target triple: \"%s\"",
                 target_triple.str().c_str());
    }

    ArchSpec preferred_arch;
    llvm::Triple preferred_triple;
    if (repl) {
      LOG_PRINTF(GetLog(LLDBLog::Types), "REPL: prefer target triple.");
      preferred_arch = target_arch;
      preferred_triple = target_triple;
    } else if (!sdk_path_override && !sdk && target_arch) {
      LOG_PRINTF(GetLog(LLDBLog::Types),
                 "No Swift debug info: prefer target triple.");
      if (!target_arch.IsCompatibleMatch(module_arch))
        HEALTH_LOG_PRINTF(
            "SwiftASTContext requested for a non-Swift translation unit. Using "
            "target triple \"%s\", which is not compatible with this "
            "translation unit's triple \"%s\". Expressions may behave "
            "unexpectedly because of this.",
            target_triple.str().c_str(), module_triple.str().c_str());
      preferred_arch = target_arch;
      preferred_triple = target_triple;
    } else if (module_arch &&
               (!target_arch || module_arch.IsFullySpecifiedTriple())) {
      LOG_PRINTF(GetLog(LLDBLog::Types),
                 "Prefer module triple.");
      preferred_arch = module_arch;
      preferred_triple = module_triple;
    } else {
      LOG_PRINTF(GetLog(LLDBLog::Types),
                 "No viable alternatives: Prefer target triple.");
      // When no viable module triple, fallback to the target triple.
      preferred_arch = target_arch;
      preferred_triple = target_triple;
    }

    llvm::Triple computed_triple;
    if (preferred_arch.IsFullySpecifiedTriple()) {
      // If a fully specified triple was passed in, for example
      // through CreateTargetWithFileAndTargetTriple(), prefer that.
      LOG_PRINTF(GetLog(LLDBLog::Types), "Fully specified triple %s.",
                 preferred_triple.str().c_str());
      computed_triple = preferred_triple;
    } else {
      // Underspecified means that one or more of vendor, os, or os
      // version (Darwin only) is missing.
      LOG_PRINTF(GetLog(LLDBLog::Types), "Underspecified triple %s.",
                 preferred_triple.str().c_str());
      if (!target_sp) {
        // For a per-module fallback context we can't go any further.
        computed_triple = preferred_triple;
      } else {
        llvm::VersionTuple platform_version;
        PlatformSP platform_sp = target_sp->GetPlatform();
        if (platform_sp)
          platform_version =
              platform_sp->GetOSVersion(target_sp->GetProcessSP().get());
        LOG_PRINTF(GetLog(LLDBLog::Types), "Platform version is %s",
                   platform_version.empty()
                       ? "<empty>"
                       : platform_version.getAsString().c_str());
        // Try to fill in the platform OS version. The idea behind using
        // the platform version is to let the expression evaluator mark
        // the expressions with the highest supported availability
        // attribute. Don't use the platform when an environment is
        // present, since there might be some ambiguity about the
        // plaform (e.g., ios-macabi runs on the macOS, but uses iOS
        // version numbers).
        if (!platform_version.empty() && preferred_triple.getEnvironment() ==
                                             llvm::Triple::UnknownEnvironment) {
          LOG_PRINTF(GetLog(LLDBLog::Types),
                     "Completing triple based on platform.");

          llvm::SmallString<32> buffer;
          {
            llvm::raw_svector_ostream os(buffer);
            os << preferred_triple.getArchName() << '-';
            os << preferred_triple.getVendorName() << '-';
            os << llvm::Triple::getOSTypeName(preferred_triple.getOS());
            os << platform_version.getAsString();
          }
          computed_triple = llvm::Triple(buffer.str());
        } else if (preferred_triple.getObjectFormat() == llvm::Triple::MachO) {
          LOG_PRINTF(GetLog(LLDBLog::Types),
                     "Completing triple based on main binary load commands.");
          computed_triple = get_executable_triple();
        }
      }
    }

    if (computed_triple.getOS() == llvm::Triple::MacOSX) {
      // Handle the case where an apparent macOS binary has been
      // force-loaded as a macCatalyst process. The Xcode test
      // runner works this way.
      llvm::Triple exe_triple = get_executable_triple();
      if (exe_triple.getOS() == llvm::Triple::IOS &&
          exe_triple.getEnvironment() == llvm::Triple::MacABI) {
        LOG_PRINTF(GetLog(LLDBLog::Types), "Adjusting triple to macCatalyst.");
        computed_triple.setOSAndEnvironmentName(
            exe_triple.getOSAndEnvironmentName());
      }
    }
    if (computed_triple == llvm::Triple()) {
      LOG_PRINTF(GetLog(LLDBLog::Types), "Failed to compute triple.");
      return {};
    }
    swift_ast_sp->SetTriple(sc, computed_triple);
  }

  llvm::Triple triple = swift_ast_sp->GetTriple();
  
  // Triple has been derived, find a matching SDK.
  if (!sdk_path_override) {
    XcodeSDK::Type sdk_type_for_triple = XcodeSDK::GetSDKTypeForTriple(triple);
    if (sdk && sdk->GetType() != sdk_type_for_triple) {
      HEALTH_LOG_PRINTF("Precise SDK is not compatible with triple. Ignoring.");
      XcodeSDK::Info info{sdk_type_for_triple, {}, sdk->IsAppleInternalSDK()};
      sdk = XcodeSDK(info);
    }
    if (!sdk) {
      XcodeSDK::Info info{sdk_type_for_triple, {}, false};
      sdk = XcodeSDK(info);
    }

    std::string sdk_path = GetSDKPath(m_description, *sdk);
    if (!sdk_path.empty()) {
      swift_ast_sp->SetPlatformSDKPath(sdk_path);
      LOG_PRINTF(GetLog(LLDBLog::Types), "Using SDK: %s", sdk_path.c_str());
    }
  }

  std::string resource_dir = HostInfo::GetSwiftResourceDir(
      triple, swift_ast_sp->GetPlatformSDKPath());
  ConfigureResourceDirs(swift_ast_sp->GetCompilerInvocation(), resource_dir,
                        triple);
  ConfigureModuleCachePath(*swift_ast_sp);
  {
    // ModuleInterfaceBuilder creates a separate CompilerInvocation to
    // perform implicit Clang module imports. They will always use the SDK
    // version as deployment target, even if that is in the future. To
    // avoid building modules twice, match this behavior.
    auto &ci_args = swift_ast_sp->GetClangImporterOptions().ExtraArgs;
    auto darwin_sdk_info = clang::parseDarwinSDKInfo(
        *llvm::vfs::getRealFileSystem(), swift_ast_sp->GetPlatformSDKPath());
    if (!darwin_sdk_info)
      llvm::consumeError(darwin_sdk_info.takeError());
    else if (*darwin_sdk_info) {
      auto sdk_triple = triple;
      sdk_triple.setOSName(std::string(triple.getOSTypeName(triple.getOS())) +
                           (*darwin_sdk_info)->getVersion().getAsString());
      ci_args.push_back("-target");
      ci_args.push_back(sdk_triple.str());
    }
    ci_args.push_back("-gmodules");
    ci_args.push_back("-gdwarf-4");
  }

  std::vector<swift::PluginSearchOption> plugin_search_options;
  std::vector<std::pair<std::string, bool>> module_search_paths;
  std::vector<std::pair<std::string, bool>> framework_search_paths;
  std::vector<std::string> extra_clang_args;

  bool discover_implicit_search_paths = false;
  bool use_all_compiler_flags = false;
  if (target_sp) {
    discover_implicit_search_paths =
        target_sp->GetSwiftDiscoverImplicitSearchPaths();
    use_all_compiler_flags =
        !got_serialized_options || target_sp->GetUseAllCompilerFlags();

    const bool is_system = false;

    for (const FileSpec &path : target_sp->GetSwiftModuleSearchPaths())
      module_search_paths.push_back({path.GetPath(), is_system});

    for (const FileSpec &path : target_sp->GetSwiftFrameworkSearchPaths())
      framework_search_paths.push_back({path.GetPath(), is_system});
  }
  if (module_sp) {
    std::string error;
    StringRef module_filter = TypeSystemSwiftTypeRef::GetSwiftModuleFor(sc);
    std::vector<std::string> extra_clang_args;
    // In a per-module fallback context, the module the "main" module of that
    // context.
    bool is_main_executable =
        target_sp ? (target_sp->GetExecutableModulePointer() == module_sp.get())
                  : true;
    ProcessModule(*module_sp, m_description, discover_implicit_search_paths,
                  use_all_compiler_flags, is_main_executable, module_filter,
                  triple, plugin_search_options, module_search_paths,
                  framework_search_paths, extra_clang_args, error);
    if (!error.empty())
      swift_ast_sp->AddDiagnostic(eSeverityError, error);
    StringRef override_opts =
        target_sp ? target_sp->GetSwiftClangOverrideOptions() : "";
    swift_ast_sp->AddExtraClangArgs(extra_clang_args, module_search_paths,
                                    framework_search_paths, override_opts);
  }

  // Now fold any extra options we were passed. This has to be done
  // BEFORE the ClangImporter is made by calling GetClangImporter or
  // these options will be ignored.
  if (target_sp)
    swift_ast_sp->AddUserClangArgs(*target_sp);

  if (extra_options) {
    swift::CompilerInvocation &compiler_invocation =
        swift_ast_sp->GetCompilerInvocation();
    Args extra_args(extra_options);
    llvm::ArrayRef<const char *> extra_args_ref(extra_args.GetArgumentVector(),
                                                extra_args.GetArgumentCount());
    compiler_invocation.parseArgs(extra_args_ref,
                                  swift_ast_sp->GetDiagnosticEngine());
  }

  swift_ast_sp->ApplyDiagnosticOptions();

  // Apply source path remappings found in each module's dSYM.
  for (ModuleSP module : modules.Modules())
    if (module)
      swift_ast_sp->RemapClangImporterOptions(module->GetSourceMappingList());

  // Apply source path remappings found in the target settings.
  if (target_sp)
    swift_ast_sp->RemapClangImporterOptions(target_sp->GetSourcePathMap());
  swift_ast_sp->FilterClangImporterOptions(
      swift_ast_sp->GetClangImporterOptions().ExtraArgs, swift_ast_sp.get());

  // This needs to happen once all the import paths are set, or
  // otherwise no modules will be found.
  swift_ast_sp->InitializeSearchPathOptions(module_search_paths,
                                            framework_search_paths);
  swift_ast_sp->SetCompilerInvocationLLDBOverrides();

  // Initialize the compiler plugin search paths.
  auto &opts = swift_ast_sp->GetSearchPathOptions();
  opts.PluginSearchOpts.insert(opts.PluginSearchOpts.end(),
                               plugin_search_options.begin(),
                               plugin_search_options.end());

  // Register the symbol context's module first. This makes it more
  // likely that compatible AST blobs are found first, since then the
  // local AST blobs overwrite any ones with the same import path from
  // another dylib.
  llvm::DenseSet<Module *> visited_modules;
  llvm::StringMap<ModuleSP> all_modules;
  for (size_t mi = 0; mi != num_images; ++mi) {
    auto image_sp = modules.GetModuleAtIndex(mi);
    std::string path = image_sp->GetSpecificationDescription();
    all_modules.insert({path, image_sp});
    all_modules.insert({llvm::sys::path::filename(path), image_sp});
  }
  std::vector<std::string> module_names;
  std::function<void(ModuleSP, unsigned)> scan_module =
      [&](ModuleSP cur_module_sp, unsigned indent) {
        if (!cur_module_sp ||
            !visited_modules.insert(cur_module_sp.get()).second)
          return;
        swift_ast_sp->RegisterSectionModules(*cur_module_sp, module_names);
        if (GetLog(LLDBLog::Types)) {
          std::string spacer(indent, '-');
          LOG_VERBOSE_PRINTF(
              GetLog(LLDBLog::Types), "+%s Dependency scan: %s", spacer.c_str(),
              cur_module_sp->GetSpecificationDescription().c_str());
        }
        if (auto object = cur_module_sp->GetObjectFile()) {
          FileSpecList file_list;
          object->GetDependentModules(file_list);
          for (auto &fs : file_list) {
            if (ModuleSP dependency = all_modules.lookup(fs.GetPath())) {
              scan_module(dependency, indent + 1);
            } else if (ModuleSP dependency =
                           all_modules.lookup(fs.GetFilename())) {
              scan_module(dependency, indent + 1);
            } else {
              if (GetLog(LLDBLog::Types)) {
                std::string spacer(indent, '-');
                LOG_VERBOSE_PRINTF(GetLog(LLDBLog::Types),
                                   "+%s Could not find %s in images",
                                   spacer.c_str(), fs.GetPath().c_str());
              }
            }
          }
        }
      };
  if (swift_context || playground)
    scan_module(module_sp, 0);
  for (size_t mi = 0; mi != num_images; ++mi) {
    auto image_sp = modules.GetModuleAtIndex(mi);
    if (!visited_modules.count(image_sp.get()))
      swift_ast_sp->RegisterSectionModules(*image_sp, module_names);
  }

  if (!for_expressions && module_sp) {
    auto ast_context = swift_ast_sp->GetASTContext();
    if (!ast_context) {
      logError("couldn't initialize Swift compiler");
     return {};
    }

    // Report progress on module importing by using a callback function in
    // swift::ASTContext
    Progress progress("Importing Swift standard library");
    ast_context->SetPreModuleImportCallback(
        [&progress](llvm::StringRef module_name,
                    swift::ASTContext::ModuleImportKind kind) {
          progress.Increment(1, module_name.str());
        });

    // Clear the callback function on scope exit to prevent an out-of-scope
    // access of the progress local variable
    auto on_exit = llvm::make_scope_exit([&]() {
      ast_context->SetPreModuleImportCallback(
          [](llvm::StringRef module_name,
             swift::ASTContext::ModuleImportKind kind) {});
    });

    const bool can_create = true;
    swift::ModuleDecl *stdlib =
        ast_context->getStdlibModule(can_create);
    swift_ast_sp->m_post_first_import = true;
    if (!stdlib || IsDWARFImported(*stdlib)) {
      logError("couldn't load the Swift stdlib");
      return {};
    }
  }

  // FIXME: It should be sufficient to just import the sc.comp_unit's module .
  if (!for_expressions && module_sp)
    swift_ast_sp->ImportSectionModules(*module_sp, module_names);

  if (target_sp)
    LOG_PRINTF(GetLog(LLDBLog::Types), "((Target*)%p) = %p",
               static_cast<void *>(target_sp.get()),
               static_cast<void *>(swift_ast_sp.get()));

  if (swift_ast_sp->HasFatalErrors()) {
    logError(swift_ast_sp->GetFatalErrors().AsCString());
    return {};
  }
  
  return swift_ast_sp;
}

void SwiftASTContext::EnumerateSupportedLanguages(
    std::set<lldb::LanguageType> &languages_for_types,
    std::set<lldb::LanguageType> &languages_for_expressions) {
  static std::vector<lldb::LanguageType> s_supported_languages_for_types(
      {lldb::eLanguageTypeSwift});

  static std::vector<lldb::LanguageType> s_supported_languages_for_expressions(
      {lldb::eLanguageTypeSwift});

  languages_for_types.insert(s_supported_languages_for_types.begin(),
                             s_supported_languages_for_types.end());
  languages_for_expressions.insert(
      s_supported_languages_for_expressions.begin(),
      s_supported_languages_for_expressions.end());
}

bool SwiftASTContext::SupportsLanguage(lldb::LanguageType language) {
  return SwiftASTContextSupportsLanguage(language);
}

Status SwiftASTContext::IsCompatible() { return GetFatalErrors(); }

Status SwiftASTContext::GetFatalErrors() const {
  Status error;
  if (HasFatalErrors())
    error = GetAllDiagnostics();
  return error;
}

Status SwiftASTContext::GetAllDiagnostics() const {
  Status error = m_fatal_errors.Clone();
  if (error.Success()) {
    // Retrieve the error message from the DiagnosticConsumer.
    DiagnosticManager diagnostic_manager;
    PrintDiagnostics(diagnostic_manager);
    // FIXME: Use diagnostic_manager.GetAsError()
    error = Status(diagnostic_manager.GetString());
    static_cast<StoringDiagnosticConsumer *>(m_diagnostic_consumer_ap.get())
        ->Clear();
  }
  return error;
}

void SwiftASTContext::LogFatalErrors() const {
  // Avoid spamming the health log with redundant copies of the fatal error.
  if (m_logged_fatal_error) {
    LOG_PRINTF(GetLog(LLDBLog::Types),
               "SwiftASTContext is in fatal error state, bailing out.");
    return;
  }
  if (!m_fatal_errors.Fail())
    GetFatalErrors();
  HEALTH_LOG_PRINTF(
      "SwiftASTContext is in fatal error state, bailing out: (%s).",
      m_fatal_errors.AsCString());
  m_logged_fatal_error = true;
}

swift::IRGenOptions &SwiftASTContext::GetIRGenOptions() {
  return m_compiler_invocation_ap->getIRGenOptions();
}

swift::TBDGenOptions &SwiftASTContext::GetTBDGenOptions() {
  return m_compiler_invocation_ap->getTBDGenOptions();
}

llvm::Triple SwiftASTContext::GetTriple() const {
  VALID_OR_RETURN(llvm::Triple());
  return llvm::Triple(m_compiler_invocation_ap->getTargetTriple());
}

llvm::Triple SwiftASTContext::GetSwiftFriendlyTriple(llvm::Triple triple) {
  if (triple.getVendor() != llvm::Triple::Apple) {
    // Add the GNU environment for Linux.  Although this is
    // technically incorrect, as the `*-unknown-linux` environment
    // represents the bare-metal environment, because Swift is
    // currently hosted only, we can get away with it.
    if (triple.isOSLinux()) {
      if (triple.getEnvironment() == llvm::Triple::UnknownEnvironment)
        triple.setEnvironment(llvm::Triple::GNU);
      // Contrary to what it appears, this is not a no-op.  This spells the
      // `unknown` vendor as `unknown` rather than the empty (``) string.  This
      // is required to ensure that the module triple matches exactly for Swift.
      if (triple.getVendor() == llvm::Triple::UnknownVendor)
        triple.setVendor(llvm::Triple::UnknownVendor);
    }

    // Set the vendor to `unknown` on Windows as the Swift standard library is
    // overly aggressive in matching the triple.  The vendor field is
    // initialized to `pc` by LLDB though there is no official vendor associated
    // with the open source toolchain, and so this field is rightly
    // canonicalized to `unknown`.  This allows loading of the Swift standard
    // library for the REPL.
    if (triple.isOSWindows() && triple.isWindowsMSVCEnvironment())
      triple.setVendor(llvm::Triple::UnknownVendor);
    triple.normalize();
    return triple;
  }

  StringRef arch_name = triple.getArchName();
  if (arch_name == "x86_64h")
    triple.setArchName("x86_64");
  else if (arch_name == "aarch64")
    triple.setArchName("arm64");
  else if (arch_name == "aarch64_32")
    triple.setArchName("arm64_32");
  return triple;
}

void SwiftASTContext::SetTriple(const SymbolContext &sc,
                                const llvm::Triple triple) {
  SetTriple(triple, sc.module_sp.get());
}

bool SwiftASTContext::SetTriple(const llvm::Triple triple, Module *module) {
  VALID_OR_RETURN(false);
  assert(!m_post_first_import);
  if (triple.str().empty())
    return false;

  // The triple may change up until a swift::irgen::IRGenModule is created.
  if (m_ir_gen_module_ap.get()) {
    LOG_PRINTF(GetLog(LLDBLog::Types),
               "(\"%s\") ignoring triple "
               "since the IRGenModule has already been created",
               triple.str().c_str());
    return false;
  }

  const unsigned unspecified = 0;
  llvm::Triple adjusted_triple = GetSwiftFriendlyTriple(triple);
  // If the OS version is unspecified, do fancy things.
  if (triple.getOSMajorVersion() == unspecified) {
    // If a triple is "<arch>-apple-darwin" change it to be
    // "<arch>-apple-macosx" otherwise the major and minor OS
    // version we append below would be wrong.
    if (triple.getVendor() == llvm::Triple::VendorType::Apple &&
        triple.getOS() == llvm::Triple::OSType::Darwin)
      adjusted_triple.setOS(llvm::Triple::OSType::MacOSX);

    // Append the min OS to the triple if we have a target
    ModuleSP module_sp;
    if (!module) {
      TargetSP target_sp(GetTargetWP().lock());
      if (target_sp) {
        module_sp = target_sp->GetExecutableModule();
        if (module_sp)
          module = module_sp.get();
      }
    }

    if (module) {
      if (ObjectFile *objfile = module->GetObjectFile())
        if (llvm::VersionTuple version = objfile->GetMinimumOSVersion())
          adjusted_triple.setOSName(adjusted_triple.getOSName().str() +
                                    version.getAsString());
    }
  }

  if (triple.isArch64Bit())
    m_pointer_byte_size = 8;
  else if (triple.isArch32Bit())
    m_pointer_byte_size = 4;
  else if (triple.isArch16Bit())
    m_pointer_byte_size = 2;
  else {
    LOG_PRINTF(GetLog(LLDBLog::Types),
               "Could not set pointer byte size using triple: %s",
               triple.str().c_str());
    m_pointer_byte_size = 0;
  }

  if (llvm::Triple(triple).getOS() == llvm::Triple::UnknownOS) {
    // This case triggers an llvm_unreachable() in the Swift compiler.
    LOG_PRINTF(GetLog(LLDBLog::Types),
               "Cannot initialize Swift with an unknown OS");
    return false;
  }
  LOG_PRINTF(GetLog(LLDBLog::Types), "(\"%s\") setting to \"%s\"",
             triple.str().c_str(), adjusted_triple.str().c_str());

  m_compiler_invocation_ap->setTargetTriple(adjusted_triple);

#ifndef NDEBUG
  assert(GetTriple() == adjusted_triple);
  // We can't call GetASTContext() here because
  // m_initialized_search_path_options and  m_initialized_clang_importer_options
  // need to be initialized before initializing the AST context.
  m_ast_context_mutex.lock();
  assert(!m_ast_context_ap ||
         (llvm::Triple(m_ast_context_ap->LangOpts.Target.getTriple()) ==
          adjusted_triple));
  m_ast_context_mutex.unlock();
#endif

  // Every time the triple is changed the LangOpts must be updated
  // too, because Swift default-initializes the EnableObjCInterop
  // flag based on the triple.
  GetLanguageOptions().EnableObjCInterop = triple.isOSDarwin();
  return true;
}

swift::CompilerInvocation &SwiftASTContext::GetCompilerInvocation() {
  return *m_compiler_invocation_ap;
}

swift::SourceManager &SwiftASTContext::GetSourceManager() {
  if (!m_source_manager_up) {
    m_source_manager_up = std::make_unique<swift::SourceManager>(
        FileSystem::Instance().GetVirtualFileSystem());
  }
  return *m_source_manager_up;
}

swift::LangOptions &SwiftASTContext::GetLanguageOptions() {
  return GetCompilerInvocation().getLangOptions();
}

swift::TypeCheckerOptions &SwiftASTContext::GetTypeCheckerOptions() {
  return GetCompilerInvocation().getTypeCheckerOptions();
}

swift::symbolgraphgen::SymbolGraphOptions &SwiftASTContext::GetSymbolGraphOptions() {
  return GetCompilerInvocation().getSymbolGraphOptions();
}

swift::CASOptions &SwiftASTContext::GetCASOptions() {
  return GetCompilerInvocation().getCASOptions();
}

swift::SerializationOptions &SwiftASTContext::GetSerializationOptions() {
  return GetCompilerInvocation().getSerializationOptions();
}

swift::DiagnosticEngine &SwiftASTContext::GetDiagnosticEngine() {
  if (!m_diagnostic_engine_ap) {
    m_diagnostic_engine_ap.reset(
        new swift::DiagnosticEngine(GetSourceManager()));

    // The following diagnostics are fatal, but they are diagnosed at
    // a very early point where the AST isn't yet destroyed beyond repair.
    m_diagnostic_engine_ap->ignoreDiagnostic(
        swift::diag::serialization_module_too_old.ID);
    m_diagnostic_engine_ap->ignoreDiagnostic(
        swift::diag::serialization_module_too_new.ID);
    m_diagnostic_engine_ap->ignoreDiagnostic(
        swift::diag::serialization_module_language_version_mismatch.ID);
  }
  return *m_diagnostic_engine_ap;
}

swift::SILOptions &SwiftASTContext::GetSILOptions() {
  return GetCompilerInvocation().getSILOptions();
}

bool SwiftASTContext::TargetHasNoSDK() {
  llvm::Triple triple(GetTriple());

  switch (triple.getOS()) {
  case llvm::Triple::OSType::MacOSX:
  case llvm::Triple::OSType::Darwin:
  case llvm::Triple::OSType::IOS:
    return false;
  default:
    return true;
  }
}

swift::ClangImporterOptions &SwiftASTContext::GetClangImporterOptions() {
  swift::ClangImporterOptions &clang_importer_options =
      GetCompilerInvocation().getClangImporterOptions();
  if (!m_initialized_clang_importer_options) {
    m_initialized_clang_importer_options = true;

    // Set the Clang module search path.
    llvm::SmallString<128> path;
    const auto &props = ModuleList::GetGlobalModuleListProperties();
    props.GetClangModulesCachePath().GetPath(path);
    clang_importer_options.ModuleCachePath = std::string(path);
 
    clang_importer_options.DebuggerSupport = true;
    // Matches swift/Frontend.cpp
    clang_importer_options.Optimization = "-Os";
    clang_importer_options.PreferSerializedBridgingHeader =
        props.GetSwiftPreferSerializedBridgingHeader();

    clang_importer_options.DisableSourceImport =
        !props.GetUseSwiftClangImporter();
  }
  return clang_importer_options;
}

swift::SearchPathOptions &SwiftASTContext::GetSearchPathOptions() {
  assert(m_initialized_search_path_options);
  return GetCompilerInvocation().getSearchPathOptions();
}

void SwiftASTContext::InitializeSearchPathOptions(
    llvm::ArrayRef<std::pair<std::string, bool>> extra_module_search_paths,
    llvm::ArrayRef<std::pair<std::string, bool>> extra_framework_search_paths) {
  swift::CompilerInvocation &invocation = GetCompilerInvocation();

  assert(!m_initialized_search_path_options);
  m_initialized_search_path_options = true;

  bool set_sdk = false;
  if (!invocation.getSDKPath().empty()) {
    FileSpec provided_sdk_path(invocation.getSDKPath());
    if (FileSystem::Instance().Exists(provided_sdk_path)) {
      // We don't check whether the SDK supports swift because we figure if
      // someone is passing this to us on the command line (e.g., for the
      // REPL), they probably know what they're doing.

      set_sdk = true;
    }
  }

  llvm::Triple triple(GetTriple());
  std::string resource_dir =
      HostInfo::GetSwiftResourceDir(triple, GetPlatformSDKPath());
  ConfigureResourceDirs(GetCompilerInvocation(), resource_dir, triple);

  std::string sdk_path = GetPlatformSDKPath().str();
  if (TargetSP target_sp = GetTargetWP().lock())
    if (FileSpec &manual_override_sdk = target_sp->GetSDKPath()) {
      set_sdk = false;
      sdk_path = manual_override_sdk.GetPath();
      LOG_PRINTF(GetLog(LLDBLog::Types), "Override target.sdk-path \"%s\"",
                 sdk_path.c_str());
    }

  if (!set_sdk) {
    if (sdk_path.empty()) {
      XcodeSDK::Info info;
      info.type = XcodeSDK::GetSDKTypeForTriple(triple);
      XcodeSDK sdk(info);
      sdk_path = GetSDKPath(m_description, sdk);
    }
    if (sdk_path.empty()) {
      // This fallback is questionable. Perhaps it should be removed.
      XcodeSDK::Info info;
      info.type = XcodeSDK::GetSDKTypeForTriple(
          HostInfo::GetArchitecture().GetTriple());
      XcodeSDK sdk(info);
      sdk_path = GetSDKPath(m_description, sdk);
    }
    if (!sdk_path.empty()) {
      // Note that calling setSDKPath() also recomputes all paths that
      // depend on the SDK path including the
      // RuntimeLibraryImportPaths, which are *only* initialized
      // through this mechanism.
      LOG_PRINTF(GetLog(LLDBLog::Types), "Setting SDK path \"%s\"",
                 sdk_path.c_str());
      invocation.setSDKPath(sdk_path);
    }

    std::vector<std::string> &lpaths =
        invocation.getSearchPathOptions().LibrarySearchPaths;
    lpaths.insert(lpaths.begin(), "/usr/lib/swift");
  }

  // Set the default host plugin paths.
  llvm::SmallString<256> plugin_path;
  llvm::sys::path::append(plugin_path, resource_dir, "host", "plugins");
  if (!FileSystem::Instance().Exists(plugin_path)) {
    LOG_PRINTF(GetLog(LLDBLog::Types), "Host plugin path %s does not exist",
               plugin_path.str().str().c_str());
  } else {
    std::string server = SwiftASTContext::GetPluginServer(plugin_path);
    if (!server.empty() && FileSystem::Instance().Exists(server))
      invocation.getSearchPathOptions().PluginSearchOpts.emplace_back(
          swift::PluginSearchOption::ExternalPluginPath{plugin_path.str().str(),
                                                        server});
  }

  llvm::StringMap<bool> processed;
  std::vector<swift::SearchPathOptions::SearchPath> invocation_import_paths(
      invocation.getSearchPathOptions().getImportSearchPaths());
  // Add all deserialized paths to the map.
  for (const auto &path : invocation_import_paths)
    processed.insert({path.Path, path.IsSystem});

  // Add/unique all extra paths.
  for (const auto &path : extra_module_search_paths) {
    auto it_notseen = processed.insert(path);
    if (it_notseen.second)
      invocation_import_paths.push_back({path.first, path.second});
  }
  invocation.getSearchPathOptions().setImportSearchPaths(
      invocation_import_paths);

  // This preserves the IsSystem bit, but deduplicates entries ignoring it.
  processed.clear();
  std::vector<swift::SearchPathOptions::SearchPath>
      invocation_framework_paths(
          invocation.getSearchPathOptions().getFrameworkSearchPaths());
  // Add all deserialized paths to the map.
  for (const auto &path : invocation_framework_paths)
    processed.insert({path.Path, path.IsSystem});

  // Add/unique all extra paths.
  for (const auto &path : extra_framework_search_paths) {
    auto it_notseen = processed.insert(path);
    if (it_notseen.second)
      invocation_framework_paths.push_back({path.first, path.second});
  }
  invocation.getSearchPathOptions().setFrameworkSearchPaths(
      invocation_framework_paths);

  invocation.computeCXXStdlibOptions();
}

ThreadSafeASTContext SwiftASTContext::GetASTContext() {
  assert(m_initialized_search_path_options &&
         m_initialized_clang_importer_options &&
         "search path options must be initialized before ClangImporter");

  m_ast_context_mutex.lock();

  if (m_ast_context_ap.get())
    return {m_ast_context_ap.get(), m_ast_context_mutex};

  m_ast_context_ap.reset(swift::ASTContext::get(
      GetLanguageOptions(), GetTypeCheckerOptions(), GetSILOptions(),
      GetSearchPathOptions(), GetClangImporterOptions(),
      GetSymbolGraphOptions(), GetCASOptions(), GetSerializationOptions(),
      GetSourceManager(), GetDiagnosticEngine(), /*OutputBackend=*/nullptr));

  if (getenv("LLDB_SWIFT_DUMP_DIAGS")) {
    // NOTE: leaking a swift::PrintingDiagnosticConsumer() here, but
    // this only gets enabled when the above environment variable is
    // set.
    GetDiagnosticEngine().addConsumer(*new swift::PrintingDiagnosticConsumer());
  }

  // Create the ClangImporter and determine the Clang module cache path.
  std::string moduleCachePath =
      GetCompilerInvocation().getClangModuleCachePath().str();
  std::unique_ptr<swift::ClangImporter> clang_importer_ap;
  auto &clang_importer_options = GetClangImporterOptions();
  if (!m_ast_context_ap->SearchPathOpts.getSDKPath().empty() ||
      TargetHasNoSDK()) {
    // Create the DWARFImporterDelegate.
    const auto &props = ModuleList::GetGlobalModuleListProperties();
    if (props.GetUseSwiftDWARFImporter())
      m_dwarfimporter_delegate_up =
          std::make_unique<SwiftDWARFImporterDelegate>(*this);
    auto importer_diags = getScopedDiagnosticConsumer();
    clang_importer_ap = swift::ClangImporter::create(
        *m_ast_context_ap, "", m_dependency_tracker.get(),
        m_dwarfimporter_delegate_up.get());

    // Handle any errors.
    if (!clang_importer_ap || importer_diags->HasErrors()) {
      AddDiagnostic(eSeverityError, "failed to create ClangImporter");
      if (GetLog(LLDBLog::Types)) {
        DiagnosticManager diagnostic_manager;
        importer_diags->PrintDiagnostics(diagnostic_manager);
        std::string underlying_error = diagnostic_manager.GetString();
        HEALTH_LOG_PRINTF("failed to initialize ClangImporter: %s",
                          underlying_error.c_str());
      }
    }
    if (clang_importer_ap) {
      auto clangModuleCache = swift::getModuleCachePathFromClang(
          clang_importer_ap->getClangInstance());
      if (!clangModuleCache.empty())
        moduleCachePath = clangModuleCache;
    }
  }
  LOG_PRINTF(GetLog(LLDBLog::Types), "Using Clang module cache path: %s",
             moduleCachePath.c_str());

  // Compute the prebuilt module cache path to use:
  // <resource-dir>/<platform>/prebuilt-modules/<version>
  llvm::Triple triple(GetTriple());
  std::optional<llvm::VersionTuple> sdk_version =
      m_ast_context_ap->LangOpts.SDKVersion;
  if (!sdk_version) {
    auto SDKInfoOrErr = clang::parseDarwinSDKInfo(
        *llvm::vfs::getRealFileSystem(),
        m_ast_context_ap->SearchPathOpts.getSDKPath());
    if (SDKInfoOrErr) {
      if (auto SDKInfo = *SDKInfoOrErr)
        sdk_version = swift::getTargetSDKVersion(*SDKInfo, triple);
    } else
      llvm::consumeError(SDKInfoOrErr.takeError());
  }
  std::string prebuiltModuleCachePath =
      swift::CompilerInvocation::computePrebuiltCachePath(
          HostInfo::GetSwiftResourceDir(triple, GetPlatformSDKPath()), triple,
          sdk_version);
  if (sdk_version)
    LOG_PRINTF(GetLog(LLDBLog::Types), "SDK version: %s",
               sdk_version->getAsString().c_str());
  LOG_PRINTF(GetLog(LLDBLog::Types),
             "Using prebuilt Swift module cache path: %s",
             prebuiltModuleCachePath.c_str());

  // Determine the Swift module loading mode to use.
  const auto &props = ModuleList::GetGlobalModuleListProperties();
  swift::ModuleLoadingMode loading_mode;
  const char *mode = nullptr;
  switch (props.GetSwiftModuleLoadingMode()) {
  case eSwiftModuleLoadingModePreferSerialized:
    loading_mode = swift::ModuleLoadingMode::PreferSerialized;
    mode = "PreferSerialized";
    break;
  case eSwiftModuleLoadingModePreferInterface:
    loading_mode = swift::ModuleLoadingMode::PreferInterface;
    mode = "PreferInterface";
    break;
  case eSwiftModuleLoadingModeOnlySerialized:
    loading_mode = swift::ModuleLoadingMode::OnlySerialized;
    mode = "OnlySerialized";
    break;
  case eSwiftModuleLoadingModeOnlyInterface:
    loading_mode = swift::ModuleLoadingMode::OnlyInterface;
    mode = "OnlyInterface";
    break;
  }
  if (mode)
    LOG_PRINTF(GetLog(LLDBLog::Types), "Swift module loading mode forced to %s",
               mode);

  // The order here matters due to fallback behaviors:
  //
  // 1. Create and install the memory buffer serialized module loader.
  std::unique_ptr<swift::ModuleLoader> memory_buffer_loader_ap(
      swift::MemoryBufferSerializedModuleLoader::create(
          *m_ast_context_ap, m_dependency_tracker.get(), loading_mode,
          /*IgnoreSwiftSourceInfo*/ false, /*BypassResilience*/ true));
  if (memory_buffer_loader_ap) {
    m_memory_buffer_module_loader =
        static_cast<swift::MemoryBufferSerializedModuleLoader *>(
            memory_buffer_loader_ap.get());
    m_ast_context_ap->addModuleLoader(std::move(memory_buffer_loader_ap));
  }

  // Add a module interface checker.
  m_ast_context_ap->addModuleInterfaceChecker(
    std::make_unique<swift::ModuleInterfaceCheckerImpl>(*m_ast_context_ap,
      moduleCachePath, prebuiltModuleCachePath,
      swift::ModuleInterfaceLoaderOptions()));

  // 2. Create and install the module interface loader.
  //
  // The ordering of 2-4 is the same as the Swift compiler's 1-3,
  // where unintuitively the serialized module loader comes before the
  // module interface loader. The reason for this is that the module
  // interface loader is actually 2-in-1 and secretly attempts to load
  // the serialized module first, and falls back to the serialized
  // module loader, if it is not usable. Contrary to the proper
  // serialized module loader it does this without emitting a
  // diagnostic in the failure case.
  std::unique_ptr<swift::ModuleLoader> module_interface_loader_ap;
  if (loading_mode != swift::ModuleLoadingMode::OnlySerialized) {
    std::unique_ptr<swift::ModuleLoader> module_interface_loader_ap(
        swift::ModuleInterfaceLoader::create(
          *m_ast_context_ap, *static_cast<swift::ModuleInterfaceCheckerImpl*>(
            m_ast_context_ap->getModuleInterfaceChecker()), m_dependency_tracker.get(),
          loading_mode));
    if (module_interface_loader_ap)
      m_ast_context_ap->addModuleLoader(std::move(module_interface_loader_ap));
  }

  // 3. Create and install the serialized module loader.
  std::unique_ptr<swift::ModuleLoader> serialized_module_loader_ap(
      swift::ImplicitSerializedModuleLoader::create(
          *m_ast_context_ap, m_dependency_tracker.get(), loading_mode));
  if (serialized_module_loader_ap)
    m_ast_context_ap->addModuleLoader(std::move(serialized_module_loader_ap));

  // 4. Install the clang importer.
  if (clang_importer_ap) {
    m_clangimporter = (swift::ClangImporter *)clang_importer_ap.get();
    m_ast_context_ap->addModuleLoader(std::move(clang_importer_ap),
                                      /*isClang=*/true);
    m_clangimporter_typesystem = std::make_shared<TypeSystemClang>(
        "ClangImporter-owned clang::ASTContext for '" + m_description,
        m_clangimporter->getClangASTContext());
  }

  // Set up the plugin loader.
  m_ast_context_ap->setPluginLoader(std::make_unique<swift::PluginLoader>(
      *m_ast_context_ap, m_dependency_tracker.get()));

  // Set up the required state for the evaluator in the TypeChecker.
  registerIDERequestFunctions(m_ast_context_ap->evaluator);
  registerParseRequestFunctions(m_ast_context_ap->evaluator);
  registerTypeCheckerRequestFunctions(m_ast_context_ap->evaluator);
  registerClangImporterRequestFunctions(m_ast_context_ap->evaluator);
  registerSILGenRequestFunctions(m_ast_context_ap->evaluator);
  registerSILOptimizerRequestFunctions(m_ast_context_ap->evaluator);
  registerTBDGenRequestFunctions(m_ast_context_ap->evaluator);
  registerIRGenRequestFunctions(m_ast_context_ap->evaluator);
  registerIRGenSILTransforms(*m_ast_context_ap);

  GetASTMap().Insert(m_ast_context_ap.get(), this);

  VALID_OR_RETURN(ThreadSafeASTContext());
  return {m_ast_context_ap.get(), m_ast_context_mutex};
}

ThreadSafeASTContext SwiftASTContext::GetASTContext() const {
  return const_cast<SwiftASTContext *>(this)->GetASTContext();
}

swift::MemoryBufferSerializedModuleLoader *
SwiftASTContext::GetMemoryBufferModuleLoader() {
  VALID_OR_RETURN(nullptr);

  GetASTContext();
  return m_memory_buffer_module_loader;
}

swift::ClangImporter *SwiftASTContext::GetClangImporter() {
  VALID_OR_RETURN(nullptr);

  GetASTContext();
  return m_clangimporter;
}

const std::vector<std::string> &SwiftASTContext::GetClangArguments() {
  return GetClangImporterOptions().ExtraArgs;
}

swift::ModuleDecl *SwiftASTContext::GetCachedModule(std::string module_name) {
  VALID_OR_RETURN(nullptr);
  SwiftModuleMap::const_iterator iter = m_swift_module_cache.find(module_name);

  if (iter != m_swift_module_cache.end())
    return const_cast<swift::ModuleDecl *>(&iter->second);
  return nullptr;
}

llvm::Expected<swift::ModuleDecl &>
SwiftASTContext::CreateEmptyModule(std::string module_name) {
  return CreateModule(module_name, /*importInfo*/ {},
                      /*populateFiles*/ [](auto, auto) {});
}

llvm::Expected<swift::ModuleDecl &> SwiftASTContext::CreateModule(
    std::string module_name, swift::ImplicitImportInfo importInfo,
    swift::ModuleDecl::PopulateFilesFn populateFiles) {
  VALID_OR_RETURN(llvm::createStringError("no context"));
  if (module_name.empty())
    return llvm::createStringError("invalid module name (empty)");

  if (GetCachedModule(module_name))
    return llvm::createStringError(
        llvm::formatv("module already exists for \"{0}\"", module_name));

  ThreadSafeASTContext ast = GetASTContext();
  if (!ast)
    return llvm::createStringError("invalid swift AST (nullptr)");

  swift::Identifier module_id(ast->getIdentifier(module_name));
  auto *module_decl =
      swift::ModuleDecl::create(module_id, **ast, importInfo, populateFiles);
  if (!module_decl)
    return llvm::createStringError(
        llvm::formatv("failed to create module for \"{0}\"", module_name));
  swift::ModuleDecl &decl_ref = *module_decl;
  m_swift_module_cache.insert(
      std::pair<llvm::StringRef, const swift::ModuleDecl &>(module_name,
                                                            decl_ref));
  return *module_decl;
}

void SwiftASTContext::CacheModule(std::string module_name,
                                  swift::ModuleDecl *module) {
  VALID_OR_RETURN();

  if (!module || module_name.empty())
    return;
  if (m_swift_module_cache.find(module_name) != m_swift_module_cache.end())
    return;
  m_swift_module_cache.insert({module_name, *module});
}

/// An RAII object to install a progress report callback.
SwiftASTContext::ModuleImportProgressRAII::ModuleImportProgressRAII(
    SwiftASTContext &ctx, std::string category)
    : m_ts(ctx.shared_from_this()), m_progress(category) {
  if (!m_ts)
    return;
  ThreadSafeASTContext ast = ctx.GetASTContext();
  if (!ast)
    return;
  ast->SetPreModuleImportCallback(
      [&](llvm::StringRef name, swift::ASTContext::ModuleImportKind kind) {
        switch (kind) {
        case swift::ASTContext::Module:
          m_progress.Increment(1, name.str());
          break;
        case swift::ASTContext::Overlay:
          m_progress.Increment(1, name.str() + " (overlay)");
          break;
        case swift::ASTContext::BridgingHeader: {
          // Module imports generate remarks, which are logged, but bridging
          // headers don't.
          auto &m_description = ctx.GetDescription();
          HEALTH_LOG_PRINTF("Compiling bridging header: %s",
                            name.str().c_str());
          m_progress.Increment(1, "Compiling bridging header: " + name.str());
          break;
        }
        }
      });
}

SwiftASTContext::ModuleImportProgressRAII::~ModuleImportProgressRAII() {
  if (!m_ts)
    return;
  ThreadSafeASTContext ast =
      llvm::cast<SwiftASTContext>(m_ts.get())->GetASTContext();
  if (!ast)
    return;
  ast->SetPreModuleImportCallback(
      [](llvm::StringRef, swift::ASTContext::ModuleImportKind) {});
}

std::unique_ptr<SwiftASTContext::ModuleImportProgressRAII>
SwiftASTContext::GetModuleImportProgressRAII(std::string category) {
  return std::make_unique<SwiftASTContext::ModuleImportProgressRAII>(*this,
                                                                     category);
}

static constexpr llvm::StringLiteral g_invalid_context =
    "could not initialize Swift compiler, run swift-healthcheck for more info";

llvm::Expected<swift::ModuleDecl &>
SwiftASTContext::GetModule(const SourceModule &module, bool *cached) {
  if (cached)
    *cached = false;

  VALID_OR_RETURN(llvm::createStringError(g_invalid_context));
  std::string module_name = llvm::join(module.path, ".");

  LOG_PRINTF(GetLog(LLDBLog::Types), "(\"%s\")", module_name.c_str());

  if (module_name.empty()) {
    LOG_PRINTF(GetLog(LLDBLog::Types), "empty module name");
    return llvm::createStringError("invalid module name (empty)");
  }

  if (swift::ModuleDecl *module_decl = GetCachedModule(module_name)) {
    if (cached)
      *cached = true;
    return *module_decl;
  }

  ThreadSafeASTContext ast = GetASTContext();
  if (!ast) {
    LOG_PRINTF(GetLog(LLDBLog::Types), "(\"%s\") invalid ASTContext",
               module.path.front().GetCString());

    return llvm::createStringError("invalid swift::ASTContext");
  }

  if (HasFatalErrors()) {
    return llvm::createStringError(
        llvm::formatv("failed to get module \"{0}\" from AST context:\n"
                      "AST context is in a fatal error state",
                      module_name));
  }

  // Create a diagnostic consumer for the diagnostics produced by the import.
  auto import_diags = getScopedDiagnosticConsumer();

  swift::ModuleDecl *module_decl = ast->getModuleByName(module_name);

  // Error handling.
  if (import_diags->HasErrors()) {
    DiagnosticManager diagnostic_manager;
    import_diags->PrintDiagnostics(diagnostic_manager);
    std::string diagnostic = diagnostic_manager.GetString();
    LOG_PRINTF(GetLog(LLDBLog::Types), "(\"%s\") -- %s",
               module.path.front().GetCString(), diagnostic.c_str());
    return llvm::createStringError(
        llvm::formatv("failed to get module \"{0}\" from AST context:\n{1}",
                      module_name, diagnostic));
  }

  if (!module_decl) {
    LOG_PRINTF(GetLog(LLDBLog::Types), "failed with no error");
    return llvm::createStringError(llvm::formatv(
        "failed to get module \"{0}\" from AST context", module_name));
  }
  LOG_PRINTF(GetLog(LLDBLog::Types), "(\"%s\") -- found %s",
             module_name.c_str(), module_decl->getName().str().str().c_str());

  m_swift_module_cache.insert({module_name, *module_decl});
  return *module_decl;
}

llvm::Expected<swift::ModuleDecl &>
SwiftASTContext::ImportStdlib() {
  SourceModule module_info;
  module_info.path.emplace_back(swift::STDLIB_NAME);
  return GetModule(module_info);
}

llvm::Expected<swift::ModuleDecl &>
SwiftASTContext::GetModule(const FileSpec &module_spec) {
  VALID_OR_RETURN(llvm::createStringError("no context"));

  ConstString module_basename(module_spec.GetFileNameStrippingExtension());

  LOG_PRINTF(GetLog(LLDBLog::Types), "(\"%s\")", module_spec.GetPath().c_str());

  if (!module_basename) {
    LOG_PRINTF(GetLog(LLDBLog::Types), "((FileSpec)\"%s\") -- no basename",
               module_spec.GetPath().c_str());
    return llvm::createStringError(
        llvm::formatv("no module basename in \"{0}\"", module_spec.GetPath()));
  }

  if (auto *cached = GetCachedModule(module_basename.GetString()))
    return *cached;

  if (!FileSystem::Instance().Exists(module_spec)) {
    LOG_PRINTF(GetLog(LLDBLog::Types), "((FileSpec)\"%s\") -- doesn't exist",
               module_spec.GetPath().c_str());
    return llvm::createStringError(
        llvm::formatv("module \"{0}\" doesn't exist", module_spec.GetPath()));
  }
  ThreadSafeASTContext ast = GetASTContext();
  if (!GetClangImporter()) {
    LOG_PRINTF(GetLog(LLDBLog::Types),
               "((FileSpec)\"%s\") -- no ClangImporter so giving up",
               module_spec.GetPath().c_str());
    return llvm::createStringError("couldn't get a ClangImporter");
  }

  std::string module_directory(module_spec.GetDirectory().GetCString());
  bool add_search_path = true;
  for (auto path : ast->SearchPathOpts.getImportSearchPaths()) {
    if (path.Path == module_directory) {
      add_search_path = false;
      break;
    }
  }
  // Add the search path if needed so we can find the module by basename.
  if (add_search_path) {
    ast->addSearchPath(module_directory, /*isFramework=*/false,
                       /*isSystem=*/false);
  }

  typedef swift::Located<swift::Identifier> ModuleNameSpec;
  llvm::StringRef module_basename_sref(module_basename.GetCString());
  ModuleNameSpec name_pair(ast->getIdentifier(module_basename_sref),
                           swift::SourceLoc());
  swift::ModuleDecl *module =
      ast->getModule(llvm::ArrayRef<ModuleNameSpec>(name_pair));
  if (module) {
    LOG_PRINTF(GetLog(LLDBLog::Types), "((FileSpec)\"%s\") -- found %s",
               module_spec.GetPath().c_str(),
               module->getName().str().str().c_str());

    m_swift_module_cache.insert({module_basename, *module});
    return *module;
  }
  LOG_PRINTF(GetLog(LLDBLog::Types),
             "((FileSpec)\"%s\") -- couldn't get from AST context",
             module_spec.GetPath().c_str());

  return llvm::createStringError(llvm::formatv(
      "failed to get module \"{0}\" from AST context", module_spec.GetPath()));
}

template<typename ModuleT> swift::ModuleDecl *
SwiftASTContext::FindAndLoadModule(const ModuleT &module, Process &process,
                                   bool import_dylib, Status &error) {
  VALID_OR_RETURN(nullptr);

  bool cached = false;
  auto swift_module_or_err = GetModule(module, &cached);
  if (!swift_module_or_err) {
    error = Status::FromError(swift_module_or_err.takeError());
    return nullptr;
  }
  swift::ModuleDecl *swift_module = &*swift_module_or_err;

  if (!swift_module)
    return nullptr;

  // If import_dylib is true, this is an explicit "import Module"
  // declaration in a user expression, and we should load the dylib
  // even if we already cached an implicit import (which may not have
  // loaded the dylib).  If target.swift-auto-import-frameworks is
  // set, all implicitly imported Swift modules' associated frameworks
  // will be imported too.
  TargetSP target_sp(GetTargetWP().lock());
  if (target_sp)
    import_dylib |= target_sp->GetSwiftAutoImportFrameworks();

  if (cached && !import_dylib)
    return swift_module;

  if (import_dylib)
    LoadModule(swift_module, process, error);

  return swift_module;
}

bool SwiftASTContext::LoadOneImage(Process &process, FileSpec &link_lib_spec,
                                   Status &error) {
  VALID_OR_RETURN(false);

  error.Clear();

  PlatformSP platform_sp = process.GetTarget().GetPlatform();
  if (platform_sp)
    return platform_sp->LoadImage(&process, FileSpec(), link_lib_spec, error) !=
           LLDB_INVALID_IMAGE_TOKEN;
  else
    return false;
}

static std::vector<std::string>
GetLibrarySearchPaths(const swift::SearchPathOptions &search_path_opts) {
  // The order in which we look up the libraries is important. The REPL
  // dlopen()s libswiftCore, and gives precedence to the just built standard
  // library instead of the one in the OS. When we type `import Foundation`,
  // we want to make sure we end up loading the correct library, i.e. the
  // one sitting next to the stdlib we just built, and then fall back to the
  // one in the OS if that's not available.
  std::vector<std::string> paths;
  for (std::string path : search_path_opts.RuntimeLibraryPaths)
    paths.push_back(path);
  for (std::string path : search_path_opts.LibrarySearchPaths)
    paths.push_back(path);
  for (const std::string &path : HostInfo::GetSwiftLibrarySearchPaths())
    paths.emplace_back(path);
  return paths;
}

void SwiftASTContext::LoadModule(swift::ModuleDecl *swift_module,
                                 Process &process, Status &error) {
  VALID_OR_RETURN();

  Status current_error;
  auto addLinkLibrary = [&](swift::LinkLibrary link_lib) {
    Status load_image_error;
    StreamString all_dlopen_errors;
    std::string library_name = link_lib.getName().str();

    if (library_name.empty()) {
      error = Status::FromErrorString(
          "Empty library name passed to addLinkLibrary");
      return;
    }

    SwiftLanguageRuntime *runtime = SwiftLanguageRuntime::Get(&process);
    if (runtime && runtime->IsInLibraryNegativeCache(library_name))
      return;

    swift::LibraryKind library_kind = link_lib.getKind();
    LOG_PRINTF(GetLog(LLDBLog::Types | LLDBLog::Expressions),
               "Loading linked %s \"%s\".",
               library_kind == swift::LibraryKind::Framework ? "framework"
                                                             : "library",
               library_name.c_str());

    switch (library_kind) {
    case swift::LibraryKind::Framework: {
      // First make sure the library isn't already loaded. Since this
      // is a framework, we make sure the file name and the framework
      // name are the same, and that we are contained in
      // FileName.framework with no other intervening frameworks.  We
      // can get more restrictive if this gives false positives.
      //
      // If the framework exists on disk but it's a static framework
      // (i.e., the binary inside is a static archive instead of a
      // dylib) this cannot be detected. The dlopen call will fail,
      // and dlerror does not contain enough information to
      // unambiguously identify this case. So it will look as if the
      // framework hasn't been found.
      ConstString library_cstr(library_name);

      std::string framework_name(library_name);
      framework_name.append(".framework");

      // Lookup the module by file basename and make sure that
      // basename has "<basename>.framework" in the path.
      ModuleSpec module_spec;
      module_spec.GetFileSpec().SetFilename(library_cstr);
      lldb_private::ModuleList matching_module_list;
      bool module_already_loaded = false;
      process.GetTarget().GetImages().FindModules(module_spec,
                                                  matching_module_list);
      if (!matching_module_list.IsEmpty()) {
        matching_module_list.ForEach(
            [&module_already_loaded, &module_spec,
             &framework_name](const ModuleSP &module_sp) -> IterationAction {
              module_already_loaded = module_spec.GetFileSpec().GetPath().find(
                                          framework_name) != std::string::npos;
              // Keep iterating if we didn't find the right module
              if (!module_already_loaded)
                return IterationAction::Continue;

              return IterationAction::Stop;
            });
      }
      // If we already have this library loaded, don't try and load it again.
      if (module_already_loaded) {
        LOG_PRINTF(GetLog(LLDBLog::Types),
                   "Skipping load of %s as it is already loaded.",
                   framework_name.c_str());
        return;
      }

      for (auto module : process.GetTarget().GetImages().Modules()) {
        FileSpec module_file = module->GetFileSpec();
        if (module_file.GetFilename() == library_cstr) {
          std::string module_path = module_file.GetPath();

          size_t framework_offset = module_path.rfind(framework_name);

          if (framework_offset != std::string::npos) {
            // The Framework is already loaded, so we don't need to try to load
            // it again.
            LOG_PRINTF(GetLog(LLDBLog::Types),
                       "Skipping load of %s as it is already loaded.",
                       framework_name.c_str());
            return;
          }
        }
      }

      std::string framework_path("@rpath/");
      framework_path.append(library_name);
      framework_path.append(".framework/");
      framework_path.append(library_name);
      FileSpec framework_spec(framework_path.c_str());

      if (LoadOneImage(process, framework_spec, load_image_error)) {
        LOG_PRINTF(GetLog(LLDBLog::Types), "Found framework at: %s.",
                   framework_path.c_str());

        return;
      } else
        all_dlopen_errors.Printf("Looking for \"%s\", error: %s\n",
                                 framework_path.c_str(),
                                 load_image_error.AsCString());

      // And then in the various framework search paths.
      std::unordered_set<std::string> seen_paths;
      std::vector<std::string> uniqued_paths;

      for (const auto &framework_search_dir :
           swift_module->getASTContext()
               .SearchPathOpts.getFrameworkSearchPaths()) {
        // The framework search dir as it comes from the AST context
        // often has duplicate entries, don't try to load along the
        // same path twice.
        std::pair<std::unordered_set<std::string>::iterator, bool>
            insert_result = seen_paths.insert(framework_search_dir.Path);
        if (insert_result.second) {
          framework_path = framework_search_dir.Path;
          framework_path.append("/");
          framework_path.append(library_name);
          framework_path.append(".framework/");
          uniqued_paths.push_back(framework_path);
        }
      }

      uint32_t token = LLDB_INVALID_IMAGE_TOKEN;
      PlatformSP platform_sp = process.GetTarget().GetPlatform();

      Status error;
      FileSpec library_spec(library_name);
      FileSpec found_path;

      if (platform_sp)
        token = platform_sp->LoadImageUsingPaths(
            &process, library_spec, uniqued_paths, error, &found_path);

      if (token != LLDB_INVALID_IMAGE_TOKEN) {
        LOG_PRINTF(GetLog(LLDBLog::Types), "Found framework at: %s.",
                   framework_path.c_str());

        return;
      } else {
        all_dlopen_errors.Printf("Failed to find framework for \"%s\" looking"
                                 " along paths:\n",
                                 library_name.c_str());
        for (const std::string &path : uniqued_paths)
          all_dlopen_errors.Printf("  %s\n", path.c_str());
      }

      // Maybe we were told to add a link library that exists in the
      // system.  I tried just specifying Foo.framework/Foo and
      // letting the system search figure that out, but if
      // DYLD_FRAMEWORK_FALLBACK_PATH is set (e.g. in Xcode's test
      // scheme) then these aren't found. So for now I dial them in
      // explicitly:
      std::string system_path("/System/Library/Frameworks/");
      system_path.append(library_name);
      system_path.append(".framework/");
      system_path.append(library_name);
      framework_spec.SetFile(system_path.c_str(), FileSpec::Style::native);
      if (LoadOneImage(process, framework_spec, load_image_error))
        return;
      else
        all_dlopen_errors.Printf("Looking for \"%s\"\n,    error: %s\n",
                                 framework_path.c_str(),
                                 load_image_error.AsCString());
    } break;
    case swift::LibraryKind::Library: {
      std::vector<std::string> search_paths =
          GetLibrarySearchPaths(swift_module->getASTContext().SearchPathOpts);

      if (LoadLibraryUsingPaths(process, library_name, search_paths, true,
                                all_dlopen_errors))
        return;
    } break;
    }

    // If we get here, we aren't going to find this image, so add it to a
    // negative cache:
    if (runtime)
      runtime->AddToLibraryNegativeCache(library_name);

    current_error = Status::FromErrorStringWithFormatv(
        "Failed to load linked library {0} of module {1} - errors:\n{2}\n",
        library_name, swift_module->getName().str().str(),
        all_dlopen_errors.GetData());
  };

  for (auto import : swift::namelookup::getAllImports(swift_module)) {
    import.importedModule->collectLinkLibraries(addLinkLibrary);
  }
  error = current_error.Clone();
}

bool SwiftASTContext::LoadLibraryUsingPaths(
    Process &process, llvm::StringRef library_name,
    std::vector<std::string> &search_paths, bool check_rpath,
    StreamString &all_dlopen_errors) {
  VALID_OR_RETURN(false);

  SwiftLanguageRuntime *runtime = SwiftLanguageRuntime::Get(&process);
  if (!runtime) {
    all_dlopen_errors.PutCString(
        "Can't load Swift libraries without a language runtime.");
    return false;
  }

  PlatformSP platform_sp(process.GetTarget().GetPlatform());

  std::string library_fullname;

  if (platform_sp) {
    library_fullname =
        platform_sp->GetFullNameForDylib(ConstString(library_name)).AsCString();
  } else // This is the old way, and we shouldn't use it except on Mac OS
  {
#ifdef __APPLE__
    library_fullname = "lib";
    library_fullname.append(library_name.str());
    library_fullname.append(".dylib");
#else
    return false;
#endif
  }

  ModuleSpec module_spec;
  module_spec.GetFileSpec().SetFilename(library_fullname);

  if (process.GetTarget().GetImages().FindFirstModule(module_spec)) {
    LOG_PRINTF(GetLog(LLDBLog::Types),
               "Skipping module %s as it is already loaded.",
               library_fullname.c_str());
    return true;
  }

  std::string library_path;
  std::unordered_set<std::string> seen_paths;
  Status load_image_error;
  std::vector<std::string> uniqued_paths;

  for (const std::string &library_search_dir : search_paths) {
    // The library search dir as it comes from the AST context often
    // has duplicate entries, so lets unique the path list before we
    // send it down to the target.
    std::pair<std::unordered_set<std::string>::iterator, bool> insert_result =
        seen_paths.insert(library_search_dir);
    if (insert_result.second)
      uniqued_paths.push_back(library_search_dir);
  }

  detail::SwiftLibraryLookupRequest library_request;
  library_request.library_name = library_fullname;
  library_request.search_paths = uniqued_paths;
  library_request.check_rpath = check_rpath;
  library_request.process_uid = process.GetUniqueID();

  // If this library was requested to load before, don't try to load it again.
  // This is mostly done for performance reasons in case the loaded image
  // check above didn't correctly detect whether a library was loaded before.
  // See also rdar://74454500 for more details.
  auto library_load_iter = library_load_cache.find(library_request);
  if (library_load_iter != library_load_cache.end())
    return library_load_iter->second;
  // Pretend for now we loaded the library successfully. There are multiple
  // early-returns in the following code for successful loads and this should
  // cover all of them. The single failure code branch at the end changes the
  // value to false if all load attempts have failed.
  library_load_cache[library_request] = true;

  FileSpec library_spec(library_fullname);
  FileSpec found_library;
  uint32_t token = LLDB_INVALID_IMAGE_TOKEN;
  Status error;
  if (platform_sp)
    token = platform_sp->LoadImageUsingPaths(
        &process, library_spec, uniqued_paths, error, &found_library);
  if (token != LLDB_INVALID_IMAGE_TOKEN) {
    LOG_PRINTF(GetLog(LLDBLog::Types), "Found library at: %s.",
               found_library.GetPath().c_str());
    return true;
  } else {
    all_dlopen_errors.Printf("Failed to find \"%s\" in paths:\n",
                             library_fullname.c_str());
    for (const std::string &search_dir : uniqued_paths)
      all_dlopen_errors.Printf("  %s\n", search_dir.c_str());
  }

  if (check_rpath) {
    // Let our RPATH help us out when finding the right library.
    library_path = "@rpath/";
    library_path += library_fullname;

    FileSpec link_lib_spec(library_path.c_str());

    if (LoadOneImage(process, link_lib_spec, load_image_error)) {
      LOG_PRINTF(GetLog(LLDBLog::Types), "Found library using RPATH at: %s.",
                 library_path.c_str());
      return true;
    } else
      all_dlopen_errors.Printf("Failed to find \"%s\" on RPATH, error: %s\n",
                               library_fullname.c_str(),
                               load_image_error.AsCString());
  }

  // Remember that this failed library failed to load so we don't try again.
  library_load_cache[library_request] = false;
  return false;
}

void SwiftASTContext::LoadExtraDylibs(Process &process, Status &error) {
  VALID_OR_RETURN();

  error.Clear();
  swift::IRGenOptions &irgen_options = GetIRGenOptions();
  for (const swift::LinkLibrary &link_lib : irgen_options.LinkLibraries) {
    // We don't have to do frameworks here, they actually record their link
    // libraries properly.
    if (link_lib.getKind() == swift::LibraryKind::Library) {
      StringRef library_name = link_lib.getName();
      StreamString errors;

      std::vector<std::string> search_paths = GetLibrarySearchPaths(
          m_compiler_invocation_ap->getSearchPathOptions());

      bool success = LoadLibraryUsingPaths(process, library_name, search_paths,
                                           false, errors);
      if (!success) {
        error = Status::FromErrorString(errors.GetData());
      }
    }
  }
}

static std::string GetBriefModuleName(Module &module) {
  std::string name;
  {
    llvm::raw_string_ostream ss(name);
    module.GetDescription(ss, eDescriptionLevelBrief);
  }
  return name;
}

void SwiftASTContext::RegisterSectionModules(
    Module &module, std::vector<std::string> &module_names) {
  VALID_OR_RETURN();

  swift::MemoryBufferSerializedModuleLoader *loader =
      GetMemoryBufferModuleLoader();
  if (!loader)
    return;

  SectionList *section_list = module.GetSectionList();
  if (!section_list)
    return;

  llvm::Triple filter = GetTriple();
  auto parse_ast_section = [&](llvm::StringRef section_data_ref, size_t n,
                               size_t total) {
    auto Result = swift::parseASTSection(*loader, section_data_ref, filter);
    if (auto E = Result.takeError()) {
      std::string error = toString(std::move(E));
      LOG_PRINTF(
          GetLog(LLDBLog::Types),
          "failed to parse AST section %zu/%zu in image \"%s\" "
          "(filter=\"%s\"). %s",
          n, total,
          module.GetFileSpec().GetFilename().AsCString("<unknown module>"),
          filter.str().c_str(), error.c_str());
      return;
    }

    // Collect the Swift module names referenced by the AST.
    for (auto module_name : *Result) {
      module_names.push_back(module_name);
      LOG_PRINTF(
          GetLog(LLDBLog::Types),
          "parsed module \"%s\" from Swift AST section %zu/%zu in "
          "image \"%s\" (filter=\"%s\").",
          module_name.c_str(), n, total,
          module.GetFileSpec().GetFilename().AsCString("<unknown module>"),
          filter.str().c_str());
    }
  };

  if (m_ast_file_data_map.find(&module) != m_ast_file_data_map.end())
    return;

  // Grab all the AST blobs from the symbol vendor.
  auto ast_file_datas = module.GetASTData(eLanguageTypeSwift);
    if (ast_file_datas.size())
  LOG_PRINTF(GetLog(LLDBLog::Types),
             "(\"%s\") retrieved %zu AST Data blobs from the symbol vendor "
             "(filter=\"%s\").",
             GetBriefModuleName(module).c_str(), ast_file_datas.size(),
             filter.str().c_str());

  // Add each of the AST blobs to the vector of AST blobs for
  // the module.
  auto &ast_vector = GetASTVectorForModule(&module);
  ast_vector.insert(ast_vector.end(), ast_file_datas.begin(),
                    ast_file_datas.end());

  // Retrieve the module names from the AST blobs retrieved
  // from the symbol vendor.
  size_t i = 0;
  for (auto ast_file_data_sp : ast_file_datas) {
    // Parse the AST section info from the AST blob.
    llvm::StringRef section_data_ref((const char *)ast_file_data_sp->GetBytes(),
                                     ast_file_data_sp->GetByteSize());
    parse_ast_section(section_data_ref, ++i, ast_file_datas.size());
  }
}

void SwiftASTContext::ImportSectionModules(
    Module &module, const std::vector<std::string> &module_names) {
  VALID_OR_RETURN();

  auto module_import_progress_raii =
      GetModuleImportProgressRAII("Importing Swift section modules");
  for (const std::string &module_name : module_names) {
    SourceModule module_info;
    module_info.path.push_back(ConstString(module_name));
    auto module_or_err = GetModule(module_info);
    if (!module_or_err)
      module.ReportWarning("unable to load swift module \"{0}\" ({1})",
                           module_name,
                           llvm::toString(module_or_err.takeError()));
  }
}

swift::Identifier SwiftASTContext::GetIdentifier(const llvm::StringRef &name) {
  VALID_OR_RETURN(swift::Identifier());

  return GetASTContext()->getIdentifier(name);
}

ConstString SwiftASTContext::GetMangledTypeName(swift::TypeBase *type_base) {
  VALID_OR_RETURN(ConstString());

  auto iter = m_type_to_mangled_name_map.find(type_base),
       end = m_type_to_mangled_name_map.end();
  if (iter != end)
    return ConstString(iter->second);

  swift::Type swift_type(type_base);

  assert(!swift_type->hasArchetype() &&
         "type has not been mapped out of context");
  ThreadSafeASTContext ast_ctx = GetASTContext();
  if (!ast_ctx)
    return {};

  swift::Mangle::ASTMangler mangler(**ast_ctx, true);
  std::string s = mangler.mangleTypeForDebugger(swift_type, nullptr);
  if (s.empty())
    return {};

  ConstString mangled_cs{StringRef(s)};
  CacheDemangledType(mangled_cs, type_base);
  return mangled_cs;
}

void SwiftASTContext::CacheDemangledType(ConstString name,
                                         swift::TypeBase *found_type) {
  VALID_OR_RETURN();

  m_type_to_mangled_name_map.insert({found_type, name.AsCString()});
  m_mangled_name_to_type_map.insert({name.AsCString(), found_type});
}

void SwiftASTContext::CacheDemangledTypeFailure(ConstString name) {
  VALID_OR_RETURN();

  m_negative_type_cache.Insert(name.AsCString());
}

/// The old TypeReconstruction implementation would reconstruct SILFunctionTypes
/// with one argument T and one result U as an AST FunctionType (T) -> U;
/// anything with multiple arguments or results was reconstructed as () -> ().
///
/// Since this is non-sensical, let's just reconstruct all SILFunctionTypes as
/// () -> () for now.
///
/// What we should really do is only mangle AST types in DebugInfo, but that
/// requires some more plumbing on the Swift side to properly handle generic
/// specializations.
static swift::Type ConvertSILFunctionTypesToASTFunctionTypes(swift::Type t) {
  return t.transformRec([](swift::TypeBase *t) -> std::optional<swift::Type> {
    if (swift::dyn_cast<swift::SILFunctionType>(t)) {
      // FIXME: Verify ExtInfo state is correct, not working by accident.
      swift::FunctionType::ExtInfo info;
      return swift::Type(
          swift::FunctionType::get({}, t->getASTContext().TheEmptyTupleType,
                                   info));
    }
    return std::nullopt;
  });
}

CompilerType
SwiftASTContext::GetTypeFromMangledTypename(ConstString mangled_typename) {
  if (llvm::isa<SwiftASTContextForExpressions>(this))
    return GetCompilerType(
        llvm::expectedToStdOptional(ReconstructType(mangled_typename))
            .value_or(nullptr));
  return GetCompilerType(mangled_typename);
}

CompilerType SwiftASTContext::GetAsClangType(ConstString mangled_name) {
  if (!swift::Demangle::isObjCSymbol(mangled_name.GetStringRef()))
    return {};

  // When we failed to look up the type because no .swiftmodule is
  // present or it couldn't be read, fall back to presenting objects
  // that look like they might be come from Objective-C (or C) as
  // Clang types. LLDB's Objective-C part is very robust against
  // malformed object pointers, so this isn't very risky.
  if (!m_module)
    return {};
  auto type_system_or_err =
      m_module->GetTypeSystemForLanguage(eLanguageTypeObjC);
  if (!type_system_or_err) {
    llvm::consumeError(type_system_or_err.takeError());
    return {};
  }

  auto *clang_ctx =
      llvm::dyn_cast_or_null<TypeSystemClang>(type_system_or_err->get());
  if (!clang_ctx)
    return {};
  DWARFASTParserClang *clang_ast_parser =
      static_cast<DWARFASTParserClang *>(clang_ctx->GetDWARFParser());

  SymbolContext sc;
  m_module->CalculateSymbolContext(&sc);
  CompilerType clang_type;
  CompilerType imported_type = GetCompilerType(mangled_name);
  if (auto ts = imported_type.GetTypeSystem()
                    .dyn_cast_or_null<TypeSystemSwiftTypeRef>())
    if (ts->IsImportedType(imported_type.GetOpaqueQualType(), nullptr))
      if (TypeSP result = ts->LookupClangType(mangled_name, sc))
        clang_type = result->GetForwardCompilerType();

  // Import the Clang type into the Clang context.
  if (!clang_type)
    return {};

  if (clang_type.GetTypeSystem().GetSharedPointer().get() != clang_ctx)
    clang_type = clang_ast_parser->GetClangASTImporter().CopyType(*clang_ctx,
                                                                  clang_type);
  // Swift doesn't know pointers. Convert top-level
  // Objective-C object types to object pointers for Clang.
  auto qual_type =
      clang::QualType::getFromOpaquePtr(clang_type.GetOpaqueQualType());
  if (qual_type->isObjCObjectOrInterfaceType())
    clang_type = clang_type.GetPointerType();

  // Fall back to (id), which is not necessarily correct.
  if (!clang_type)
    clang_type = clang_ctx->GetBasicType(eBasicTypeObjCID);
  return clang_type;
}

swift::TypeBase *
SwiftASTContext::ReconstructTypeOrWarn(ConstString mangled_typename) {
  Status error;

  auto reconstructed_type = ReconstructType(mangled_typename);
  if (!reconstructed_type)
    AddDiagnostic(eSeverityWarning,
                  llvm::toString(reconstructed_type.takeError()));
  return *reconstructed_type;
}

llvm::Expected<swift::TypeBase *>
SwiftASTContext::ReconstructType(ConstString mangled_typename) {
  VALID_OR_RETURN(nullptr);

  const char *mangled_cstr = mangled_typename.AsCString();
  if (mangled_typename.IsEmpty() || !SwiftLanguageRuntime::IsSwiftMangledName(
                                        mangled_typename.GetStringRef())) {
    return llvm::createStringError("typename \"" +
                                   mangled_typename.GetStringRef() +
                                   "\" is not a valid Swift mangled name");
  }

  if (swift_demangle::ContainsError(mangled_typename)) {
    LOG_PRINTF(GetLog(LLDBLog::Types),
               "(\"%s\") -- cannot reconstruct ErrorType", mangled_cstr);
    return llvm::createStringError("cannot reconstruct ErrorType");
  }

  LOG_VERBOSE_PRINTF(GetLog(LLDBLog::Types), "(\"%s\")", mangled_cstr);

  ThreadSafeASTContext ast_ctx = GetASTContext();
  if (!ast_ctx) {
    LOG_PRINTF(GetLog(LLDBLog::Types), "(\"%s\") -- null Swift AST Context",
               mangled_cstr);
    return llvm::createStringError("null Swift AST Context");
  }

  // If we were to crash doing this, remember what type caused it.
  llvm::PrettyStackTraceFormat PST("error finding type for %s", mangled_cstr);
  swift::TypeBase *found_type = m_mangled_name_to_type_map.lookup(mangled_cstr);
  if (found_type) {
    LOG_VERBOSE_PRINTF(GetLog(LLDBLog::Types),
                       "(\"%s\") -- found in the positive cache", mangled_cstr);
    assert(&found_type->getASTContext() == *ast_ctx);
    return found_type;
  }

  if (m_negative_type_cache.Lookup(mangled_cstr)) {
    LOG_PRINTF(GetLog(LLDBLog::Types),
               "(\"%s\") -- found in the negative cache", mangled_cstr);
    return llvm::createStringError("type for typename \"" +
                                   mangled_typename.GetString() +
                                   "\" was not found (cached)");
  }

  LOG_PRINTF(GetLog(LLDBLog::Types), "(\"%s\") -- not cached, searching",
             mangled_cstr);

  found_type = swift::Demangle::getTypeForMangling(
                   **ast_ctx, mangled_typename.GetStringRef())
                   .getPointer();
  assert(!found_type || &found_type->getASTContext() == *ast_ctx);

  // This type might have been been found in reflection and annotated with
  // @_originallyDefinedIn. The compiler emits a typelias for these type
  // pointing them back to the types with the real module name.
  if (!found_type) {
    auto adjusted =
        GetTypeSystemSwiftTypeRef()->AdjustTypeForOriginallyDefinedInModule(
            mangled_typename);
    found_type =
        swift::Demangle::getTypeForMangling(**ast_ctx, adjusted).getPointer();
  }
  // Objective-C classes sometimes have private subclasses that are invisible
  // to the Swift compiler because they are declared and defined in a .m file.
  // If we can't reconstruct an ObjC type, walk up the type hierarchy until we
  // find something we can import, or until we run out of types
  while (!found_type) {
    CompilerType clang_type = GetAsClangType(mangled_typename);
    if (!clang_type)
      break;

    auto clang_ctx =
        clang_type.GetTypeSystem().dyn_cast_or_null<TypeSystemClang>();
    if (!clang_ctx)
      break;
    auto *interface_decl = TypeSystemClang::GetAsObjCInterfaceDecl(clang_type);
    if (!interface_decl)
      break;
    auto *super_interface_decl = interface_decl->getSuperClass();
    if (!super_interface_decl)
      break;
    CompilerType super_type = clang_ctx->GetTypeForDecl(super_interface_decl);
    if (!super_type)
      break;
    auto super_mangled_typename = super_type.GetMangledTypeName();
    found_type = swift::Demangle::getTypeForMangling(
                     **ast_ctx, super_mangled_typename.GetStringRef())
                     .getPointer();
    assert(!found_type || &found_type->getASTContext() == *ast_ctx);
  }

  if (found_type) {
    swift::TypeBase *ast_type =
        ConvertSILFunctionTypesToASTFunctionTypes(found_type).getPointer();
    assert(ast_type);
    assert(&ast_type->getASTContext() == *ast_ctx);
    // This transformation is lossy: all SILFunction types are mapped
    // to the same AST type. We thus cannot cache the result, since
    // the mapping isn't bijective.
    if (ast_type == found_type)
      CacheDemangledType(mangled_typename, ast_type);
    CompilerType result_type = {weak_from_this(), ast_type};
    LOG_PRINTF(GetLog(LLDBLog::Types), "(\"%s\") -- found %s", mangled_cstr,
               result_type.GetTypeName().GetCString());
    return ast_type;
  }

  LOG_PRINTF(GetLog(LLDBLog::Types), "(\"%s\") -- not found", mangled_cstr);

  CacheDemangledTypeFailure(mangled_typename);
  return llvm::createStringError("type for typename \"" +
                                 mangled_typename.GetString() +
                                 "\" was not found");
}

CompilerType SwiftASTContext::GetAnyObjectType() {
  VALID_OR_RETURN(CompilerType());
  return ToCompilerType({GetASTContext()->getAnyObjectType()});
}

static CompilerType ValueDeclToType(swift::ValueDecl *decl) {
  if (decl) {
    switch (decl->getKind()) {
    case swift::DeclKind::TypeAlias: {
      swift::TypeAliasDecl *alias_decl =
          swift::cast<swift::TypeAliasDecl>(decl);
      swift::Type swift_type = swift::TypeAliasType::get(
          alias_decl, swift::Type(), llvm::ArrayRef<swift::Type>(),
          alias_decl->getUnderlyingType());
      return ToCompilerType({swift_type.getPointer()});
    }

    case swift::DeclKind::Enum:
    case swift::DeclKind::Struct:
    case swift::DeclKind::Protocol:
    case swift::DeclKind::Class: {
      swift::NominalTypeDecl *nominal_decl =
          swift::cast<swift::NominalTypeDecl>(decl);
      swift::Type swift_type = nominal_decl->getDeclaredType();
      return ToCompilerType({swift_type.getPointer()});
    }

    default:
      break;
    }
  }
  return CompilerType();
}

static CompilerType DeclToType(swift::Decl *decl) {
  if (swift::ValueDecl *value_decl =
          swift::dyn_cast_or_null<swift::ValueDecl>(decl))
    return ValueDeclToType(value_decl);
  return {};
}

static SwiftASTContext::TypeOrDecl DeclToTypeOrDecl(swift::Decl *decl) {
  if (decl) {
    switch (decl->getKind()) {
    case swift::DeclKind::BuiltinTuple:
    case swift::DeclKind::Import:
    case swift::DeclKind::Extension:
    case swift::DeclKind::PatternBinding:
    case swift::DeclKind::TopLevelCode:
    case swift::DeclKind::GenericTypeParam:
    case swift::DeclKind::AssociatedType:
    case swift::DeclKind::EnumElement:
    case swift::DeclKind::EnumCase:
    case swift::DeclKind::Param:
    case swift::DeclKind::Macro:
    case swift::DeclKind::MacroExpansion:     
    case swift::DeclKind::Module:
    case swift::DeclKind::Missing:
    case swift::DeclKind::MissingMember:
    case swift::DeclKind::Using:
      break;

    case swift::DeclKind::InfixOperator:
    case swift::DeclKind::PrefixOperator:
    case swift::DeclKind::PostfixOperator:
    case swift::DeclKind::PrecedenceGroup:
      return decl;

    case swift::DeclKind::TypeAlias: {
      swift::TypeAliasDecl *alias_decl =
          swift::cast<swift::TypeAliasDecl>(decl);
      swift::Type swift_type = swift::TypeAliasType::get(
          alias_decl, swift::Type(), llvm::ArrayRef<swift::Type>(),
          alias_decl->getUnderlyingType());
      return ToCompilerType(swift_type.getPointer());
    }
    case swift::DeclKind::OpaqueType: {
      swift::TypeDecl *type_decl = swift::cast<swift::TypeDecl>(decl);
      swift::Type swift_type = type_decl->getDeclaredInterfaceType();
      return ToCompilerType(swift_type.getPointer());
    }
    case swift::DeclKind::Enum:
    case swift::DeclKind::Struct:
    case swift::DeclKind::Class:
    case swift::DeclKind::Protocol: {
      swift::NominalTypeDecl *nominal_decl =
          swift::cast<swift::NominalTypeDecl>(decl);
      swift::Type swift_type = nominal_decl->getDeclaredType();
      return ToCompilerType(swift_type.getPointer());
    }

    case swift::DeclKind::Func:
    case swift::DeclKind::Var:
      return decl;

    case swift::DeclKind::Subscript:
    case swift::DeclKind::Constructor:
    case swift::DeclKind::Destructor:
      break;

    case swift::DeclKind::Accessor:
      break;
    }
  }
  return CompilerType();
}

size_t
SwiftASTContext::FindContainedTypeOrDecl(llvm::StringRef name,
                                         TypeOrDecl container_type_or_decl,
                                         TypesOrDecls &results, bool append) {
  VALID_OR_RETURN(0);

  if (!append)
    results.clear();
  size_t size_before = results.size();

  CompilerType container_type = container_type_or_decl.Apply<CompilerType>(
      [](CompilerType type) -> CompilerType { return type; },
      [](swift::Decl *decl) -> CompilerType {
        return DeclToType(decl);
      });

  if (!name.empty() &&
      container_type.GetTypeSystem().isa_and_nonnull<TypeSystemSwift>()) {
    swift::Type swift_type = GetSwiftTypeIgnoringErrors(container_type);
    if (!swift_type)
      return 0;
    swift::CanType swift_can_type(swift_type->getCanonicalType());
    swift::NominalType *nominal_type =
        swift_can_type->getAs<swift::NominalType>();
    if (!nominal_type)
      return 0;
    swift::NominalTypeDecl *nominal_decl = nominal_type->getDecl();
    llvm::ArrayRef<swift::ValueDecl *> decls = nominal_decl->lookupDirect(
        swift::DeclName(GetASTContext()->getIdentifier(name)));
    for (auto *decl : decls)
      results.emplace(DeclToTypeOrDecl(decl));
  }
  return results.size() - size_before;
}

CompilerType SwiftASTContext::FindType(const char *name,
                                       swift::ModuleDecl *swift_module) {
  VALID_OR_RETURN(CompilerType());

  std::set<CompilerType> search_results;

  FindTypes(name, swift_module, search_results, false);

  if (search_results.empty())
    return {};
  else
    return *search_results.begin();
}

std::optional<SwiftASTContext::TypeOrDecl>
SwiftASTContext::FindTypeOrDecl(const char *name,
                                swift::ModuleDecl *swift_module) {
  VALID_OR_RETURN(std::optional<SwiftASTContext::TypeOrDecl>());

  TypesOrDecls search_results;

  FindTypesOrDecls(name, swift_module, search_results, false);

  if (search_results.empty())
    return std::optional<SwiftASTContext::TypeOrDecl>();
  else
    return *search_results.begin();
}

size_t SwiftASTContext::FindTypes(const char *name,
                                  swift::ModuleDecl *swift_module,
                                  std::set<CompilerType> &results,
                                  bool append) {
  VALID_OR_RETURN(0);

  if (!append)
    results.clear();

  size_t before = results.size();

  TypesOrDecls types_or_decls_results;
  FindTypesOrDecls(name, swift_module, types_or_decls_results);

  for (const auto &result : types_or_decls_results) {
    CompilerType type = result.Apply<CompilerType>(
        [](CompilerType type) -> CompilerType { return type; },
        [](swift::Decl *decl) -> CompilerType {
          if (swift::ValueDecl *value_decl =
                  swift::dyn_cast_or_null<swift::ValueDecl>(decl)) {
            swift::Type swift_type = value_decl->getInterfaceType();
            if (swift_type)
              return ToCompilerType({swift_type->getMetatypeInstanceType()});
          }
          return CompilerType();
        });
    results.emplace(type);
  }

  return results.size() - before;
}

size_t SwiftASTContext::FindTypesOrDecls(const char *name,
                                         swift::ModuleDecl *swift_module,
                                         TypesOrDecls &results, bool append) {
  VALID_OR_RETURN(0);

  if (!append)
    results.clear();

  size_t before = results.size();

  if (name && name[0] && swift_module) {
    llvm::SmallVector<swift::ValueDecl *, 4> value_decls;
    swift::Identifier identifier(GetIdentifier(llvm::StringRef(name)));
    if (strchr(name, '.'))
      swift_module->lookupValue(identifier, swift::NLKind::QualifiedLookup,
                                value_decls);
    else
      swift_module->lookupValue(identifier, swift::NLKind::UnqualifiedLookup,
                                value_decls);
    if (identifier.isOperator()) {
      if (auto *op = swift_module->lookupPrefixOperator(identifier))
        results.emplace(DeclToTypeOrDecl(op));

      if (auto *op = swift_module->lookupInfixOperator(identifier).getSingle())
        results.emplace(DeclToTypeOrDecl(op));

      if (auto *op = swift_module->lookupPostfixOperator(identifier))
        results.emplace(DeclToTypeOrDecl(op));
    }
    if (auto *pg = swift_module->lookupPrecedenceGroup(identifier).getSingle())
      results.emplace(DeclToTypeOrDecl(pg));

    for (auto *decl : value_decls)
      results.emplace(DeclToTypeOrDecl(decl));
  }

  return results.size() - before;
}

size_t SwiftASTContext::FindType(const char *name,
                                 std::set<CompilerType> &results, bool append) {
  VALID_OR_RETURN(0);

  if (!append)
    results.clear();
  auto iter = m_swift_module_cache.begin(), end = m_swift_module_cache.end();

  size_t count = 0;

  std::function<void(swift::ModuleDecl &)> lookup_func =
      [this, name, &results, &count](swift::ModuleDecl &module) -> void {
    CompilerType candidate(this->FindType(name, &module));
    if (candidate) {
      ++count;
      results.insert(candidate);
    }
  };

  for (; iter != end; iter++)
    lookup_func(const_cast<swift::ModuleDecl &>(iter->second));

  if (m_scratch_module)
    lookup_func(*m_scratch_module);

  return count;
}

CompilerType SwiftASTContext::ImportType(CompilerType &type, Status &error) {
  VALID_OR_RETURN(CompilerType());

  auto ast_context = GetASTContext();
  if (!ast_context)
    return CompilerType();

  auto ts = type.GetTypeSystem();
  auto swift_ast_ctx = ts.dyn_cast_or_null<SwiftASTContext>();

  if (!swift_ast_ctx && (!ts || !ts.isa_and_nonnull<TypeSystemSwift>())) {
    error = Status::FromErrorString(
        "Can't import clang type into a Swift ASTContext.");
    return CompilerType();
  } else if (swift_ast_ctx.get() == this) {
    // This is the same AST context, so the type is already imported.
    return type;
  }

  // For now we're going to do this all using mangled names.  If we
  // find that is too slow, we can use the TypeBase * in the
  // CompilerType to match this to the version of the type we got from
  // the mangled name in the original swift::ASTContext.
  ConstString mangled_name(type.GetMangledTypeName());
  if (!mangled_name)
    return {};
  if (ts.isa_and_nonnull<TypeSystemSwiftTypeRef>())
    if (auto my_ts_tr = GetTypeSystemSwiftTypeRef())
      return my_ts_tr->GetTypeFromMangledTypename(mangled_name);
  swift::TypeBase *our_type_base =
      m_mangled_name_to_type_map.lookup(mangled_name.GetCString());
  if (our_type_base)
    return ToCompilerType({our_type_base});
  CompilerType our_type(GetTypeFromMangledTypename(mangled_name));
  if (error.Success())
    return our_type;
  return {};
}

swift::IRGenDebugInfoLevel SwiftASTContext::GetGenerateDebugInfo() {
  return GetIRGenOptions().DebugInfoLevel;
}

swift::PrintOptions SwiftASTContext::GetUserVisibleTypePrintingOptions(
    bool print_help_if_available) {
  swift::PrintOptions print_options;
  print_options.SynthesizeSugarOnTypes = true;
  print_options.VarInitializers = true;
  print_options.TypeDefinitions = true;
  print_options.PrintGetSetOnRWProperties = true;
  print_options.SkipImplicit = false;
  print_options.PreferTypeRepr = true;
  print_options.FunctionDefinitions = true;
  print_options.FullyQualifiedTypesIfAmbiguous = true;
  print_options.FullyQualifiedTypes = true;
  print_options.ExplodePatternBindingDecls = false;
  print_options.PrintDocumentationComments = print_help_if_available;
  return print_options;
}

void SwiftASTContext::SetGenerateDebugInfo(swift::IRGenDebugInfoLevel b) {
  GetIRGenOptions().DebugInfoLevel = b;
}

llvm::TargetOptions *SwiftASTContext::getTargetOptions() {
  if (m_target_options_ap.get() == NULL) {
    m_target_options_ap.reset(new llvm::TargetOptions());
  }
  return m_target_options_ap.get();
}

swift::ModuleDecl *SwiftASTContext::GetScratchModule() {
  VALID_OR_RETURN(nullptr);

  if (m_scratch_module == nullptr) {
    ThreadSafeASTContext ast_ctx = GetASTContext();
    m_scratch_module = swift::ModuleDecl::createEmpty(
      ast_ctx->getIdentifier("__lldb_scratch_module"), **ast_ctx);
  }
  return m_scratch_module;
}

swift::Lowering::TypeConverter *SwiftASTContext::GetSILTypes() {
  VALID_OR_RETURN(nullptr);

  if (m_sil_types_ap.get() == NULL)
    m_sil_types_ap.reset(
        new swift::Lowering::TypeConverter(*GetScratchModule()));

  return m_sil_types_ap.get();
}

swift::SILModule *SwiftASTContext::GetSILModule() {
  VALID_OR_RETURN(nullptr);

  if (m_sil_module_ap.get() == NULL)
    m_sil_module_ap = swift::SILModule::createEmptyModule(
        GetScratchModule(), *GetSILTypes(), GetSILOptions());
  return m_sil_module_ap.get();
}

swift::irgen::IRGenerator &
SwiftASTContext::GetIRGenerator(swift::IRGenOptions &opts,
                                swift::SILModule &module) {
  if (m_ir_generator_ap.get() == nullptr) {
    m_ir_generator_ap.reset(new swift::irgen::IRGenerator(opts, module));
  }

  return *m_ir_generator_ap.get();
}

swift::irgen::IRGenModule &SwiftASTContext::GetIRGenModule() {
  VALID_OR_RETURN(*m_ir_gen_module_ap);

  llvm::call_once(m_ir_gen_module_once, [this]() {
    // Make sure we have a good ClangImporter.
    GetClangImporter();

    swift::IRGenOptions &ir_gen_opts = GetIRGenOptions();

    std::string error_str;
    llvm::Triple llvm_triple = GetTriple();
    const llvm::Target *llvm_target =
        llvm::TargetRegistry::lookupTarget(llvm_triple.str(), error_str);

    llvm::CodeGenOptLevel optimization_level = llvm::CodeGenOptLevel::None;

    // Create a target machine.
    llvm::TargetMachine *target_machine = llvm_target->createTargetMachine(
        llvm_triple.str(),
        "generic", // cpu
        "",        // features
        *getTargetOptions(),
        llvm::Reloc::Static, // TODO verify with Sean, Default went away
        std::nullopt, optimization_level);
    if (target_machine) {
      // Set the module's string representation.
      const llvm::DataLayout data_layout = target_machine->createDataLayout();

      swift::SILModule *sil_module = GetSILModule();
      if (sil_module != nullptr) {
        swift::irgen::IRGenerator &ir_generator =
            GetIRGenerator(ir_gen_opts, *sil_module);
        swift::PrimarySpecificPaths PSPs =
            GetCompilerInvocation()
                .getFrontendOptions()
                .InputsAndOutputs.getPrimarySpecificPathsForAtMostOnePrimary();

        std::lock_guard<std::recursive_mutex> global_context_locker(
            IRExecutionUnit::GetLLVMGlobalContextMutex());
        m_ir_gen_module_ap.reset(new swift::irgen::IRGenModule(
            ir_generator, ir_generator.createTargetMachine(), nullptr,
            ir_gen_opts.ModuleName, PSPs.OutputFilename,
            PSPs.MainInputFilenameForDebugInfo, ""));
        llvm::Module *llvm_module = m_ir_gen_module_ap->getModule();
        llvm_module->setDataLayout(data_layout.getStringRepresentation());
        llvm_module->setTargetTriple(llvm_triple);
      }
    }
  });
  return *m_ir_gen_module_ap;
}

CompilerType
SwiftASTContext::CreateTupleType(const std::vector<TupleElement> &elements) {
  VALID_OR_RETURN(CompilerType());

  Status error;
  if (elements.size() == 0)
    return ToCompilerType({GetASTContext()->TheEmptyTupleType});
  else {
    std::vector<swift::TupleTypeElt> tuple_elems;
    for (const TupleElement &element : elements) {
      if (auto swift_type = GetSwiftTypeIgnoringErrors(element.element_type)) {
        if (element.element_name.IsEmpty()) {
          tuple_elems.push_back(swift::TupleTypeElt(swift_type));
        } else {
          tuple_elems.push_back(swift::TupleTypeElt(
              swift_type, GetASTContext()->getIdentifier(
                              element.element_name.GetCString())));
        }
      } else
        return {};
    }
    llvm::ArrayRef<swift::TupleTypeElt> fields(tuple_elems);
    ThreadSafeASTContext ast_ctx = GetASTContext();
    return ToCompilerType(
        {swift::TupleType::get(fields, **ast_ctx)});
  }
}

bool SwiftASTContext::IsTupleType(lldb::opaque_compiler_type_t type) {
  VALID_OR_RETURN(false);

  auto swift_type = GetSwiftType(type);
  return llvm::isa<::swift::TupleType>(swift_type);
}

std::optional<TypeSystemSwift::NonTriviallyManagedReferenceKind>
SwiftASTContext::GetNonTriviallyManagedReferenceKind(
    lldb::opaque_compiler_type_t type) {
  VALID_OR_RETURN({});
  if (swift::CanType swift_can_type = GetCanonicalSwiftType(type)) {
    const swift::TypeKind type_kind = swift_can_type->getKind();
    switch (type_kind) {
    default:
      return {};
    case swift::TypeKind::UnmanagedStorage:
      return NonTriviallyManagedReferenceKind::eUnmanaged;
    case swift::TypeKind::UnownedStorage:
      return NonTriviallyManagedReferenceKind::eUnowned;
    case swift::TypeKind::WeakStorage:
      return NonTriviallyManagedReferenceKind::eWeak;
    }
  }
  return {};
}

CompilerType SwiftASTContext::CreateGenericTypeParamType(
    unsigned int depth, unsigned int index,
    swift::Mangle::ManglingFlavor flavor) {
  ThreadSafeASTContext ast_ctx = GetASTContext();
  auto ast_flavor = GetManglingFlavor();
  assert(flavor == ast_flavor && "Requested flavor and ast flavor diverge!");
  return ToCompilerType(
      swift::GenericTypeParamType::getType(depth, index, **ast_ctx));
}

CompilerType SwiftASTContext::GetErrorType() {
  VALID_OR_RETURN(CompilerType());

  ThreadSafeASTContext swift_ctx = GetASTContext();
  if (swift_ctx) {
    // Getting the error type requires the Stdlib module be loaded,
    // but doesn't cause it to be loaded.  Do that here.
    swift_ctx->getStdlibModule(true);
    swift::NominalTypeDecl *error_type_decl = GetASTContext()->getErrorDecl();
    if (error_type_decl) {
      auto *error_type = error_type_decl->getDeclaredType().getPointer();
      return ToCompilerType({error_type});
    }
  }
  return {};
}

SwiftASTContext *SwiftASTContext::GetSwiftASTContext(swift::ASTContext *ast) {
  SwiftASTContext *swift_ast = GetASTMap().Lookup(ast);
  return swift_ast;
}

uint32_t SwiftASTContext::GetPointerByteSize() {
  return m_pointer_byte_size;
}

bool SwiftASTContext::HasDiagnostics() const {
  assert(m_diagnostic_consumer_ap);
  return (
      static_cast<StoringDiagnosticConsumer *>(m_diagnostic_consumer_ap.get())
          ->HasDiagnostics());
}
bool SwiftASTContext::HasClangImporterErrors() const {
  assert(m_diagnostic_consumer_ap);
  return (
      static_cast<StoringDiagnosticConsumer *>(m_diagnostic_consumer_ap.get())
          ->NumClangErrors() != 0);
}

void SwiftASTContext::AddDiagnostic(lldb::Severity severity,
                                    llvm::StringRef message) {
  HEALTH_LOG_PRINTF("%s", message.str().c_str());

  if (auto target_sp = GetTargetWP().lock()) {
    auto debugger_id = target_sp->GetDebugger().GetID();
    switch (severity) {
    case eSeverityError:
      Debugger::ReportError(message.str(), debugger_id);
      break;
    case eSeverityWarning:
    case eSeverityInfo:
      Debugger::ReportWarning(message.str(), debugger_id);
      break;
    }
  }

  assert(m_diagnostic_consumer_ap);
  if (!m_diagnostic_consumer_ap.get())
    return;

  auto diagnostic = std::make_unique<Diagnostic>(
      eDiagnosticOriginLLDB, LLDB_INVALID_COMPILER_ID,
      DiagnosticDetail{{}, severity, message.str(), message.str()});
  static_cast<StoringDiagnosticConsumer *>(m_diagnostic_consumer_ap.get())
      ->AddDiagnostic(std::move(diagnostic));
}

bool SwiftASTContext::HasFatalErrors(swift::ASTContext *ast_context) {
  return (ast_context && ast_context->Diags.hasFatalErrorOccurred());
}

void SwiftASTContext::PrintDiagnostics(DiagnosticManager &diagnostic_manager,
                                       uint32_t bufferID, uint32_t first_line,
                                       uint32_t last_line) const {
  // VALID_OR_RETURN cannot be used here here since would exit on error.
  if (!GetASTContext()) {
    RaiseFatalError("Swift compiler could not be initialized");
    return;
  }

  // Forward diagnostics into diagnostic_manager.
  // If there is a fatal error, also copy the error into m_fatal_errors.
  assert(m_diagnostic_consumer_ap);
  auto &diags =
      *static_cast<StoringDiagnosticConsumer *>(m_diagnostic_consumer_ap.get());
  if (GetASTContext()->Diags.hasFatalErrorOccurred() &&
      !m_reported_fatal_error) {
    DiagnosticManager fatal_diagnostics;
    diags.PrintDiagnostics(fatal_diagnostics, {}, bufferID, first_line,
                           last_line);
    if (fatal_diagnostics.Diagnostics().size())
      RaiseFatalError(fatal_diagnostics.GetString());
    else
      RaiseFatalError("Unknown fatal error occurred.");

    m_reported_fatal_error = true;

    for (const DiagnosticList::value_type &fatal_diagnostic :
         fatal_diagnostics.Diagnostics()) {
      // FIXME: Need to add a CopyDiagnostic operation for copying
      //        diagnostics from one manager to another.
      diagnostic_manager.AddDiagnostic(
          fatal_diagnostic->GetMessage(), fatal_diagnostic->GetSeverity(),
          fatal_diagnostic->getKind(), fatal_diagnostic->GetCompilerID());
    }
  } else {
    diags.PrintDiagnostics(diagnostic_manager, {}, bufferID, first_line,
                           last_line);
  }
}

void SwiftASTContextForExpressions::ModulesDidLoad(ModuleList &module_list) {
  ClearModuleDependentCaches();

  // Scan the new modules for Swift contents and try to import it if
  // safe, otherwise poison this context.
  TargetSP target_sp = GetTargetWP().lock();
  if (!target_sp)
    return;

  bool discover_implicit_search_paths =
      target_sp->GetSwiftDiscoverImplicitSearchPaths();
  bool use_all_compiler_flags = target_sp->GetUseAllCompilerFlags();
  unsigned num_images = module_list.GetSize();
  for (size_t mi = 0; mi != num_images; ++mi) {
    std::vector<swift::PluginSearchOption> plugin_search_options;
    std::vector<std::pair<std::string, bool>> module_search_paths;
    std::vector<std::pair<std::string, bool>> framework_search_paths;
    std::vector<std::string> extra_clang_args;
    lldb::ModuleSP module_sp = module_list.GetModuleAtIndex(mi);
    if (!module_sp)
      continue;
    std::string error;
    StringRef module_filter;
    ProcessModule(*module_sp, m_description, discover_implicit_search_paths,
                  use_all_compiler_flags,
                  target_sp->GetExecutableModulePointer() == module_sp.get(),
                  module_filter, GetTriple(), plugin_search_options,
                  module_search_paths, framework_search_paths, extra_clang_args,
                  error);
    if (!error.empty())
      AddDiagnostic(eSeverityError, error);
    // If the use-all-compiler-flags setting is enabled, the
    // expression context is supposed to merge all search paths
    // from all dylibs.
    // TODO: Maybe we should also do this for compiler plugins?
    if (use_all_compiler_flags && !extra_clang_args.empty()) {
      // We cannot reconfigure ClangImporter after its creation.
      // Instead poison the SwiftASTContext so it gets recreated.
      RaiseFatalError(
          "New Swift image added: " + module_sp->GetFileSpec().GetPath() +
          "ClangImporter needs to be reinitialized.");
      HEALTH_LOG_PRINTF(
          "New Swift image added: %s. ClangImporter needs to be reinitialized.",
          module_sp->GetFileSpec().GetPath().c_str());
    }

    // Scan the dylib for .swiftast sections.
    std::vector<std::string> module_names;
    RegisterSectionModules(*module_sp, module_names);
  }
}

void SwiftASTContext::ClearModuleDependentCaches() {
  m_negative_type_cache.Clear();
}

void SwiftASTContext::LogConfiguration(bool repl, bool playground) {
  // It makes no sense to call VALID_OR_RETURN here. We specifically
  // want the logs in the error case!
  HEALTH_LOG_PRINTF("(SwiftASTContext*)%p:", static_cast<void *>(this));

  auto ast_context = GetASTContext();
  if (!ast_context) {
    HEALTH_LOG_PRINTF("  (no AST context)");
    return;
  }
  if (repl)
    HEALTH_LOG_PRINTF("  REPL                             : true");
  if (playground)
    HEALTH_LOG_PRINTF("  Playground                       : true");
  HEALTH_LOG_PRINTF("  Swift/C++ interop                : %s",
                    ast_context->LangOpts.EnableCXXInterop ? "on" : "off");
  HEALTH_LOG_PRINTF("  Swift/Objective-C interop        : %s",
                    ast_context->LangOpts.EnableObjCInterop ? "on" : "off");

  HEALTH_LOG_PRINTF("  Architecture                     : %s",
                    ast_context->LangOpts.Target.getTriple().c_str());
  HEALTH_LOG_PRINTF(
      "  SDK path                         : %s",
      ast_context->SearchPathOpts.getSDKPath().str().c_str());
  HEALTH_LOG_PRINTF(
      "  Runtime resource path            : %s",
      ast_context->SearchPathOpts.RuntimeResourcePath.c_str());
  HEALTH_LOG_PRINTF("  Runtime library paths            : (%llu items)",
                    (unsigned long long)ast_context->SearchPathOpts
                        .RuntimeLibraryPaths.size());

  for (const auto &runtime_library_path :
       ast_context->SearchPathOpts.RuntimeLibraryPaths)
    HEALTH_LOG_PRINTF("    %s", runtime_library_path.c_str());

  HEALTH_LOG_PRINTF("  Runtime library import paths     : (%llu items)",
                    (unsigned long long)ast_context->SearchPathOpts
                        .getRuntimeLibraryImportPaths()
                        .size());

  for (const auto &runtime_import_path :
       ast_context->SearchPathOpts.getRuntimeLibraryImportPaths())
    HEALTH_LOG_PRINTF("    %s", runtime_import_path.c_str());

  HEALTH_LOG_PRINTF("  Framework search paths           : (%llu items)",
                    (unsigned long long)ast_context->SearchPathOpts
                        .getFrameworkSearchPaths()
                        .size());
  for (const auto &framework_search_path :
       ast_context->SearchPathOpts.getFrameworkSearchPaths())
    HEALTH_LOG_PRINTF("    %s", framework_search_path.Path.c_str());

  HEALTH_LOG_PRINTF("  Import search paths              : (%llu items)",
                    (unsigned long long)ast_context->SearchPathOpts
                        .getImportSearchPaths()
                        .size());
  for (const auto &import_search_path :
       ast_context->SearchPathOpts.getImportSearchPaths())
    HEALTH_LOG_PRINTF("    %s", import_search_path.Path.c_str());

  swift::ClangImporterOptions &clang_importer_options =
      GetClangImporterOptions();

  if (!clang_importer_options.BridgingHeader.empty())
    HEALTH_LOG_PRINTF("  Bridging Header               : %s",
                      clang_importer_options.BridgingHeader.c_str());
  if (auto *expr_ctx = llvm::dyn_cast<SwiftASTContextForExpressions>(this))
    HEALTH_LOG_PRINTF("  Explicit modules              : %s",
                      expr_ctx->HasExplicitModules() ? "true" : "false");

  HEALTH_LOG_PRINTF(
      "  Extra clang arguments            : (%llu items)",
      (unsigned long long)clang_importer_options.ExtraArgs.size());
  for (std::string &extra_arg : clang_importer_options.ExtraArgs)
    HEALTH_LOG_PRINTF("    %s", extra_arg.c_str());

  HEALTH_LOG_PRINTF("  Plugin search options            : (%llu items)",
                    (unsigned long long)ast_context->SearchPathOpts
                        .PluginSearchOpts.size());
  for (auto &elem : ast_context->SearchPathOpts.PluginSearchOpts) {
    if (auto *opt =
            elem.dyn_cast<swift::PluginSearchOption::LoadPluginLibrary>()) {
      HEALTH_LOG_PRINTF("    -load-plugin-library %s",
                        opt->LibraryPath.c_str());
      continue;
    }
    if (auto *opt =
            elem.dyn_cast<swift::PluginSearchOption::LoadPluginExecutable>()) {
      HEALTH_LOG_PRINTF("    -load-plugin-executable %s#%s",
                        opt->ExecutablePath.c_str(),
                        [](auto path_names) -> std::string {
                          std::string s;
                          llvm::raw_string_ostream os(s);
                          llvm::interleaveComma(path_names, os);
                          return os.str();
                        }(opt->ModuleNames)
                                                   .c_str());
      continue;
    }
    if (auto *opt = elem.dyn_cast<swift::PluginSearchOption::PluginPath>()) {
      HEALTH_LOG_PRINTF("    -plugin-path %s", opt->SearchPath.c_str());
      continue;
    }
    if (auto *opt =
            elem.dyn_cast<swift::PluginSearchOption::ExternalPluginPath>())
      HEALTH_LOG_PRINTF("    -external-plugin-path %s#%s",
                        opt->SearchPath.c_str(), opt->ServerPath.c_str());
  }
}

bool SwiftASTContext::HasTarget() {
  lldb::TargetWP empty_wp, target_wp = GetTargetWP();

  // If either call to "std::weak_ptr::owner_before(...) value returns
  // true, this indicates that m_section_wp once contained (possibly
  // still does) a reference to a valid shared pointer. This helps us
  // know if we had a valid reference to a target which is now invalid
  // because the target was deleted.
  return empty_wp.owner_before(target_wp) || target_wp.owner_before(empty_wp);
}

bool SwiftASTContext::CheckProcessChanged() {
  if (HasTarget()) {
    TargetSP target_sp(GetTargetWP().lock());
    if (target_sp) {
      Process *process = target_sp->GetProcessSP().get();
      if (m_process == NULL) {
        if (process)
          m_process = process;
      } else {
        if (m_process != process)
          return true;
      }
    }
  }
  return false;
}

void SwiftASTContext::AddDebuggerClient(
    swift::DebuggerClient *debugger_client) {
  m_debugger_clients.push_back(
      std::unique_ptr<swift::DebuggerClient>(debugger_client));
}

#ifndef NDEBUG
bool SwiftASTContext::Verify(opaque_compiler_type_t type) {
  // Manual casting to avoid construction a temporary CompilerType
  // that would recursively trigger another call to Verify().
  swift::TypeBase *swift_type = reinterpret_cast<swift::TypeBase *>(type);
  // Check that type is a Swift type and belongs this AST context.
  return !swift_type || &swift_type->getASTContext() == *GetASTContext();
}
#endif

bool SwiftASTContext::IsArrayType(opaque_compiler_type_t type,
                                  CompilerType *element_type_ptr,
                                  uint64_t *size, bool *is_incomplete) {
  VALID_OR_RETURN_CHECK_TYPE(type, false);

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  swift::BoundGenericStructType *struct_type =
      swift_can_type->getAs<swift::BoundGenericStructType>();
  if (struct_type) {
    swift::StructDecl *struct_decl = struct_type->getDecl();
    llvm::StringRef name = struct_decl->getName().get();
    // This is sketchy, but it matches the behavior of GetArrayElementType().
    if (name != "Array" && name != "ContiguousArray" && name != "ArraySlice")
      return false;
    if (!struct_decl->getModuleContext()->isStdlibModule())
      return false;
    const llvm::ArrayRef<swift::Type> &args = struct_type->getGenericArgs();
    if (args.size() != 1)
      return false;
    if (is_incomplete)
      *is_incomplete = true;
    if (size)
      *size = 0;
    if (element_type_ptr)
      *element_type_ptr = ToCompilerType(args[0].getPointer());
    return true;
  }

  return false;
}

bool SwiftASTContext::IsAggregateType(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, false);

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  auto referent_type = swift_can_type->getReferenceStorageReferent();
  return (referent_type->is<swift::TupleType>() ||
          referent_type->is<swift::BuiltinVectorType>() ||
          referent_type->getAnyNominal());
}

bool SwiftASTContext::IsFunctionType(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, false);

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  const swift::TypeKind type_kind = swift_can_type->getKind();
  switch (type_kind) {
  case swift::TypeKind::Function:
  case swift::TypeKind::GenericFunction:
    return true;
  case swift::TypeKind::SILFunction:
    return false; // TODO: is this correct?
  default:
    return false;
  }
}

size_t
SwiftASTContext::GetNumberOfFunctionArguments(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, 0);

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  auto func = swift::dyn_cast_or_null<swift::AnyFunctionType>(swift_can_type);
  if (func) {
    return func.getParams().size();
  }

  return 0;
}

CompilerType
SwiftASTContext::GetFunctionArgumentAtIndex(opaque_compiler_type_t type,
                                            const size_t index) {
  VALID_OR_RETURN_CHECK_TYPE(type, CompilerType());

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  auto func = swift::dyn_cast<swift::AnyFunctionType>(swift_can_type);
  if (func) {
    auto params = func.getParams();
    if (index < params.size()) {
      auto param = params[index];
      return {weak_from_this(), param.getParameterType().getPointer()};
    }
  }
  return {};
}

bool SwiftASTContext::IsFunctionPointerType(opaque_compiler_type_t type) {
  return IsFunctionType(type); // FIXME: think about this
}

bool SwiftASTContext::IsPointerType(opaque_compiler_type_t type,
                                    CompilerType *pointee_type) {
  VALID_OR_RETURN(false);

  if (type) {
    swift::CanType swift_can_type(GetCanonicalSwiftType(type));
    auto referent_type = swift_can_type->getReferenceStorageReferent();
    return (referent_type->is<swift::BuiltinRawPointerType>() ||
            referent_type->is<swift::BuiltinNativeObjectType>() ||
            referent_type->is<swift::BuiltinUnsafeValueBufferType>() ||
            referent_type->is<swift::BuiltinBridgeObjectType>());
  }

  if (pointee_type)
    pointee_type->Clear();
  return false;
}

bool SwiftASTContext::IsPointerOrReferenceType(opaque_compiler_type_t type,
                                               CompilerType *pointee_type) {
  return IsPointerType(type, pointee_type) ||
         IsReferenceType(type, pointee_type, nullptr);
}

bool SwiftASTContext::IsReferenceType(opaque_compiler_type_t type,
                                      CompilerType *pointee_type,
                                      bool *is_rvalue) {
  if (type) {
    swift::CanType swift_can_type(GetCanonicalSwiftType(type));
    const swift::TypeKind type_kind = swift_can_type->getKind();
    switch (type_kind) {
    case swift::TypeKind::LValue:
      if (pointee_type)
        *pointee_type = GetNonReferenceType(type);
      return true;
    default:
      break;
    }
  }
  if (pointee_type)
    pointee_type->Clear();
  return false;
}

bool SwiftASTContext::IsDefined(opaque_compiler_type_t type) {
  return type != nullptr;
}

bool SwiftASTContext::IsPossibleDynamicType(opaque_compiler_type_t type,
                                            CompilerType *dynamic_pointee_type,
                                            bool check_cplusplus,
                                            bool check_objc) {
  VALID_OR_RETURN_CHECK_TYPE(type, false);

  auto can_type = GetCanonicalSwiftType(type);
  if (!can_type)
    return false;

  if (can_type->getClassOrBoundGenericClass() ||
      can_type->isAnyExistentialType())
    return true;

  if (!IsImportedType(type) &&
      (swift::isa<swift::EnumType>(can_type) ||
       swift::isa<swift::BoundGenericEnumType>(can_type)))
    return true;

  // Dynamic Self types are resolved inside DoArchetypeBindingForType(),
  // right before the actual archetype binding.
  if (can_type->hasDynamicSelfType())
    return true;

  if (can_type->hasArchetype() || can_type->hasOpaqueArchetype() ||
      can_type->hasTypeParameter())
    return true;

  if (can_type == GetASTContext()->TheRawPointerType)
    return true;
  if (can_type == GetASTContext()->TheNativeObjectType)
    return true;
  if (can_type == GetASTContext()->TheBridgeObjectType)
    return true;

  if (auto *bound_type =
          llvm::dyn_cast<swift::BoundGenericType>(can_type.getPointer())) {
    for (auto generic_arg : bound_type->getGenericArgs()) {
      if (IsPossibleDynamicType(generic_arg.getPointer(), dynamic_pointee_type,
                                check_cplusplus, check_objc))
        return true;
    }
  }

  if (dynamic_pointee_type)
    dynamic_pointee_type->Clear();
  return false;
}

bool SwiftASTContext::IsTypedefType(opaque_compiler_type_t type) {
  if (!type)
    return false;
  swift::Type swift_type(GetSwiftType(type));
  return swift::isa<swift::TypeAliasType>(swift_type.getPointer());
}

bool SwiftASTContext::IsVoidType(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, false);

  return type == GetASTContext()->TheEmptyTupleType.getPointer();
}

bool SwiftASTContext::IsGenericType(const CompilerType &compiler_type) {
  if (auto swift_ast_ctx =
          compiler_type.GetTypeSystem().dyn_cast_or_null<SwiftASTContext>()) {
    if (swift::Type swift_type =
            swift_ast_ctx->GetSwiftTypeIgnoringErrors(compiler_type))
      return swift_type->hasTypeParameter();
  } else
    return compiler_type.GetTypeInfo() & eTypeIsGenericTypeParam;
  return false;
}

static CompilerType
BindGenericTypeParameters(CompilerType type, ExecutionContextScope *exe_scope) {
  if (!exe_scope)
    return type;
  auto *frame = exe_scope->CalculateStackFrame().get();
  auto *runtime = SwiftLanguageRuntime::Get(exe_scope->CalculateProcess());
  if (!frame || !runtime)
    return type;
  ExecutionContext exe_ctx;
  exe_scope->CalculateExecutionContext(exe_ctx);
  if (auto bound = llvm::expectedToOptional(
                       runtime->BindGenericTypeParameters(*frame, type))
                       .value_or(CompilerType()))
    return bound;
  return type;
}

bool SwiftASTContext::IsErrorType(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, false);
  ProtocolInfo protocol_info;
  if (GetProtocolTypeInfo({weak_from_this(), type}, protocol_info))
    return protocol_info.m_is_errortype;
  return false;
}

CompilerType SwiftASTContext::GetReferentType(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, CompilerType());

  swift::Type swift_type = GetSwiftType(type);
  if (!swift_type)
    return {};
  swift::TypeBase *swift_typebase = swift_type.getPointer();
  if (swift_type && llvm::isa<swift::WeakStorageType>(swift_typebase))
    return {weak_from_this(), type};

  auto ref_type = swift_type->getReferenceStorageReferent();
  return ToCompilerType({ref_type});
}

CompilerType
SwiftASTContext::GetStaticSelfType(lldb::opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, CompilerType());

  swift::Type swift_type = GetSwiftType(type);
  if (auto *dyn_self =
          llvm::dyn_cast_or_null<swift::DynamicSelfType>(swift_type))
    return ToCompilerType({dyn_self->getSelfType().getPointer()});
  return {weak_from_this(), type};
}

bool SwiftASTContext::GetProtocolTypeInfo(const CompilerType &type,
                                          ProtocolInfo &protocol_info) {
  if (swift::CanType swift_can_type = GetCanonicalSwiftType(type)) {
    if (!swift_can_type.isExistentialType())
      return false;

    swift::ExistentialLayout layout = swift_can_type.getExistentialLayout();
    protocol_info.m_is_class_only = layout.requiresClass();
    protocol_info.m_is_objc = layout.isObjC();
    protocol_info.m_is_anyobject = layout.isAnyObject();
    protocol_info.m_is_errortype = layout.isErrorExistential();

    if (auto superclass = layout.explicitSuperclass) {
      protocol_info.m_superclass = ToCompilerType({superclass.getPointer()});
    }

    unsigned num_witness_tables = 0;
    unsigned num_protocols = 0;
    for (auto protoDecl : layout.getProtocols()) {
      // Ignore invertible protocols like Copyable entirely. They're marker
      // protocols that are not mangled into generic signatures. Only their
      // absence is mangled.
      // FIXME: this should probably be filtering all marker protocols,
      //  including Sendable, since marker protocols lack a witness table.
      if (protoDecl->getInvertibleProtocolKind())
        continue;

      num_protocols++;

      if (!protoDecl->isObjC())
        num_witness_tables++;
    }

    protocol_info.m_num_protocols = num_protocols;

    if (layout.isErrorExistential()) {
      // Error existential -- instance pointer only.
      protocol_info.m_num_payload_words = 0;
      protocol_info.m_num_storage_words = 1;
    } else if (layout.requiresClass()) {
      // Class-constrained existential -- instance pointer plus
      // witness tables.
      protocol_info.m_num_payload_words = 0;
      protocol_info.m_num_storage_words = 1 + num_witness_tables;
    } else {
      // Opaque existential -- three words of inline storage, metadata
      // and witness tables.
      protocol_info.m_num_payload_words = swift::NumWords_ValueBuffer;
      protocol_info.m_num_storage_words =
          swift::NumWords_ValueBuffer + 1 + num_witness_tables;
    }

    return true;
  }

  return false;
}

CompilerType
SwiftASTContext::GetTypeRefType(lldb::opaque_compiler_type_t type) {
  if (auto ts = GetTypeSystemSwiftTypeRef())
    return ts->GetTypeFromMangledTypename(GetMangledTypeName(type));
  return {};
}

//----------------------------------------------------------------------
// Type Completion
//----------------------------------------------------------------------

ConstString SwiftASTContext::GetTypeName(opaque_compiler_type_t type,
                                         bool BaseOnly) {
  VALID_OR_RETURN_CHECK_TYPE(
      type, ConstString("<invalid Swift context or opaque type>"));
  std::string type_name;
  swift::Type swift_type(GetSwiftType(type));

  swift::Type normalized_type =
      swift_type.transformRec([](swift::Type type) -> swift::Type {
        if (swift::SyntaxSugarType *syntax_sugar_type =
                swift::dyn_cast<swift::SyntaxSugarType>(type.getPointer())) {
          return syntax_sugar_type->getSinglyDesugaredType();
        }
        return type;
      });

  swift::PrintOptions print_options;
  print_options.FullyQualifiedTypes = true;
  print_options.SynthesizeSugarOnTypes = false;
  type_name = normalized_type.getString(print_options);
  return ConstString(type_name);
}

/// Build a dictionary of Archetype names that appear in \p type.
static llvm::DenseMap<swift::CanType, swift::Identifier>
GetArchetypeNames(swift::Type swift_type, swift::ASTContext &ast_ctx,
                  const SymbolContext *sc,
                  swift::Mangle::ManglingFlavor flavor) {
  llvm::DenseMap<swift::CanType, swift::Identifier> dict;

  assert(&swift_type->getASTContext() == &ast_ctx);
  if (!sc)
    return dict;

  llvm::DenseMap<std::pair<uint64_t, uint64_t>, StringRef> names;
  SwiftLanguageRuntime::GetGenericParameterNamesForFunction(*sc, nullptr,
                                                            flavor, names);
  swift_type.visit([&](swift::Type type) {
    if (!type->isTypeParameter() || dict.count(type->getCanonicalType()))
      return;
    auto *param = type->getAs<swift::GenericTypeParamType>();
    if (!param)
      return;
    auto it = names.find({param->getDepth(), param->getIndex()});
    if (it != names.end()) {
      swift::Identifier ident = ast_ctx.getIdentifier(it->second);
      dict.insert({type->getCanonicalType(), ident});
    }
  });
  return dict;
}

ConstString SwiftASTContext::GetDisplayTypeName(opaque_compiler_type_t type,
                                                const SymbolContext *sc) {
  VALID_OR_RETURN_CHECK_TYPE(
      type, ConstString("<invalid Swift context or opaque type>"));
  std::string type_name(GetTypeName(type, false).AsCString(""));
  if (type) {
    swift::Type swift_type(GetSwiftType(type));
    swift::PrintOptions print_options;
    print_options.FullyQualifiedTypes = false;
    print_options.SynthesizeSugarOnTypes = true;
    print_options.FullyQualifiedTypesIfAmbiguous = true;
    ThreadSafeASTContext ast_ctx = GetASTContext();
    auto dict =
        GetArchetypeNames(swift_type, **ast_ctx, sc, GetManglingFlavor());
    print_options.AlternativeTypeNames = &dict;
    type_name = swift_type.getString(print_options);
  }
  return ConstString(type_name);
}

uint32_t
SwiftASTContext::GetTypeInfo(opaque_compiler_type_t type,
                             CompilerType *pointee_or_element_clang_type) {
  VALID_OR_RETURN_CHECK_TYPE(type, 0);

  if (pointee_or_element_clang_type)
    pointee_or_element_clang_type->Clear();

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  const swift::TypeKind type_kind = swift_can_type->getKind();
  uint32_t swift_flags = eTypeIsSwift;
  if (swift_can_type->hasUnboundGenericType() ||
      swift_can_type->hasTypeParameter())
    swift_flags |= eTypeHasUnboundGeneric;
  if (swift_can_type->hasDynamicSelfType())
    swift_flags |= eTypeHasDynamicSelf;
  switch (type_kind) {
  case swift::TypeKind::BuiltinDefaultActorStorage:
  case swift::TypeKind::BuiltinExecutor:
  case swift::TypeKind::BuiltinJob:
  case swift::TypeKind::BuiltinNonDefaultDistributedActorStorage:
  case swift::TypeKind::BuiltinRawUnsafeContinuation:
  case swift::TypeKind::BuiltinTuple:
  case swift::TypeKind::BuiltinUnboundGeneric:
  case swift::TypeKind::ElementArchetype:
  case swift::TypeKind::Error:
  case swift::TypeKind::ErrorUnion:
  case swift::TypeKind::InOut:
  case swift::TypeKind::Integer:
  case swift::TypeKind::Locatable:
  case swift::TypeKind::Module:
  case swift::TypeKind::ExistentialArchetype:
  case swift::TypeKind::ParameterizedProtocol:
  case swift::TypeKind::Placeholder:
  case swift::TypeKind::PrimaryArchetype:
  case swift::TypeKind::SILBlockStorage:
  case swift::TypeKind::SILBox:
  case swift::TypeKind::SILFunction:
  case swift::TypeKind::SILMoveOnlyWrapped:
  case swift::TypeKind::SILToken:
  case swift::TypeKind::TypeVariable:
  case swift::TypeKind::VariadicSequence:
    LOG_PRINTF(GetLog(LLDBLog::Types), "Unexpected type: %s",
               swift_can_type.getString().c_str());
    assert(false && "Internal compiler type");
    break;
  case swift::TypeKind::UnboundGeneric:
  case swift::TypeKind::OpaqueTypeArchetype:
    break;

  case swift::TypeKind::GenericFunction:
  case swift::TypeKind::Function:
    swift_flags |= eTypeIsPointer | eTypeHasValue;
    break;
  case swift::TypeKind::BuiltinInteger:
  case swift::TypeKind::BuiltinIntegerLiteral:
    swift_flags |=
        eTypeIsBuiltIn | eTypeHasValue | eTypeIsScalar | eTypeIsInteger;
    break;
  case swift::TypeKind::BuiltinFloat:
    swift_flags |=
        eTypeIsBuiltIn | eTypeHasValue | eTypeIsScalar | eTypeIsFloat;
    break;
  case swift::TypeKind::BuiltinRawPointer:
    swift_flags |= eTypeIsBuiltIn | eTypeHasChildren | eTypeIsPointer |
                   eTypeIsScalar | eTypeHasValue;
    break;
  case swift::TypeKind::BuiltinNativeObject:
    swift_flags |= eTypeIsBuiltIn | eTypeHasChildren | eTypeIsPointer |
                   eTypeIsScalar | eTypeHasValue;
    break;
  case swift::TypeKind::BuiltinBridgeObject:
    swift_flags |= eTypeIsBuiltIn | eTypeHasChildren | eTypeIsPointer |
                   eTypeIsScalar | eTypeHasValue | eTypeIsObjC;
    break;
  case swift::TypeKind::BuiltinUnsafeValueBuffer:
    swift_flags |=
        eTypeIsBuiltIn | eTypeIsPointer | eTypeIsScalar | eTypeHasValue;
    break;
  case swift::TypeKind::BuiltinFixedArray:
    return eTypeIsBuiltIn | eTypeHasChildren;
  case swift::TypeKind::BuiltinVector:
    // TODO: OR in eTypeIsFloat or eTypeIsInteger as needed
    return eTypeIsBuiltIn | eTypeHasChildren | eTypeIsVector;
    break;

  case swift::TypeKind::Tuple:
    swift_flags |= eTypeHasChildren | eTypeIsTuple;
    break;
  case swift::TypeKind::UnmanagedStorage:
  case swift::TypeKind::UnownedStorage:
  case swift::TypeKind::WeakStorage:
    swift_flags |= ToCompilerType(swift_can_type->getReferenceStorageReferent())
                       .GetTypeInfo(pointee_or_element_clang_type);
    break;
  case swift::TypeKind::BoundGenericEnum:
    LLVM_FALLTHROUGH;
  case swift::TypeKind::Enum: {
    SwiftEnumDescriptor *cached_enum_info = GetCachedEnumInfo(type);
    if (cached_enum_info)
      swift_flags |= eTypeHasValue | eTypeIsEnumeration | eTypeHasChildren;
    else
      swift_flags |= eTypeIsEnumeration;
  } break;

  case swift::TypeKind::BoundGenericStruct:
    LLVM_FALLTHROUGH;
  case swift::TypeKind::Struct:
    if (auto *ndecl = swift_can_type.getAnyNominal())
      if (llvm::dyn_cast_or_null<clang::EnumDecl>(ndecl->getClangDecl())) {
        swift_flags |= eTypeHasChildren | eTypeIsEnumeration | eTypeHasValue;
        break;
      }

    swift_flags |= eTypeHasChildren | eTypeIsStructUnion;
    break;

  case swift::TypeKind::BoundGenericClass:
    LLVM_FALLTHROUGH;
  case swift::TypeKind::Class:
    swift_flags |= eTypeHasChildren | eTypeIsClass | eTypeHasValue |
                   eTypeInstanceIsPointer;
    break;

  case swift::TypeKind::Protocol: {
    auto *protocol =
        llvm::cast<swift::ProtocolType>(swift_can_type.getPointer());
    swift::ProtocolDecl *decl = protocol->getDecl();
    if (decl->isObjC())
      swift_flags |= eTypeIsObjC | eTypeHasValue;
    LLVM_FALLTHROUGH;
  }
  case swift::TypeKind::ProtocolComposition:
  case swift::TypeKind::Existential: {
    swift_flags |= eTypeHasChildren | eTypeIsStructUnion | eTypeIsProtocol;
  } break;
  case swift::TypeKind::ExistentialMetatype:
    swift_flags |= eTypeIsProtocol;
    LLVM_FALLTHROUGH;
  case swift::TypeKind::Metatype:
    swift_flags |= eTypeIsMetatype | eTypeHasValue;
    break;

  case swift::TypeKind::DependentMember:
  case swift::TypeKind::GenericTypeParam:
    swift_flags |= eTypeHasValue | eTypeIsScalar | eTypeIsPointer |
                   eTypeIsGenericTypeParam;
    break;

  case swift::TypeKind::LValue:
    if (pointee_or_element_clang_type)
      *pointee_or_element_clang_type = GetNonReferenceType(type);
    swift_flags |= eTypeHasChildren | eTypeIsReference | eTypeHasValue;
    break;
  case swift::TypeKind::DynamicSelf:
    swift_flags |= eTypeHasValue;
    break;

  case swift::TypeKind::Pack:
  case swift::TypeKind::PackElement:
  case swift::TypeKind::PackExpansion:
  case swift::TypeKind::PackArchetype:
  case swift::TypeKind::BuiltinPackIndex:
  case swift::TypeKind::SILPack:
    swift_flags |= eTypeIsPack;
    break;

  case swift::TypeKind::Optional:
  case swift::TypeKind::TypeAlias:
  case swift::TypeKind::Dictionary:
  case swift::TypeKind::ArraySlice:
  case swift::TypeKind::InlineArray:
    assert(false && "Not a canonical type");
    break;
  }
  return swift_flags;
}

lldb::TypeClass SwiftASTContext::GetTypeClass(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, lldb::eTypeClassInvalid);

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  const swift::TypeKind type_kind = swift_can_type->getKind();
  switch (type_kind) {
  case swift::TypeKind::BuiltinDefaultActorStorage:
  case swift::TypeKind::BuiltinNonDefaultDistributedActorStorage:
  case swift::TypeKind::BuiltinExecutor:
  case swift::TypeKind::BuiltinJob:
  case swift::TypeKind::BuiltinPackIndex:    
  case swift::TypeKind::BuiltinTuple:
  case swift::TypeKind::BuiltinUnboundGeneric:
  case swift::TypeKind::Locatable:
  case swift::TypeKind::Integer:
  case swift::TypeKind::Pack:
  case swift::TypeKind::PackElement:
  case swift::TypeKind::PackExpansion:
  case swift::TypeKind::ParameterizedProtocol:
  case swift::TypeKind::Placeholder:
  case swift::TypeKind::SILBlockStorage:
  case swift::TypeKind::SILBox:
  case swift::TypeKind::SILMoveOnlyWrapped:
  case swift::TypeKind::SILPack:
  case swift::TypeKind::SILToken:
  case swift::TypeKind::PackArchetype:
  case swift::TypeKind::VariadicSequence:
    assert(false && "Internal compiler type");
    break;
  case swift::TypeKind::BuiltinBridgeObject:
  case swift::TypeKind::BuiltinFloat:
  case swift::TypeKind::BuiltinInteger:
  case swift::TypeKind::BuiltinIntegerLiteral:
  case swift::TypeKind::BuiltinNativeObject:
  case swift::TypeKind::BuiltinRawPointer:
  case swift::TypeKind::BuiltinRawUnsafeContinuation:
  case swift::TypeKind::BuiltinUnsafeValueBuffer:
    return lldb::eTypeClassBuiltin;
  case swift::TypeKind::BuiltinVector:
    return lldb::eTypeClassVector;
  case swift::TypeKind::BuiltinFixedArray:
  case swift::TypeKind::Tuple:
    return lldb::eTypeClassArray;
  case swift::TypeKind::UnmanagedStorage:
  case swift::TypeKind::UnownedStorage:
  case swift::TypeKind::WeakStorage:
    return ToCompilerType(swift_can_type->getReferenceStorageReferent())
        .GetTypeClass();
  case swift::TypeKind::Enum:
  case swift::TypeKind::BoundGenericEnum:
    return lldb::eTypeClassUnion;
  case swift::TypeKind::Struct:
  case swift::TypeKind::BoundGenericStruct:
    return lldb::eTypeClassStruct;
  case swift::TypeKind::Class:
  case swift::TypeKind::BoundGenericClass:
    return lldb::eTypeClassClass;
  case swift::TypeKind::GenericTypeParam:
  case swift::TypeKind::DependentMember:
  case swift::TypeKind::Protocol:
  case swift::TypeKind::ProtocolComposition:
  case swift::TypeKind::Existential:
  case swift::TypeKind::Metatype:
  case swift::TypeKind::Module:
  case swift::TypeKind::PrimaryArchetype:
  case swift::TypeKind::ElementArchetype:
  case swift::TypeKind::OpaqueTypeArchetype:
  case swift::TypeKind::ExistentialArchetype:
  case swift::TypeKind::UnboundGeneric:
  case swift::TypeKind::TypeVariable:
  case swift::TypeKind::ErrorUnion:
  case swift::TypeKind::ExistentialMetatype:
  case swift::TypeKind::DynamicSelf:
  case swift::TypeKind::Error:
    return lldb::eTypeClassOther;
  case swift::TypeKind::Function:
  case swift::TypeKind::GenericFunction:
  case swift::TypeKind::SILFunction:
    return lldb::eTypeClassFunction;
  case swift::TypeKind::InOut:
  case swift::TypeKind::LValue:
    return lldb::eTypeClassReference;

  case swift::TypeKind::Optional:
  case swift::TypeKind::TypeAlias:
  case swift::TypeKind::Dictionary:
  case swift::TypeKind::ArraySlice:
  case swift::TypeKind::InlineArray:
    assert(false && "Not a canonical type");
    break;
  }

  return lldb::eTypeClassOther;
}

//----------------------------------------------------------------------
// Creating related types
//----------------------------------------------------------------------

CompilerType
SwiftASTContext::GetArrayElementType(opaque_compiler_type_t type,
                                     ExecutionContextScope *exe_scope) {
  VALID_OR_RETURN_CHECK_TYPE(type, CompilerType());

  CompilerType element_type;
  swift::CanType swift_type(GetCanonicalSwiftType(type));
  // There are a couple of structs that mean "Array" in Swift:
  // Array<T>
  // ContiguousArray<T>
  // Slice<T>
  // Treat them as arrays for convenience sake.
  swift::BoundGenericStructType *boundGenericStructType(
      swift_type->getAs<swift::BoundGenericStructType>());
  if (boundGenericStructType) {
    auto args = boundGenericStructType->getGenericArgs();
    swift::StructDecl *decl = boundGenericStructType->getDecl();
    if (args.size() == 1 && decl->getModuleContext()->isStdlibModule()) {
      const char *declname = decl->getName().get();
      if (0 == strcmp(declname, "ContiguousArray") ||
          0 == strcmp(declname, "Array") ||
          0 == strcmp(declname, "ArraySlice")) {
        assert(*GetASTContext() == &args[0].getPointer()->getASTContext());
        element_type = ToCompilerType(args[0].getPointer());
      }
    }
  }

  return element_type;
}

CompilerType SwiftASTContext::GetCanonicalType(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, CompilerType());

  return ToCompilerType({GetCanonicalSwiftType(type).getPointer()});
}

CompilerType
SwiftASTContext::GetInstanceType(opaque_compiler_type_t type,
                                 ExecutionContextScope *exe_scope) {
  VALID_OR_RETURN_CHECK_TYPE(type, CompilerType());

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  assert((&swift_can_type->getASTContext() == *GetASTContext()) &&
         "input type belongs to different SwiftASTContext");
  auto metatype_type = swift::dyn_cast<swift::AnyMetatypeType>(swift_can_type);
  if (metatype_type)
    return ToCompilerType({metatype_type.getInstanceType().getPointer()});

  return ToCompilerType({GetSwiftType(type)});
}

CompilerType
SwiftASTContext::GetFullyUnqualifiedType(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, CompilerType());

  return ToCompilerType({GetSwiftType(type)});
}

int SwiftASTContext::GetFunctionArgumentCount(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, 0);
  return GetNumberOfFunctionArguments(type);
}

CompilerType
SwiftASTContext::GetFunctionArgumentTypeAtIndex(opaque_compiler_type_t type,
                                                size_t idx) {
  VALID_OR_RETURN_CHECK_TYPE(type, CompilerType());
  return GetFunctionArgumentAtIndex(type, idx);
}

CompilerType
SwiftASTContext::GetFunctionReturnType(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, CompilerType());

  auto func =
      swift::dyn_cast<swift::AnyFunctionType>(GetCanonicalSwiftType(type));
  if (func)
    return ToCompilerType({func.getResult().getPointer()});

  return {};
}

size_t SwiftASTContext::GetNumMemberFunctions(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, 0);
  size_t num_functions = 0;
  swift::CanType swift_can_type(GetCanonicalSwiftType(type));

  auto nominal_decl = swift_can_type.getAnyNominal();
  if (nominal_decl) {
    auto iter = nominal_decl->getMembers().begin();
    auto end = nominal_decl->getMembers().end();
    for (; iter != end; iter++) {
      switch (iter->getKind()) {
      case swift::DeclKind::Constructor:
      case swift::DeclKind::Destructor:
      case swift::DeclKind::Func:
        num_functions += 1;
        break;
      default:
        break;
      }
    }
  }

  return num_functions;
}

TypeMemberFunctionImpl
SwiftASTContext::GetMemberFunctionAtIndex(opaque_compiler_type_t type,
                                          size_t idx) {
  VALID_OR_RETURN_CHECK_TYPE(type, TypeMemberFunctionImpl());

  std::string name("");
  CompilerType result_type;
  MemberFunctionKind kind(MemberFunctionKind::eMemberFunctionKindUnknown);
  swift::AbstractFunctionDecl *the_decl_we_care_about = nullptr;

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));

  auto nominal_decl = swift_can_type.getAnyNominal();
  if (nominal_decl) {
    auto iter = nominal_decl->getMembers().begin();
    auto end = nominal_decl->getMembers().end();
    for (; iter != end; iter++) {
      auto decl_kind = iter->getKind();
      switch (decl_kind) {
      case swift::DeclKind::Constructor:
      case swift::DeclKind::Destructor:
      case swift::DeclKind::Func: {
        if (idx == 0) {
          swift::AbstractFunctionDecl *abstract_func_decl =
              llvm::dyn_cast_or_null<swift::AbstractFunctionDecl>(*iter);
          if (abstract_func_decl) {
            switch (decl_kind) {
            case swift::DeclKind::Constructor:
              name.clear();
              kind = lldb::eMemberFunctionKindConstructor;
              the_decl_we_care_about = abstract_func_decl;
              break;
            case swift::DeclKind::Destructor:
              name.clear();
              kind = lldb::eMemberFunctionKindDestructor;
              the_decl_we_care_about = abstract_func_decl;
              break;
            case swift::DeclKind::Func:
            default: {
              swift::FuncDecl *func_decl =
                  llvm::dyn_cast<swift::FuncDecl>(*iter);
              if (func_decl) {
                if (func_decl->getBaseIdentifier().empty())
                  name.clear();
                else
                  name.assign(func_decl->getBaseIdentifier().get());
                if (func_decl->isStatic())
                  kind = lldb::eMemberFunctionKindStaticMethod;
                else
                  kind = lldb::eMemberFunctionKindInstanceMethod;
                the_decl_we_care_about = func_decl;
              }
            }
            }
            result_type = ToCompilerType(
                abstract_func_decl->getInterfaceType().getPointer());
          }
        } else
          --idx;
      } break;
      default:
        break;
      }
    }
  }

  if (the_decl_we_care_about && (kind != eMemberFunctionKindUnknown))
    return TypeMemberFunctionImpl(
        result_type, CompilerDecl(this, the_decl_we_care_about), name, kind);

  return TypeMemberFunctionImpl();
}

CompilerType SwiftASTContext::GetPointeeType(opaque_compiler_type_t type) {
  return {};
}

CompilerType SwiftASTContext::GetPointerType(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, CompilerType());

  auto swift_type = GetSwiftType(type);
  auto pointer_type =
      swift_type->wrapInPointer(swift::PointerTypeKind::PTK_UnsafePointer);
  if (pointer_type)
    return ToCompilerType(pointer_type);

  return {};
}

CompilerType SwiftASTContext::GetTypedefedType(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, CompilerType());

  swift::Type swift_type(GetSwiftType(type));
  swift::TypeAliasType *name_alias_type =
      swift::dyn_cast<swift::TypeAliasType>(swift_type.getPointer());
  if (name_alias_type) {
    return ToCompilerType({name_alias_type->getSinglyDesugaredType()});
  }

  return {};
}

CompilerType SwiftASTContext::GetUnboundType(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, CompilerType());

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  swift::BoundGenericType *bound_generic_type =
      swift_can_type->getAs<swift::BoundGenericType>();
  if (bound_generic_type) {
    swift::NominalTypeDecl *nominal_type_decl = bound_generic_type->getDecl();
    if (nominal_type_decl)
      return ToCompilerType({nominal_type_decl->getDeclaredType()});
  }

  return ToCompilerType({GetSwiftType(type)});
}

//----------------------------------------------------------------------
// Exploring the type
//----------------------------------------------------------------------

const swift::irgen::TypeInfo *
SwiftASTContext::GetSwiftTypeInfo(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, nullptr);

  auto &irgen_module = GetIRGenModule();
  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  swift::SILType swift_sil_type = irgen_module.getLoweredType(swift_can_type);
  return &irgen_module.getTypeInfo(swift_sil_type);
}

const swift::irgen::FixedTypeInfo *
SwiftASTContext::GetSwiftFixedTypeInfo(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, nullptr);

  const swift::irgen::TypeInfo *type_info = GetSwiftTypeInfo(type);
  if (type_info) {
    if (type_info->isFixedSize())
      return swift::cast<const swift::irgen::FixedTypeInfo>(type_info);
  }
  return nullptr;
}

bool SwiftASTContext::IsFixedSize(CompilerType compiler_type) {
  VALID_OR_RETURN(false);
  const swift::irgen::FixedTypeInfo *type_info =
      GetSwiftFixedTypeInfo(compiler_type.GetOpaqueQualType());
  if (type_info)
    return type_info->isFixedSize();
  return false;
}

llvm::Expected<uint64_t>
SwiftASTContext::GetBitSize(opaque_compiler_type_t type,
                            ExecutionContextScope *exe_scope) {
  VALID_OR_RETURN_CHECK_TYPE(type, llvm::createStringError(g_invalid_context));

  // If the type has type parameters, bind them first.
  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  if (swift_can_type->hasTypeParameter()) {
    if (!exe_scope)
      return llvm::createStringError(
          "Cannot resolve generic type without a running process");
    ExecutionContext exe_ctx;
    exe_scope->CalculateExecutionContext(exe_ctx);
    CompilerType bound_type =
        BindGenericTypeParameters({weak_from_this(), type}, exe_scope);

    // Check that the type has been bound successfully -- and if not,
    // log the event and bail out to avoid an infinite loop.
    swift::CanType swift_bound_type(GetCanonicalSwiftType(bound_type));
    if (!swift_bound_type || swift_bound_type->hasTypeParameter())
      return llvm::createStringError("Cannot bind type: %s",
                                     bound_type.GetTypeName().AsCString(""));

    // Note that the bound type may be in a different AST context.
    auto size_or_err = bound_type.GetBitSize(exe_scope);
    if (!size_or_err && swift_bound_type->hasAnyPack()) {
      // FIXME: We miss reflection support for pack types.
      LLDB_LOG_ERROR(GetLog(LLDBLog::Types), size_or_err.takeError(),
                     "Could not get size of pack type. Ignoring: {0}");
      return GetPointerByteSize() * 3 * 8;
    }
    return size_or_err;
  }

  // LLDB's ValueObject subsystem expects functions to be a single
  // pointer in size to print them correctly. This is not true for
  // swift (where functions aren't necessarily a single pointer in
  // size), so we need to work around the limitation here.
  if (swift_can_type->getKind() == swift::TypeKind::Function)
    return GetPointerByteSize() * 8;

  // Ask the static type type system.
  const swift::irgen::FixedTypeInfo *fixed_type_info =
      GetSwiftFixedTypeInfo(type);
  if (fixed_type_info)
    return fixed_type_info->getFixedSize().getValue() * 8;

  // Ask the dynamic type system.
  if (!exe_scope)
    return llvm::createStringError(
        "Cannot determine size of type without a running process");

  if (auto *runtime = SwiftLanguageRuntime::Get(exe_scope->CalculateProcess()))
    return runtime->GetBitSize({weak_from_this(), type}, exe_scope);
  return llvm::createStringError("Cannot determine size of type");
}

std::optional<uint64_t>
SwiftASTContext::GetByteStride(opaque_compiler_type_t type,
                               ExecutionContextScope *exe_scope) {
  VALID_OR_RETURN_CHECK_TYPE(type, std::nullopt);

  // If the type has type parameters, bind them first.
  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  if (swift_can_type->hasTypeParameter()) {
    if (!exe_scope)
      return {};
    ExecutionContext exe_ctx;
    exe_scope->CalculateExecutionContext(exe_ctx);
    CompilerType bound_type =
        BindGenericTypeParameters({weak_from_this(), type}, exe_scope);

    // Check that the type has been bound successfully -- and if not,
    // log the event and bail out to avoid an infinite loop.
    swift::CanType swift_bound_type(GetCanonicalSwiftType(bound_type));
    if (swift_bound_type && swift_bound_type->hasTypeParameter()) {
      LOG_PRINTF(GetLog(LLDBLog::Types), "Can't bind type: %s",
                 bound_type.GetTypeName().AsCString());
      return {};
    }

    // Note thay the bound type may be in a different AST context.
    return bound_type.GetByteStride(exe_scope);
  }

  // Ask the static type type system.
  const swift::irgen::FixedTypeInfo *fixed_type_info =
      GetSwiftFixedTypeInfo(type);
  if (fixed_type_info)
    return fixed_type_info->getFixedStride().getValue();

  // Ask the dynamic type system.
  if (!exe_scope)
    return {};
  if (auto *runtime = SwiftLanguageRuntime::Get(exe_scope->CalculateProcess()))
    return runtime->GetByteStride({weak_from_this(), type});
  return {};
}

std::optional<size_t>
SwiftASTContext::GetTypeBitAlign(opaque_compiler_type_t type,
                                 ExecutionContextScope *exe_scope) {
  VALID_OR_RETURN_CHECK_TYPE(type, std::nullopt);

  // If the type has type parameters, bind them first.
  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  if (swift_can_type->hasTypeParameter()) {
    if (!exe_scope)
      return {};
    ExecutionContext exe_ctx;
    exe_scope->CalculateExecutionContext(exe_ctx);
    CompilerType bound_type =
        BindGenericTypeParameters({weak_from_this(), type}, exe_scope);

    // Check that the type has been bound successfully -- and if not,
    // log the event and bail out to avoid an infinite loop.
    swift::CanType swift_bound_type(GetCanonicalSwiftType(bound_type));
    if (swift_bound_type && swift_bound_type->hasTypeParameter()) {
      LOG_PRINTF(GetLog(LLDBLog::Types), "Can't bind type: %s",
                 bound_type.GetTypeName().AsCString());
      return {};

    }
    // Note thay the bound type may be in a different AST context.
    return bound_type.GetTypeBitAlign(exe_scope);
  }

  const swift::irgen::FixedTypeInfo *fixed_type_info =
      GetSwiftFixedTypeInfo(type);
  if (fixed_type_info)
    return fixed_type_info->getFixedAlignment().getValue() * 8;

  // Ask the dynamic type system.
  if (!exe_scope)
    return {};
  if (auto *runtime = SwiftLanguageRuntime::Get(exe_scope->CalculateProcess()))
    return runtime->GetBitAlignment({weak_from_this(), type}, exe_scope);
  return {};
}

lldb::Encoding SwiftASTContext::GetEncoding(opaque_compiler_type_t type,
                                            uint64_t &count) {
  VALID_OR_RETURN_CHECK_TYPE(type, lldb::eEncodingInvalid);

  count = 1;
  swift::CanType swift_can_type(GetCanonicalSwiftType(type));

  const swift::TypeKind type_kind = swift_can_type->getKind();
  switch (type_kind) {
  case swift::TypeKind::BuiltinDefaultActorStorage:
  case swift::TypeKind::BuiltinNonDefaultDistributedActorStorage:
  case swift::TypeKind::BuiltinExecutor:
  case swift::TypeKind::BuiltinJob:
  case swift::TypeKind::BuiltinTuple:
  case swift::TypeKind::BuiltinRawUnsafeContinuation:
  case swift::TypeKind::Error:
  case swift::TypeKind::InOut:
  case swift::TypeKind::Locatable:
  case swift::TypeKind::Module:
  case swift::TypeKind::BuiltinPackIndex:
  case swift::TypeKind::Pack:
  case swift::TypeKind::PackElement:
  case swift::TypeKind::PackExpansion:
  case swift::TypeKind::SILPack:
  case swift::TypeKind::ParameterizedProtocol:
  case swift::TypeKind::Placeholder:
  case swift::TypeKind::SILBlockStorage:
  case swift::TypeKind::SILBox:
  case swift::TypeKind::SILMoveOnlyWrapped:
  case swift::TypeKind::SILFunction:
  case swift::TypeKind::SILToken:
  case swift::TypeKind::PackArchetype:
  case swift::TypeKind::TypeVariable:
  case swift::TypeKind::ErrorUnion:
  case swift::TypeKind::VariadicSequence:
    break;
  case swift::TypeKind::BuiltinInteger:
  case swift::TypeKind::BuiltinIntegerLiteral:
    return lldb::eEncodingSint; // TODO: detect if an integer is unsigned
  case swift::TypeKind::BuiltinFloat:
    return lldb::eEncodingIEEE754; // TODO: detect if an integer is unsigned

  case swift::TypeKind::PrimaryArchetype:
  case swift::TypeKind::ExistentialArchetype:
  case swift::TypeKind::ElementArchetype:
  case swift::TypeKind::OpaqueTypeArchetype:
  case swift::TypeKind::BuiltinRawPointer:
  case swift::TypeKind::BuiltinNativeObject:
  case swift::TypeKind::BuiltinUnsafeValueBuffer:
  case swift::TypeKind::BuiltinBridgeObject:
  case swift::TypeKind::Class: // Classes are pointers in swift...
  case swift::TypeKind::BoundGenericClass:
  case swift::TypeKind::GenericTypeParam:
  case swift::TypeKind::DependentMember:
  case swift::TypeKind::Integer:
    return lldb::eEncodingUint;

  case swift::TypeKind::BuiltinFixedArray:
  case swift::TypeKind::BuiltinUnboundGeneric:
  case swift::TypeKind::BuiltinVector:
  case swift::TypeKind::Tuple:
    break;
  case swift::TypeKind::UnmanagedStorage:
  case swift::TypeKind::UnownedStorage:
  case swift::TypeKind::WeakStorage:
    return ToCompilerType(swift_can_type->getReferenceStorageReferent())
        .GetEncoding(count);
    break;

  case swift::TypeKind::ExistentialMetatype:
  case swift::TypeKind::Metatype:
    return lldb::eEncodingUint;

  case swift::TypeKind::GenericFunction:
  case swift::TypeKind::Function:
    return lldb::eEncodingUint;

  case swift::TypeKind::Enum:
  case swift::TypeKind::BoundGenericEnum:
    break;

  case swift::TypeKind::Struct:
  case swift::TypeKind::Protocol:
  case swift::TypeKind::ProtocolComposition:
  case swift::TypeKind::Existential:
    break;
  case swift::TypeKind::LValue:
    return lldb::eEncodingUint;
  case swift::TypeKind::UnboundGeneric:
  case swift::TypeKind::BoundGenericStruct:
  case swift::TypeKind::DynamicSelf:
    break;

  case swift::TypeKind::Optional:
  case swift::TypeKind::TypeAlias:
  case swift::TypeKind::Dictionary:
  case swift::TypeKind::ArraySlice:
  case swift::TypeKind::InlineArray:
    assert(false && "Not a canonical type");
    break;
  }
  count = 0;
  return lldb::eEncodingInvalid;
}

llvm::Expected<uint32_t>
SwiftASTContext::GetNumChildren(opaque_compiler_type_t type,
                                bool omit_empty_base_classes,
                                const ExecutionContext *exe_ctx) {
  VALID_OR_RETURN_CHECK_TYPE(type, llvm::createStringError(g_invalid_context));

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));

  const swift::TypeKind type_kind = swift_can_type->getKind();
  switch (type_kind) {
  case swift::TypeKind::BuiltinBridgeObject:
  case swift::TypeKind::BuiltinDefaultActorStorage:
  case swift::TypeKind::BuiltinExecutor:
  case swift::TypeKind::BuiltinFloat:
  case swift::TypeKind::BuiltinInteger:
  case swift::TypeKind::BuiltinIntegerLiteral:
  case swift::TypeKind::BuiltinJob:
  case swift::TypeKind::BuiltinNativeObject:
  case swift::TypeKind::BuiltinNonDefaultDistributedActorStorage:
  case swift::TypeKind::BuiltinPackIndex:
  case swift::TypeKind::BuiltinRawPointer:
  case swift::TypeKind::BuiltinRawUnsafeContinuation:
  case swift::TypeKind::BuiltinTuple:
  case swift::TypeKind::BuiltinUnboundGeneric:
  case swift::TypeKind::BuiltinUnsafeValueBuffer:
  case swift::TypeKind::BuiltinVector:
  case swift::TypeKind::DependentMember:
  case swift::TypeKind::DynamicSelf:
  case swift::TypeKind::ElementArchetype:
  case swift::TypeKind::Error:
  case swift::TypeKind::ErrorUnion:
  case swift::TypeKind::ExistentialMetatype:
  case swift::TypeKind::Function:
  case swift::TypeKind::GenericFunction:
  case swift::TypeKind::GenericTypeParam:
  case swift::TypeKind::InOut:
  case swift::TypeKind::Integer:
  case swift::TypeKind::Locatable:
  case swift::TypeKind::Metatype:
  case swift::TypeKind::Module:
  case swift::TypeKind::OpaqueTypeArchetype:
  case swift::TypeKind::ExistentialArchetype:
  case swift::TypeKind::Pack:
  case swift::TypeKind::PackArchetype:
  case swift::TypeKind::PackElement:
  case swift::TypeKind::PackExpansion:
  case swift::TypeKind::ParameterizedProtocol:
  case swift::TypeKind::Placeholder:
  case swift::TypeKind::PrimaryArchetype:
  case swift::TypeKind::SILBlockStorage:
  case swift::TypeKind::SILBox:
  case swift::TypeKind::SILFunction:
  case swift::TypeKind::SILMoveOnlyWrapped:
  case swift::TypeKind::SILPack:
  case swift::TypeKind::SILToken:
  case swift::TypeKind::TypeVariable:
  case swift::TypeKind::VariadicSequence:
    break;
  case swift::TypeKind::UnmanagedStorage:
  case swift::TypeKind::UnownedStorage:
  case swift::TypeKind::WeakStorage:
    return ToCompilerType(swift_can_type->getReferenceStorageReferent())
        .GetNumChildren(omit_empty_base_classes, exe_ctx);

  case swift::TypeKind::Enum:
  case swift::TypeKind::BoundGenericEnum:
    return 0;

  case swift::TypeKind::BoundGenericStruct:
  case swift::TypeKind::BuiltinFixedArray:
  case swift::TypeKind::Struct:
  case swift::TypeKind::Tuple:
    return GetNumFields(type);

  case swift::TypeKind::Class:
  case swift::TypeKind::BoundGenericClass: {
    auto class_decl = swift_can_type->getClassOrBoundGenericClass();
    return (class_decl->hasSuperclass() ? 1 : 0) + GetNumFields(type);
  }

  case swift::TypeKind::Protocol:
  case swift::TypeKind::ProtocolComposition:
  case swift::TypeKind::Existential: {
    ProtocolInfo protocol_info;
    if (!GetProtocolTypeInfo(ToCompilerType(GetSwiftType(type)), protocol_info))
      break;

    return protocol_info.m_num_storage_words;
  }

  case swift::TypeKind::LValue: {
    swift::LValueType *lvalue_type =
        swift_can_type->castTo<swift::LValueType>();
    swift::TypeBase *deref_type = lvalue_type->getObjectType().getPointer();

    llvm::Expected<uint32_t> num_pointee_children =
        ToCompilerType(deref_type)
            .GetNumChildren(omit_empty_base_classes, exe_ctx);
    if (!num_pointee_children)
      return num_pointee_children;
    // If this type points to a simple type (or to a class), then it
    // has 1 child.
    if (*num_pointee_children == 0 || deref_type->getClassOrBoundGenericClass())
      return 1;
    return num_pointee_children;
  } break;

  case swift::TypeKind::UnboundGeneric:
    break;

  case swift::TypeKind::Optional:
  case swift::TypeKind::TypeAlias:
  case swift::TypeKind::Dictionary:
  case swift::TypeKind::ArraySlice:
  case swift::TypeKind::InlineArray:
    assert(false && "Not a canonical type");
    break;
  }

  return 0;
}

#pragma mark Aggregate Types

uint32_t
SwiftASTContext::GetNumDirectBaseClasses(opaque_compiler_type_t opaque_type) {
  if (!opaque_type)
    return 0;

  swift::CanType swift_can_type(GetCanonicalSwiftType(opaque_type));
  swift::ClassDecl *class_decl = swift_can_type->getClassOrBoundGenericClass();
  if (class_decl) {
    if (class_decl->hasSuperclass())
      return 1;
  }

  return 0;
}

uint32_t SwiftASTContext::GetNumFields(opaque_compiler_type_t type,
                                       ExecutionContext *exe_ctx) {
  VALID_OR_RETURN_CHECK_TYPE(type, 0);

  uint32_t count = 0;

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));

  const swift::TypeKind type_kind = swift_can_type->getKind();
  switch (type_kind) {
  case swift::TypeKind::BuiltinBridgeObject:
  case swift::TypeKind::BuiltinDefaultActorStorage:
  case swift::TypeKind::BuiltinExecutor:
  case swift::TypeKind::BuiltinFloat:
  case swift::TypeKind::BuiltinInteger:
  case swift::TypeKind::BuiltinIntegerLiteral:
  case swift::TypeKind::BuiltinJob:
  case swift::TypeKind::BuiltinNativeObject:
  case swift::TypeKind::BuiltinNonDefaultDistributedActorStorage:
  case swift::TypeKind::BuiltinPackIndex:
  case swift::TypeKind::BuiltinRawPointer:
  case swift::TypeKind::BuiltinRawUnsafeContinuation:
  case swift::TypeKind::BuiltinTuple:
  case swift::TypeKind::BuiltinUnboundGeneric:
  case swift::TypeKind::BuiltinUnsafeValueBuffer:
  case swift::TypeKind::BuiltinVector:
  case swift::TypeKind::Error:
  case swift::TypeKind::ErrorUnion:
  case swift::TypeKind::InOut:
  case swift::TypeKind::Integer:
  case swift::TypeKind::Locatable:
  case swift::TypeKind::Module:
  case swift::TypeKind::Pack:
  case swift::TypeKind::PackArchetype:
  case swift::TypeKind::PackElement:
  case swift::TypeKind::PackExpansion:
  case swift::TypeKind::ParameterizedProtocol:
  case swift::TypeKind::Placeholder:
  case swift::TypeKind::SILBlockStorage:
  case swift::TypeKind::SILBox:
  case swift::TypeKind::SILFunction:
  case swift::TypeKind::SILMoveOnlyWrapped:
  case swift::TypeKind::SILPack:
  case swift::TypeKind::SILToken:
  case swift::TypeKind::TypeVariable:
  case swift::TypeKind::VariadicSequence:
    break;
  case swift::TypeKind::UnmanagedStorage:
  case swift::TypeKind::UnownedStorage:
  case swift::TypeKind::WeakStorage:
    return ToCompilerType(swift_can_type->getReferenceStorageReferent())
        .GetNumFields();
  case swift::TypeKind::GenericTypeParam:
  case swift::TypeKind::DependentMember:
    break;

  case swift::TypeKind::Enum:
  case swift::TypeKind::BoundGenericEnum:
    return 0;

  case swift::TypeKind::Tuple:
    return swift::cast<swift::TupleType>(swift_can_type)->getNumElements();
  case swift::TypeKind::BuiltinFixedArray:
    return swift::cast<swift::BuiltinFixedArrayType>(swift_can_type)
        ->getFixedInhabitedSize()
        .value_or(0);

  case swift::TypeKind::Struct:
  case swift::TypeKind::Class:
  case swift::TypeKind::BoundGenericClass:
  case swift::TypeKind::BoundGenericStruct: {
    auto nominal = swift_can_type->getAnyNominal();
    // Imported unions don't have stored properties.
    if (auto *ntd =
            llvm::dyn_cast_or_null<swift::NominalTypeDecl>(nominal->getDecl()))
      if (auto *rd =
              llvm::dyn_cast_or_null<clang::RecordDecl>(ntd->getClangDecl()))
        if (rd->isUnion()) {
          swift::DeclRange ms = ntd->getMembers();
          return std::count_if(ms.begin(), ms.end(), [](swift::Decl *D) {
            return llvm::isa<swift::VarDecl>(D);
          });
        }
    return GetStoredProperties(nominal).size();
  }

  case swift::TypeKind::Protocol:
  case swift::TypeKind::ProtocolComposition:
  case swift::TypeKind::Existential:
    return llvm::expectedToStdOptional(
               GetNumChildren(type, /*omit_empty_base_classes=*/false, nullptr))
        .value_or(0);

  case swift::TypeKind::ExistentialMetatype:
  case swift::TypeKind::Metatype:
    return 0;

  case swift::TypeKind::ElementArchetype:
  case swift::TypeKind::OpaqueTypeArchetype:
  case swift::TypeKind::ExistentialArchetype:
  case swift::TypeKind::PrimaryArchetype:
  case swift::TypeKind::Function:
  case swift::TypeKind::GenericFunction:
  case swift::TypeKind::LValue:
  case swift::TypeKind::UnboundGeneric:
  case swift::TypeKind::DynamicSelf:
    break;

  case swift::TypeKind::Optional:
  case swift::TypeKind::TypeAlias:
  case swift::TypeKind::Dictionary:
  case swift::TypeKind::ArraySlice:
  case swift::TypeKind::InlineArray:
    assert(false && "Not a canonical type");
    break;
  }

  return count;
}

CompilerType SwiftASTContext::GetDirectBaseClassAtIndex(
    opaque_compiler_type_t opaque_type, size_t idx, uint32_t *bit_offset_ptr) {
  VALID_OR_RETURN_CHECK_TYPE(opaque_type, CompilerType());

  swift::CanType swift_can_type(GetCanonicalSwiftType(opaque_type));
  swift::ClassDecl *class_decl =
      swift_can_type->getClassOrBoundGenericClass();
  if (class_decl) {
    swift::Type base_class_type = class_decl->getSuperclass();
    if (base_class_type)
      return ToCompilerType({base_class_type.getPointer()});
  }

  return {};
}

/// Retrieve the printable name of a tuple element.
static std::string GetTupleElementName(const swift::TupleType *tuple_type,
                                       unsigned index,
                                       llvm::StringRef printed_index = "") {
  const auto &element = tuple_type->getElement(index);

  // Use the element name if there is one.
  if (!element.getName().empty())
    return element.getName().str().str();

  // If we know the printed index already, use that.
  if (!printed_index.empty())
    return printed_index.str();

  // Print the index and return that.
  std::string str;
  llvm::raw_string_ostream(str) << index;
  return str;
}

/// Retrieve the printable name of a type referenced as a superclass.
std::string
SwiftASTContext::GetSuperclassName(const CompilerType &superclass_type) {
  return GetUnboundType(superclass_type.GetOpaqueQualType())
      .GetTypeName()
      .AsCString("<no type name>");
}

/// Retrieve the type and name of a child of an existential type.
static std::pair<CompilerType, std::string> GetExistentialTypeChild(
    SwiftASTContext &swift_ast_ctx, swift::ASTContext &ast, CompilerType type,
    const SwiftASTContext::ProtocolInfo &protocol_info, unsigned idx) {
  assert(idx < protocol_info.m_num_storage_words &&
         "caller is responsible for validating index");

  // A payload word for a non-class, non-error existential.
  if (idx < protocol_info.m_num_payload_words) {
    std::string name;
    llvm::raw_string_ostream(name) << "payload_data_" << idx;

    auto raw_pointer = ast.TheRawPointerType;
    return {ToCompilerType(raw_pointer.getPointer()), std::move(name)};
  }

  // The instance for a class-bound existential.
  if (idx == 0 && protocol_info.m_is_class_only) {
    CompilerType class_type;
    if (protocol_info.m_superclass) {
      class_type = protocol_info.m_superclass;
    } else {
      auto raw_pointer = ast.TheRawPointerType;
      class_type = ToCompilerType(raw_pointer.getPointer());
    }

    return {class_type, "object"};
  }

  // The instance for an error existential.
  if (idx == 0 && protocol_info.m_is_errortype) {
    auto raw_pointer = ast.TheRawPointerType;
    return {ToCompilerType(raw_pointer.getPointer()), "error"};
  }

  // The metatype for a non-class, non-error existential.
  if (idx && idx == protocol_info.m_num_payload_words) {
    // The metatype for a non-class, non-error existential.
    auto any_metatype = swift::ExistentialMetatypeType::get(ast.TheAnyType);
    return {ToCompilerType(any_metatype), "metadata"};
  }

  // A witness table. Figure out which protocol it corresponds to.
  unsigned witness_table_idx = idx - protocol_info.m_num_payload_words - 1;
  swift::CanType swift_can_type(swift_ast_ctx.GetCanonicalSwiftType(type));
  swift::ExistentialLayout layout = swift_can_type.getExistentialLayout();

  std::string name;
  for (auto proto : layout.getProtocols()) {
    if (proto->isObjC())
      continue;

    if (witness_table_idx == 0) {
      name = "wtable";
      break;
    }
    --witness_table_idx;
  }

  auto raw_pointer = ast.TheRawPointerType;
  return {ToCompilerType(raw_pointer.getPointer()), std::move(name)};
}

CompilerType SwiftASTContext::GetFieldAtIndex(opaque_compiler_type_t type,
                                              size_t idx, std::string &name,
                                              uint64_t *bit_offset_ptr,
                                              uint32_t *bitfield_bit_size_ptr,
                                              bool *is_bitfield_ptr) {
  VALID_OR_RETURN_CHECK_TYPE(type, CompilerType());

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));

  const swift::TypeKind type_kind = swift_can_type->getKind();
  switch (type_kind) {
  case swift::TypeKind::BuiltinBridgeObject:
  case swift::TypeKind::BuiltinDefaultActorStorage:
  case swift::TypeKind::BuiltinExecutor:
  case swift::TypeKind::BuiltinFloat:
  case swift::TypeKind::BuiltinInteger:
  case swift::TypeKind::BuiltinIntegerLiteral:
  case swift::TypeKind::BuiltinJob:
  case swift::TypeKind::BuiltinNativeObject:
  case swift::TypeKind::BuiltinNonDefaultDistributedActorStorage:
  case swift::TypeKind::BuiltinPackIndex:
  case swift::TypeKind::BuiltinRawPointer:
  case swift::TypeKind::BuiltinRawUnsafeContinuation:
  case swift::TypeKind::BuiltinTuple:
  case swift::TypeKind::BuiltinUnboundGeneric:
  case swift::TypeKind::BuiltinUnsafeValueBuffer:
  case swift::TypeKind::BuiltinVector:
  case swift::TypeKind::Error:
  case swift::TypeKind::ErrorUnion:
  case swift::TypeKind::InOut:
  case swift::TypeKind::Integer:
  case swift::TypeKind::Locatable:
  case swift::TypeKind::Module:
  case swift::TypeKind::Pack:
  case swift::TypeKind::PackArchetype:
  case swift::TypeKind::PackElement:
  case swift::TypeKind::PackExpansion:
  case swift::TypeKind::ParameterizedProtocol:
  case swift::TypeKind::Placeholder:
  case swift::TypeKind::SILBlockStorage:
  case swift::TypeKind::SILBox:
  case swift::TypeKind::SILFunction:
  case swift::TypeKind::SILMoveOnlyWrapped:
  case swift::TypeKind::SILPack:
  case swift::TypeKind::SILToken:
  case swift::TypeKind::TypeVariable:
  case swift::TypeKind::VariadicSequence:
    break;
  case swift::TypeKind::UnmanagedStorage:
  case swift::TypeKind::UnownedStorage:
  case swift::TypeKind::WeakStorage:
    return ToCompilerType(swift_can_type->getReferenceStorageReferent())
        .GetFieldAtIndex(idx, name, bit_offset_ptr, bitfield_bit_size_ptr,
                         is_bitfield_ptr);
  case swift::TypeKind::GenericTypeParam:
  case swift::TypeKind::DependentMember:
    break;

  case swift::TypeKind::Enum:
  case swift::TypeKind::BoundGenericEnum:
    break;

  case swift::TypeKind::Tuple: {
    auto tuple_type = swift::cast<swift::TupleType>(swift_can_type);
    if (idx >= tuple_type->getNumElements())
      break;

    // We cannot reliably get layout information without an execution
    // context.
    if (bit_offset_ptr)
      *bit_offset_ptr = LLDB_INVALID_IVAR_OFFSET;
    if (bitfield_bit_size_ptr)
      *bitfield_bit_size_ptr = 0;
    if (is_bitfield_ptr)
      *is_bitfield_ptr = false;

    name = GetTupleElementName(tuple_type, idx);

    const auto &child = tuple_type->getElement(idx);
    return ToCompilerType(child.getType().getPointer());
  }

  case swift::TypeKind::BuiltinFixedArray: {
    auto fixed_array =
        swift::cast<swift::BuiltinFixedArrayType>(swift_can_type);
    return ToCompilerType(fixed_array->getElementType());
  }

  case swift::TypeKind::Class:
  case swift::TypeKind::BoundGenericClass: {
    auto class_decl = swift_can_type->getClassOrBoundGenericClass();
    if (class_decl->hasSuperclass()) {
      if (idx == 0) {
        swift::Type superclass_swift_type = swift_can_type->getSuperclass();
        CompilerType superclass_type =
            ToCompilerType(superclass_swift_type.getPointer());

        name = GetSuperclassName(superclass_type);

        // We cannot reliably get layout information without an
        // execution context.
        if (bit_offset_ptr)
          *bit_offset_ptr = LLDB_INVALID_IVAR_OFFSET;
        if (bitfield_bit_size_ptr)
          *bitfield_bit_size_ptr = 0;
        if (is_bitfield_ptr)
          *is_bitfield_ptr = false;

        return superclass_type;
      }

      // Adjust the index to refer into the stored properties.
      --idx;
    }

    LLVM_FALLTHROUGH;
  }

  case swift::TypeKind::Struct:
  case swift::TypeKind::BoundGenericStruct: {
    auto nominal = swift_can_type->getAnyNominal();
    auto stored_properties = GetStoredProperties(nominal);
    if (idx >= stored_properties.size())
      break;

    auto property = stored_properties[idx];
    name = property->getBaseName().userFacingName().str();

    // We cannot reliably get layout information without an execution
    // context.
    if (bit_offset_ptr)
      *bit_offset_ptr = LLDB_INVALID_IVAR_OFFSET;
    if (bitfield_bit_size_ptr)
      *bitfield_bit_size_ptr = 0;
    if (is_bitfield_ptr)
      *is_bitfield_ptr = false;

    swift::Type child_swift_type = swift_can_type->getTypeOfMember(
        property);
    return ToCompilerType(child_swift_type.getPointer());
  }

  case swift::TypeKind::Protocol:
  case swift::TypeKind::ProtocolComposition:
  case swift::TypeKind::Existential: {
    ProtocolInfo protocol_info;
    if (!GetProtocolTypeInfo(ToCompilerType(GetSwiftType(type)), protocol_info))
      break;

    if (idx >= protocol_info.m_num_storage_words)
      break;

    CompilerType compiler_type = ToCompilerType(GetSwiftType(type));
    CompilerType child_type;
    if (!GetASTContext())
      return {};
    ThreadSafeASTContext ast_ctx = GetASTContext();
    std::tie(child_type, name) = GetExistentialTypeChild(
        *this, **ast_ctx, compiler_type, protocol_info, idx);

    std::optional<uint64_t> child_size =
        llvm::expectedToOptional(child_type.GetByteSize(nullptr));
    if (!child_size)
      return {};
    if (bit_offset_ptr)
      *bit_offset_ptr = idx * *child_size * 8;
    if (bitfield_bit_size_ptr)
      *bitfield_bit_size_ptr = 0;
    if (is_bitfield_ptr)
      *is_bitfield_ptr = false;

    return child_type;
  }

  case swift::TypeKind::ExistentialMetatype:
  case swift::TypeKind::Metatype:
    break;

  case swift::TypeKind::ElementArchetype:
  case swift::TypeKind::OpaqueTypeArchetype:
  case swift::TypeKind::ExistentialArchetype:
  case swift::TypeKind::PrimaryArchetype:
  case swift::TypeKind::Function:
  case swift::TypeKind::GenericFunction:
  case swift::TypeKind::LValue:
  case swift::TypeKind::UnboundGeneric:
  case swift::TypeKind::DynamicSelf:
    break;

  case swift::TypeKind::Optional:
  case swift::TypeKind::TypeAlias:
  case swift::TypeKind::Dictionary:
  case swift::TypeKind::ArraySlice:
  case swift::TypeKind::InlineArray:
    assert(false && "Not a canonical type");
    break;
  }

  return CompilerType();
}

// If a pointer to a pointee type (the clang_type arg) says that it
// has no children, then we either need to trust it, or override it
// and return a different result. For example, an "int *" has one
// child that is an integer, but a function pointer doesn't have any
// children. Likewise if a Record type claims it has no children, then
// there really is nothing to show.
uint32_t SwiftASTContext::GetNumPointeeChildren(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, 0);

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));

  const swift::TypeKind type_kind = swift_can_type->getKind();
  switch (type_kind) {
  case swift::TypeKind::BoundGenericClass:
  case swift::TypeKind::BoundGenericEnum:
  case swift::TypeKind::BoundGenericStruct:
  case swift::TypeKind::BuiltinDefaultActorStorage:
  case swift::TypeKind::BuiltinExecutor:
  case swift::TypeKind::BuiltinFixedArray:
  case swift::TypeKind::BuiltinJob:
  case swift::TypeKind::BuiltinNonDefaultDistributedActorStorage:
  case swift::TypeKind::BuiltinPackIndex:
  case swift::TypeKind::BuiltinRawUnsafeContinuation:
  case swift::TypeKind::BuiltinTuple:
  case swift::TypeKind::BuiltinUnboundGeneric:
  case swift::TypeKind::Class:
  case swift::TypeKind::DependentMember:
  case swift::TypeKind::DynamicSelf:
  case swift::TypeKind::ElementArchetype:
  case swift::TypeKind::Enum:
  case swift::TypeKind::Error:
  case swift::TypeKind::ErrorUnion:
  case swift::TypeKind::Existential:
  case swift::TypeKind::ExistentialMetatype:
  case swift::TypeKind::Function:
  case swift::TypeKind::GenericFunction:
  case swift::TypeKind::GenericTypeParam:
  case swift::TypeKind::InOut:
  case swift::TypeKind::Integer:
  case swift::TypeKind::Locatable:
  case swift::TypeKind::Metatype:
  case swift::TypeKind::Module:
  case swift::TypeKind::OpaqueTypeArchetype:
  case swift::TypeKind::ExistentialArchetype:
  case swift::TypeKind::Pack:
  case swift::TypeKind::PackArchetype:
  case swift::TypeKind::PackElement:
  case swift::TypeKind::PackExpansion:
  case swift::TypeKind::ParameterizedProtocol:
  case swift::TypeKind::Placeholder:
  case swift::TypeKind::PrimaryArchetype:
  case swift::TypeKind::Protocol:
  case swift::TypeKind::ProtocolComposition:
  case swift::TypeKind::SILBlockStorage:
  case swift::TypeKind::SILBox:
  case swift::TypeKind::SILFunction:
  case swift::TypeKind::SILMoveOnlyWrapped:
  case swift::TypeKind::SILPack:
  case swift::TypeKind::SILToken:
  case swift::TypeKind::Struct:
  case swift::TypeKind::Tuple:
  case swift::TypeKind::TypeVariable:
  case swift::TypeKind::UnboundGeneric:
  case swift::TypeKind::VariadicSequence:
    return 0;
  case swift::TypeKind::BuiltinInteger:
  case swift::TypeKind::BuiltinIntegerLiteral:
  case swift::TypeKind::BuiltinFloat:
  case swift::TypeKind::BuiltinRawPointer:
  case swift::TypeKind::BuiltinUnsafeValueBuffer:
  case swift::TypeKind::BuiltinNativeObject:
  case swift::TypeKind::BuiltinBridgeObject:
    return 1;
  case swift::TypeKind::BuiltinVector:
    return 0;
  case swift::TypeKind::UnmanagedStorage:
  case swift::TypeKind::UnownedStorage:
  case swift::TypeKind::WeakStorage:
    return GetNumPointeeChildren(
        swift::cast<swift::ReferenceStorageType>(swift_can_type).getPointer());
  case swift::TypeKind::LValue:
    return 1;

  case swift::TypeKind::Optional:
  case swift::TypeKind::TypeAlias:
  case swift::TypeKind::Dictionary:
  case swift::TypeKind::ArraySlice:
  case swift::TypeKind::InlineArray:
    assert(false && "Not a canonical type");
    break;
  }

  return 0;
}

static std::optional<uint64_t> GetInstanceVariableOffset_Metadata(
    ValueObject *valobj, ExecutionContext *exe_ctx, const CompilerType &type,
    StringRef ivar_name, const CompilerType &ivar_type) {
  llvm::SmallString<1> m_description;
  LOG_PRINTF(GetLog(LLDBLog::Types), "ivar_name = %s, type = %s",
             ivar_name.str().c_str(), type.GetTypeName().AsCString());

  Process *process = exe_ctx->GetProcessPtr();
  if (!process) {
    LOG_PRINTF(GetLog(LLDBLog::Types), "no process");
    return {};
  }

  SwiftLanguageRuntime *runtime = SwiftLanguageRuntime::Get(process);
  if (!runtime) {
    LOG_PRINTF(GetLog(LLDBLog::Types), "no runtime");
    return {};
  }

  Status error;
  std::optional<uint64_t> offset =
      runtime->GetMemberVariableOffset(type, valobj, ivar_name, &error);
  if (offset)
    LOG_PRINTF(GetLog(LLDBLog::Types), "for %s: %" PRIu64, ivar_name.str().c_str(),
               *offset);
  else
    LOG_PRINTF(GetLog(LLDBLog::Types), "resolver failure: %s",
               error.AsCString());

  return offset;
}

static std::optional<uint64_t>
GetInstanceVariableOffset(ValueObject *valobj, ExecutionContext *exe_ctx,
                          const CompilerType &class_type, StringRef ivar_name,
                          const CompilerType &ivar_type) {
  if (ivar_name.empty())
    return {};

  if (!exe_ctx)
    return {};

  Target *target = exe_ctx->GetTargetPtr();
  if (!target)
    return {};

  return GetInstanceVariableOffset_Metadata(valobj, exe_ctx, class_type,
                                            ivar_name, ivar_type);
}

llvm::Expected<CompilerType> SwiftASTContext::GetChildCompilerTypeAtIndex(
    opaque_compiler_type_t type, ExecutionContext *exe_ctx, size_t idx,
    bool transparent_pointers, bool omit_empty_base_classes,
    bool ignore_array_bounds, std::string &child_name,
    uint32_t &child_byte_size, int32_t &child_byte_offset,
    uint32_t &child_bitfield_bit_size, uint32_t &child_bitfield_bit_offset,
    bool &child_is_base_class, bool &child_is_deref_of_parent,
    ValueObject *valobj, uint64_t &language_flags) {
  VALID_OR_RETURN_CHECK_TYPE(type, CompilerType());

  auto get_type_size = [&exe_ctx](CompilerType type) {
    auto *exe_scope =
        exe_ctx ? exe_ctx->GetBestExecutionContextScope() : nullptr;
    return type.GetByteSize(exe_scope);
  };

  language_flags = 0;
  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  ThreadSafeASTContext ast_ctx = GetASTContext();
  assert(&swift_can_type->getASTContext() == *ast_ctx);

  const swift::TypeKind type_kind = swift_can_type->getKind();
  switch (type_kind) {
  case swift::TypeKind::BuiltinBridgeObject:
  case swift::TypeKind::BuiltinDefaultActorStorage:
  case swift::TypeKind::BuiltinExecutor:
  case swift::TypeKind::BuiltinFloat:
  case swift::TypeKind::BuiltinInteger:
  case swift::TypeKind::BuiltinIntegerLiteral:
  case swift::TypeKind::BuiltinJob:
  case swift::TypeKind::BuiltinNativeObject:
  case swift::TypeKind::BuiltinNonDefaultDistributedActorStorage:
  case swift::TypeKind::BuiltinPackIndex:
  case swift::TypeKind::BuiltinRawPointer:
  case swift::TypeKind::BuiltinRawUnsafeContinuation:
  case swift::TypeKind::BuiltinTuple:
  case swift::TypeKind::BuiltinUnboundGeneric:
  case swift::TypeKind::BuiltinUnsafeValueBuffer:
  case swift::TypeKind::BuiltinVector:
  case swift::TypeKind::DependentMember:
  case swift::TypeKind::DynamicSelf:
  case swift::TypeKind::ElementArchetype:
  case swift::TypeKind::Error:
  case swift::TypeKind::ErrorUnion:
  case swift::TypeKind::ExistentialMetatype:
  case swift::TypeKind::Function:
  case swift::TypeKind::GenericFunction:
  case swift::TypeKind::GenericTypeParam:
  case swift::TypeKind::InOut:
  case swift::TypeKind::Integer:
  case swift::TypeKind::Locatable:
  case swift::TypeKind::Metatype:
  case swift::TypeKind::Module:
  case swift::TypeKind::OpaqueTypeArchetype:
  case swift::TypeKind::ExistentialArchetype:
  case swift::TypeKind::Pack:
  case swift::TypeKind::PackArchetype:
  case swift::TypeKind::PackElement:
  case swift::TypeKind::PackExpansion:
  case swift::TypeKind::ParameterizedProtocol:
  case swift::TypeKind::Placeholder:
  case swift::TypeKind::PrimaryArchetype:
  case swift::TypeKind::SILBlockStorage:
  case swift::TypeKind::SILBox:
  case swift::TypeKind::SILFunction:
  case swift::TypeKind::SILMoveOnlyWrapped:
  case swift::TypeKind::SILPack:
  case swift::TypeKind::SILToken:
  case swift::TypeKind::TypeVariable:
  case swift::TypeKind::UnboundGeneric:
  case swift::TypeKind::VariadicSequence:
    break;

  case swift::TypeKind::UnmanagedStorage:
  case swift::TypeKind::UnownedStorage:
  case swift::TypeKind::WeakStorage:
    return ToCompilerType(swift_can_type->getReferenceStorageReferent())
        .GetChildCompilerTypeAtIndex(
            exe_ctx, idx, transparent_pointers, omit_empty_base_classes,
            ignore_array_bounds, child_name, child_byte_size, child_byte_offset,
            child_bitfield_bit_size, child_bitfield_bit_offset,
            child_is_base_class, child_is_deref_of_parent, valobj,
            language_flags);

  case swift::TypeKind::Enum:
  case swift::TypeKind::BoundGenericEnum: {
    ExecutionContextScope *exe_scope = nullptr;
    if (exe_ctx)
      exe_scope = exe_ctx->GetBestExecutionContextScope();
    if (exe_scope)
      if (auto *runtime =
              SwiftLanguageRuntime::Get(exe_scope->CalculateProcess()))
        return runtime->GetChildCompilerTypeAtIndex(
            {weak_from_this(), type}, idx, transparent_pointers,
            omit_empty_base_classes, ignore_array_bounds, child_name,
            child_byte_size, child_byte_offset, child_bitfield_bit_size,
            child_bitfield_bit_offset, child_is_base_class,
            child_is_deref_of_parent, valobj, language_flags);
    return llvm::createStringError("cannot project enum type without process");

    SwiftEnumDescriptor *cached_enum_info = GetCachedEnumInfo(type);
    if (cached_enum_info &&
        idx < cached_enum_info->GetNumElementsWithPayload()) {
      const SwiftEnumDescriptor::ElementInfo *element_info =
          cached_enum_info->GetElementWithPayloadAtIndex(idx);
      child_name.assign(element_info->name.GetCString());
      auto size_or_err = get_type_size(element_info->payload_type);
      if (!size_or_err)
        return llvm::joinErrors(
            llvm::createStringError("could not get size for enum element " +
                                    llvm::Twine(idx)),
            size_or_err.takeError());
      child_byte_size = *size_or_err;
      child_byte_offset = 0;
      child_bitfield_bit_size = 0;
      child_bitfield_bit_offset = 0;
      child_is_base_class = false;
      child_is_deref_of_parent = false;
      if (element_info->is_indirect) {
        language_flags |= LanguageFlags::eIsIndirectEnumCase;
        return ToCompilerType(ast_ctx->TheRawPointerType.getPointer());
      } else
        return element_info->payload_type;
    }
  } break;

  case swift::TypeKind::Tuple: {
    auto tuple_type = swift::cast<swift::TupleType>(swift_can_type);
    if (idx >= tuple_type->getNumElements())
      break;

    const auto &child = tuple_type->getElement(idx);

    // Format the integer.
    llvm::SmallString<16> printed_idx;
    llvm::raw_svector_ostream(printed_idx) << idx;
    child_name = GetTupleElementName(tuple_type, idx, printed_idx);

    CompilerType child_type = ToCompilerType(child.getType().getPointer());
    auto size_or_err = get_type_size(child_type);
    if (!size_or_err)
      return llvm::joinErrors(
          llvm::createStringError("could not get size for tuple element " +
                                  child_name),
          size_or_err.takeError());
    child_byte_size = *size_or_err;
    child_is_base_class = false;
    child_is_deref_of_parent = false;

    CompilerType compiler_type = ToCompilerType(GetSwiftType(type));
    std::optional<uint64_t> offset = GetInstanceVariableOffset(
        valobj, exe_ctx, compiler_type, printed_idx.c_str(), child_type);
    if (!offset)
      return llvm::createStringError("could not get offset for tuple element " +
                                     child_name);

    child_byte_offset = *offset;
    child_bitfield_bit_size = 0;
    child_bitfield_bit_offset = 0;

    return child_type;
  }

  case swift::TypeKind::BuiltinFixedArray: {
    auto fixed_array =
        swift::cast<swift::BuiltinFixedArrayType>(swift_can_type);
    auto num_elts = fixed_array->getFixedInhabitedSize();
    if (!num_elts)
      break;
    if (idx >= *num_elts)
      break;

    CompilerType child_type = ToCompilerType(fixed_array->getElementType());
    llvm::raw_string_ostream(child_name) << idx;

    auto size_or_err = get_type_size(child_type);
    if (!size_or_err)
      return llvm::joinErrors(
          llvm::createStringError("could not get size of fixes array element " +
                                  child_name),
          size_or_err.takeError());
    child_byte_size = *size_or_err;
    child_is_base_class = false;
    child_is_deref_of_parent = false;

    CompilerType compiler_type = ToCompilerType(GetSwiftType(type));
    // FIXME: This is *not* generally correct, but there is no
    // reflection metadata yet.
    uint64_t offset = idx * child_byte_size;

    // FIXME: Are there sub-byte strides?
    child_byte_offset = offset;
    child_bitfield_bit_size = 0;
    child_bitfield_bit_offset = 0;

    return child_type;
  }

  case swift::TypeKind::Class:
  case swift::TypeKind::BoundGenericClass: {
    auto class_decl = swift_can_type->getClassOrBoundGenericClass();
    // Child 0 is the superclass, if there is one.
    if (class_decl->hasSuperclass()) {
      if (idx == 0) {
        swift::Type superclass_swift_type = swift_can_type->getSuperclass();
        CompilerType superclass_type =
            ToCompilerType(superclass_swift_type.getPointer());

        child_name = GetSuperclassName(superclass_type);
        auto size_or_err = get_type_size(superclass_type);
        if (!size_or_err)
          return llvm::joinErrors(
              llvm::createStringError("could not get size of super class"),
              size_or_err.takeError());
        child_byte_size = *size_or_err;
        child_is_base_class = true;
        child_is_deref_of_parent = false;

        child_byte_offset = 0;
        child_bitfield_bit_size = 0;
        child_bitfield_bit_offset = 0;

        return superclass_type;
      }

      // Adjust the index to refer into the stored properties.
      --idx;
    }
    LLVM_FALLTHROUGH;
  }

  case swift::TypeKind::Struct:
  case swift::TypeKind::BoundGenericStruct: {
    auto nominal = swift_can_type->getAnyNominal();

    // Imported unions don't have stored properties, iterate over the
    // VarDecls instead.
    if (auto *ntd =
            llvm::dyn_cast_or_null<swift::NominalTypeDecl>(nominal->getDecl()))
      if (auto *rd =
              llvm::dyn_cast_or_null<clang::RecordDecl>(ntd->getClangDecl()))
        if (rd->isUnion()) {
          unsigned count = 0;
          for (swift::Decl *D : ntd->getMembers()) {
            auto *VD = llvm::dyn_cast_or_null<swift::VarDecl>(D);
            if (!VD)
              continue;
            if (count++ < idx)
              continue;

            CompilerType child_type =
                ToCompilerType(VD->getTypeInContext().getPointer());
            child_name = VD->getNameStr().str();
            auto size_or_err = get_type_size(child_type);
            if (!size_or_err)
              return llvm::joinErrors(
                  llvm::createStringError("could not get size of field " +
                                          child_name),
                  size_or_err.takeError());
            child_byte_size = *size_or_err;
            child_is_base_class = false;
            child_is_deref_of_parent = false;
            child_byte_offset = 0;
            child_bitfield_bit_size = 0;
            child_bitfield_bit_offset = 0;
            return child_type;
          }
          return llvm::createStringError("could not get size of field " +
                                         child_name);
        }

    auto stored_properties = GetStoredProperties(nominal);
    if (idx >= stored_properties.size())
      break;

    // Find the stored property with this index.
    auto property = stored_properties[idx];
    swift::Type child_swift_type = swift_can_type->getTypeOfMember(
        property);

    CompilerType child_type = ToCompilerType(child_swift_type.getPointer());
    child_name = property->getBaseName().userFacingName().str();
    auto size_or_err = get_type_size(child_type);
    if (!size_or_err)
      return llvm::joinErrors(
          llvm::createStringError("could not get size of field " +
                                  child_name),
          size_or_err.takeError());
    child_byte_size = *size_or_err;
    child_is_base_class = false;
    child_is_deref_of_parent = false;

    CompilerType compiler_type = ToCompilerType(GetSwiftType(type));
    std::optional<uint64_t> offset = GetInstanceVariableOffset(
        valobj, exe_ctx, compiler_type, child_name.c_str(), child_type);
    if (!offset)
      return llvm::createStringError("could not get offset of field " +
                                     child_name);
    child_byte_offset = *offset;
    child_bitfield_bit_size = 0;
    child_bitfield_bit_offset = 0;
    return child_type;
  }

  case swift::TypeKind::Protocol:
  case swift::TypeKind::ProtocolComposition:
  case swift::TypeKind::Existential: {
    ProtocolInfo protocol_info;
    if (!GetProtocolTypeInfo(ToCompilerType(GetSwiftType(type)), protocol_info))
      break;

    if (idx >= protocol_info.m_num_storage_words)
      break;

    CompilerType compiler_type = ToCompilerType(GetSwiftType(type));
    CompilerType child_type;
    if (!GetASTContext())
      return llvm::createStringError("no ast context");

    std::tie(child_type, child_name) = GetExistentialTypeChild(
        *this, **ast_ctx, compiler_type, protocol_info, idx);
    auto size_or_err = get_type_size(child_type);
    if (!size_or_err)
      return llvm::joinErrors(
          llvm::createStringError("could not get size of field " +
                                  llvm::Twine(idx)),
          size_or_err.takeError());
    child_byte_size = *size_or_err;
    child_byte_offset = idx * child_byte_size;
    child_bitfield_bit_size = 0;
    child_bitfield_bit_offset = 0;
    child_is_base_class = false;
    child_is_deref_of_parent = false;

    return child_type;
  }

  case swift::TypeKind::LValue:
    if (idx < llvm::expectedToStdOptional(
                  GetNumChildren(type, omit_empty_base_classes, exe_ctx))
                  .value_or(0)) {
      CompilerType pointee_clang_type(GetNonReferenceType(type));
      Flags pointee_clang_type_flags(pointee_clang_type.GetTypeInfo());
      const char *parent_name = valobj ? valobj->GetName().GetCString() : NULL;
      if (parent_name) {
        child_name.assign(1, '&');
        child_name += parent_name;
      }

      // We have a pointer to a simple type
      if (idx == 0) {
        auto size_or_err = get_type_size(pointee_clang_type);
        if (!size_or_err)
          return llvm::joinErrors(
              llvm::createStringError("could not get size of lvalue"),
              size_or_err.takeError());
        child_byte_size = *size_or_err;
        child_byte_offset = 0;
        return pointee_clang_type;
      }
    }
    break;

  case swift::TypeKind::Optional:
  case swift::TypeKind::TypeAlias:
  case swift::TypeKind::Dictionary:
  case swift::TypeKind::ArraySlice:
  case swift::TypeKind::InlineArray:
    assert(false && "Not a canonical type");
    break;
  }
  return CompilerType();
}

// Look for a child member (doesn't include base classes, but it does
// include their members) in the type hierarchy. Returns an index path
// into "clang_type" on how to reach the appropriate member.
//
//    class A
//    {
//    public:
//        int m_a;
//        int m_b;
//    };
//
//    class B
//    {
//    };
//
//    class C :
//        public B,
//        public A
//    {
//    };
//
// If we have a clang type that describes "class C", and we wanted to
// look for "m_b" in it:
//
// With omit_empty_base_classes == false we would get an integer array back
// with:
// { 1,  1 }
// The first index 1 is the child index for "class A" within class C.
// The second index 1 is the child index for "m_b" within class A.
//
// With omit_empty_base_classes == true we would get an integer array back with:
// { 0,  1 }
// The first index 0 is the child index for "class A" within class C
// (since class B doesn't have any members it doesn't count).  The
// second index 1 is the child index for "m_b" within class A.

size_t SwiftASTContext::GetIndexOfChildMemberWithName(
    opaque_compiler_type_t type, StringRef name, ExecutionContext *exe_ctx,
    bool omit_empty_base_classes, std::vector<uint32_t> &child_indexes) {
  VALID_OR_RETURN_CHECK_TYPE(type, 0);

  if (!name.empty()) {
    swift::CanType swift_can_type(GetCanonicalSwiftType(type));

    const swift::TypeKind type_kind = swift_can_type->getKind();
    switch (type_kind) {
    case swift::TypeKind::BuiltinBridgeObject:
    case swift::TypeKind::BuiltinDefaultActorStorage:
    case swift::TypeKind::BuiltinExecutor:
    case swift::TypeKind::BuiltinFloat:
    case swift::TypeKind::BuiltinInteger:
    case swift::TypeKind::BuiltinIntegerLiteral:
    case swift::TypeKind::BuiltinJob:
    case swift::TypeKind::BuiltinNativeObject:
    case swift::TypeKind::BuiltinNonDefaultDistributedActorStorage:
    case swift::TypeKind::BuiltinPackIndex:
    case swift::TypeKind::BuiltinRawPointer:
    case swift::TypeKind::BuiltinRawUnsafeContinuation:
    case swift::TypeKind::BuiltinTuple:
    case swift::TypeKind::BuiltinUnboundGeneric:
    case swift::TypeKind::BuiltinUnsafeValueBuffer:
    case swift::TypeKind::BuiltinVector:
    case swift::TypeKind::DynamicSelf:
    case swift::TypeKind::ElementArchetype:
    case swift::TypeKind::Error:
    case swift::TypeKind::ErrorUnion:
    case swift::TypeKind::ExistentialMetatype:
    case swift::TypeKind::Function:
    case swift::TypeKind::GenericFunction:
    case swift::TypeKind::InOut:
    case swift::TypeKind::Integer:
    case swift::TypeKind::Locatable:
    case swift::TypeKind::Metatype:
    case swift::TypeKind::Module:
    case swift::TypeKind::OpaqueTypeArchetype:
    case swift::TypeKind::ExistentialArchetype:
    case swift::TypeKind::Pack:
    case swift::TypeKind::PackArchetype:
    case swift::TypeKind::PackElement:
    case swift::TypeKind::PackExpansion:
    case swift::TypeKind::ParameterizedProtocol:
    case swift::TypeKind::Placeholder:
    case swift::TypeKind::PrimaryArchetype:
    case swift::TypeKind::SILBlockStorage:
    case swift::TypeKind::SILBox:
    case swift::TypeKind::SILFunction:
    case swift::TypeKind::SILMoveOnlyWrapped:
    case swift::TypeKind::SILPack:
    case swift::TypeKind::SILToken:
    case swift::TypeKind::TypeVariable:
    case swift::TypeKind::UnboundGeneric:
    case swift::TypeKind::VariadicSequence:
      break;

    case swift::TypeKind::UnmanagedStorage:
    case swift::TypeKind::UnownedStorage:
    case swift::TypeKind::WeakStorage:
      return ToCompilerType(swift_can_type->getReferenceStorageReferent())
          .GetIndexOfChildMemberWithName(name, exe_ctx, omit_empty_base_classes,
                                         child_indexes);
    case swift::TypeKind::GenericTypeParam:
    case swift::TypeKind::DependentMember:
      break;

    case swift::TypeKind::Enum:
    case swift::TypeKind::BoundGenericEnum: {
      return 0;
      SwiftEnumDescriptor *cached_enum_info = GetCachedEnumInfo(type);
      if (cached_enum_info) {
        ConstString const_name(name);
        const size_t num_sized_elements =
            cached_enum_info->GetNumElementsWithPayload();
        for (size_t i = 0; i < num_sized_elements; ++i) {
          if (cached_enum_info->GetElementWithPayloadAtIndex(i)->name ==
              const_name) {
            child_indexes.push_back(i);
            return child_indexes.size();
          }
        }
      }
    } break;

    case swift::TypeKind::Tuple: {
      // For tuples only always look for the member by number first as
      // a tuple element can be named, yet still be accessed by the
      // number.
      swift::TupleType *tuple_type = swift_can_type->castTo<swift::TupleType>();
      uint32_t tuple_idx = 0;
      if (llvm::to_integer(name, tuple_idx)) {
        if (tuple_idx < tuple_type->getNumElements()) {
          child_indexes.push_back(tuple_idx);
          return child_indexes.size();
        }
        return 0;
      }

      // Otherwise, perform lookup by name.
      for (uint32_t tuple_idx : swift::range(tuple_type->getNumElements())) {
        if (tuple_type->getElement(tuple_idx).getName().str() == name) {
          child_indexes.push_back(tuple_idx);
          return child_indexes.size();
        }
      }

      return 0;
    }
    case swift::TypeKind::BuiltinFixedArray: {
      auto fixed_array =
          swift::cast<swift::BuiltinFixedArrayType>(swift_can_type);
      auto num_elts = fixed_array->getFixedInhabitedSize();
      if (!num_elts)
        return 0;

      uint32_t array_idx = 0;
      if (llvm::to_integer(name, array_idx))
        if (array_idx < *num_elts) {
          child_indexes.push_back(array_idx);
          return child_indexes.size();
        }

      return 0;
    }

    case swift::TypeKind::Struct:
    case swift::TypeKind::Class:
    case swift::TypeKind::BoundGenericClass:
    case swift::TypeKind::BoundGenericStruct: {
      auto nominal = swift_can_type->getAnyNominal();
      auto stored_properties = GetStoredProperties(nominal);
      auto class_decl = llvm::dyn_cast<swift::ClassDecl>(nominal);

      // Search the stored properties.
      for (unsigned idx : indices(stored_properties)) {
        auto property = stored_properties[idx];
        if (property->getBaseName().userFacingName() == name) {
          // We found it!

          // If we have a superclass, adjust the index accordingly.
          if (class_decl && class_decl->hasSuperclass())
            ++idx;

          child_indexes.push_back(idx);
          return child_indexes.size();
        }
      }

      // Search the superclass, if there is one.
      if (class_decl && class_decl->hasSuperclass()) {
        // Push index zero for the base class
        child_indexes.push_back(0);

        // Look in the superclass.
        swift::Type superclass_swift_type = swift_can_type->getSuperclass();
        CompilerType superclass_type =
            ToCompilerType(superclass_swift_type.getPointer());
        if (superclass_type.GetIndexOfChildMemberWithName(
                name, exe_ctx, omit_empty_base_classes, child_indexes))
          return child_indexes.size();

        // We didn't find a stored property matching "name" in our
        // superclass, pop the superclass zero index that we pushed on
        // above.
        child_indexes.pop_back();
      }
    } break;

    case swift::TypeKind::Protocol:
    case swift::TypeKind::ProtocolComposition:
    case swift::TypeKind::Existential: {
      ProtocolInfo protocol_info;
      if (!GetProtocolTypeInfo(ToCompilerType(GetSwiftType(type)),
                               protocol_info))
        break;

      CompilerType compiler_type = ToCompilerType(GetSwiftType(type));
      for (unsigned idx : swift::range(protocol_info.m_num_storage_words)) {
        CompilerType child_type;
        std::string child_name;
        ThreadSafeASTContext ast_ctx = GetASTContext();
        if (!ast_ctx)
          return 0;
        std::tie(child_type, child_name) = GetExistentialTypeChild(
            *this, **ast_ctx, compiler_type, protocol_info, idx);
        if (name == child_name) {
          child_indexes.push_back(idx);
          return child_indexes.size();
        }
      }
    } break;

    case swift::TypeKind::LValue: {
      CompilerType pointee_clang_type(GetNonReferenceType(type));

      if (pointee_clang_type.IsAggregateType()) {
        return pointee_clang_type.GetIndexOfChildMemberWithName(
            name, exe_ctx, omit_empty_base_classes, child_indexes);
      }
    } break;

    case swift::TypeKind::Optional:
    case swift::TypeKind::TypeAlias:
    case swift::TypeKind::Dictionary:
    case swift::TypeKind::ArraySlice:
    case swift::TypeKind::InlineArray:
      assert(false && "Not a canonical type");
      break;
    }
  }
  return 0;
}

size_t SwiftASTContext::GetNumTemplateArguments(opaque_compiler_type_t type,
                                                bool expand_pack) {
  VALID_OR_RETURN_CHECK_TYPE(type, 0);

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));

  const swift::TypeKind type_kind = swift_can_type->getKind();
  switch (type_kind) {
  case swift::TypeKind::UnboundGeneric: {
    swift::UnboundGenericType *unbound_generic_type =
        swift_can_type->castTo<swift::UnboundGenericType>();
    auto *nominal_type_decl = unbound_generic_type->getDecl();
    swift::GenericParamList *generic_param_list =
        nominal_type_decl->getGenericParams();
    return generic_param_list->getParams().size();
  } break;
  case swift::TypeKind::BoundGenericClass:
  case swift::TypeKind::BoundGenericStruct:
  case swift::TypeKind::BoundGenericEnum: {
    swift::BoundGenericType *bound_generic_type =
        swift_can_type->castTo<swift::BoundGenericType>();
    return bound_generic_type->getGenericArgs().size();
  }
  default:
    break;
  }

  return 0;
}

bool SwiftASTContext::GetSelectedEnumCase(const CompilerType &type,
                                          const DataExtractor &data,
                                          ConstString *name, bool *has_payload,
                                          CompilerType *payload,
                                          bool *is_indirect) {
  swift::CanType swift_can_type = GetCanonicalSwiftType(type);
  if (!swift_can_type)
    return false;
  auto *ast = GetSwiftASTContext(&swift_can_type->getASTContext());
  if (!ast)
    return false;
  SwiftEnumDescriptor *cached_enum_info =
      ast->GetCachedEnumInfo(swift_can_type.getPointer());
  if (!cached_enum_info)
    return false;
  auto enum_elem_info = cached_enum_info->GetElementFromData(data, true);
  if (!enum_elem_info)
    return false;
  if (name)
    *name = enum_elem_info->name;
  if (has_payload)
    *has_payload = enum_elem_info->has_payload;
  if (payload)
    *payload = enum_elem_info->payload_type;
  if (is_indirect)
    *is_indirect = enum_elem_info->is_indirect;
  return true;
}

lldb::GenericKind
SwiftASTContext::GetGenericArgumentKind(opaque_compiler_type_t type,
                                        size_t idx) {
  VALID_OR_RETURN_CHECK_TYPE(type, eNullGenericKindType);
  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  if (swift_can_type->getAs<swift::UnboundGenericType>())
    return eUnboundGenericKindType;
  if (auto *bound_generic_type =
          swift_can_type->getAs<swift::BoundGenericType>())
    if (idx < bound_generic_type->getGenericArgs().size())
      return eBoundGenericKindType;

  return eNullGenericKindType;
}

CompilerType SwiftASTContext::GetBoundGenericType(opaque_compiler_type_t type,
                                                  size_t idx) {
  VALID_OR_RETURN_CHECK_TYPE(type, CompilerType());

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  assert(&swift_can_type->getASTContext() == *GetASTContext());
  if (auto *bound_generic_type =
          swift_can_type->getAs<swift::BoundGenericType>())
    if (idx < bound_generic_type->getGenericArgs().size())
      return ToCompilerType(
          {bound_generic_type->getGenericArgs()[idx].getPointer()});

  return {};
}

CompilerType SwiftASTContext::GetUnboundGenericType(opaque_compiler_type_t type,
                                                    size_t idx) {
  VALID_OR_RETURN_CHECK_TYPE(type, CompilerType());

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  assert(&swift_can_type->getASTContext() == *GetASTContext());
  if (auto *unbound_generic_type =
          swift_can_type->getAs<swift::UnboundGenericType>()) {
    auto *nominal_type_decl = unbound_generic_type->getDecl();
    swift::GenericSignature generic_sig =
        nominal_type_decl->getGenericSignature();
    llvm::ArrayRef<swift::GenericTypeParamType *> params =
        generic_sig.getGenericParams();
    swift::Type paramTy = params[idx];
    return ToCompilerType({nominal_type_decl->mapTypeIntoContext(paramTy)
                               ->castTo<swift::ArchetypeType>()});
  }

  return {};
}

CompilerType SwiftASTContext::GetGenericArgumentType(CompilerType ct,
                                                     size_t idx) {
  swift::Type swift_type = GetSwiftTypeIgnoringErrors(ct);
  if (!swift_type)
    return {};
  auto *ast = GetSwiftASTContext(&swift_type->getASTContext());
  if (!ast)
    return {};
  return ast->GetGenericArgumentType(swift_type.getPointer(), idx);
}

CompilerType
SwiftASTContext::GetGenericArgumentType(opaque_compiler_type_t type,
                                        size_t idx) {
  VALID_OR_RETURN_CHECK_TYPE(type, CompilerType());

  switch (GetGenericArgumentKind(type, idx)) {
  case eBoundGenericKindType:
    return GetBoundGenericType(type, idx);
  case eUnboundGenericKindType:
    return GetUnboundGenericType(type, idx);
  default:
    break;
  }
  return {};
}

CompilerType
SwiftASTContext::GetTypeForFormatters(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, CompilerType());

  return {weak_from_this(), type};
}

LazyBool SwiftASTContext::ShouldPrintAsOneLiner(opaque_compiler_type_t type,
                                                ValueObject *valobj) {
  if (type) {
    CompilerType can_compiler_type(GetCanonicalType(type));
    auto ts =
        can_compiler_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
    if (ts &&
        ts->IsImportedType(can_compiler_type.GetOpaqueQualType(), nullptr))
      return eLazyBoolNo;
  }
  if (valobj) {
    if (valobj->IsBaseClass())
      return eLazyBoolNo;
    if ((valobj->GetLanguageFlags() & LanguageFlags::eIsIndirectEnumCase) ==
        LanguageFlags::eIsIndirectEnumCase)
      return eLazyBoolNo;
  }

  return eLazyBoolCalculate;
}

bool SwiftASTContext::IsMeaninglessWithoutDynamicResolution(
    opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, false);

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  return swift_can_type->hasTypeParameter();
}

//----------------------------------------------------------------------
// Dumping types
//----------------------------------------------------------------------
#ifndef NDEBUG
LLVM_DUMP_METHOD void SwiftASTContext::dump(opaque_compiler_type_t type) const {
  if (!type)
    return;
  swift::Type swift_type =
      const_cast<SwiftASTContext *>(this)->GetSwiftType(type);
  swift_type.dump();
}
#endif

bool SwiftASTContext::DumpTypeValue(
    opaque_compiler_type_t type, Stream &s, lldb::Format format,
    const lldb_private::DataExtractor &data, lldb::offset_t byte_offset,
    size_t byte_size, uint32_t bitfield_bit_size, uint32_t bitfield_bit_offset,
    ExecutionContextScope *exe_scope, bool is_base_class) {
  VALID_OR_RETURN_CHECK_TYPE(type, false);

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));

  const swift::TypeKind type_kind = swift_can_type->getKind();
  switch (type_kind) {
  case swift::TypeKind::BoundGenericStruct:
  case swift::TypeKind::BuiltinDefaultActorStorage:
  case swift::TypeKind::BuiltinExecutor:
  case swift::TypeKind::BuiltinFixedArray:
  case swift::TypeKind::BuiltinJob:
  case swift::TypeKind::BuiltinNonDefaultDistributedActorStorage:
  case swift::TypeKind::BuiltinPackIndex:
  case swift::TypeKind::BuiltinRawUnsafeContinuation:
  case swift::TypeKind::BuiltinTuple:
  case swift::TypeKind::BuiltinUnboundGeneric:
  case swift::TypeKind::BuiltinVector:
  case swift::TypeKind::DynamicSelf:
  case swift::TypeKind::Error:
  case swift::TypeKind::ErrorUnion:
  case swift::TypeKind::Existential:
  case swift::TypeKind::InOut:
  case swift::TypeKind::Integer:
  case swift::TypeKind::Locatable:
  case swift::TypeKind::Module:
  case swift::TypeKind::Pack:
  case swift::TypeKind::PackArchetype:
  case swift::TypeKind::PackElement:
  case swift::TypeKind::PackExpansion:
  case swift::TypeKind::ParameterizedProtocol:
  case swift::TypeKind::Placeholder:
  case swift::TypeKind::ProtocolComposition:
  case swift::TypeKind::SILBlockStorage:
  case swift::TypeKind::SILBox:
  case swift::TypeKind::SILFunction:
  case swift::TypeKind::SILMoveOnlyWrapped:
  case swift::TypeKind::SILPack:
  case swift::TypeKind::SILToken:
  case swift::TypeKind::Tuple:
  case swift::TypeKind::TypeVariable:
  case swift::TypeKind::UnboundGeneric:
  case swift::TypeKind::VariadicSequence:
    break;

  case swift::TypeKind::Class:
  case swift::TypeKind::BoundGenericClass:
    // If we have a class that is in a variable then it is a pointer,
    // else if it is a base class, it has no value.
    if (is_base_class)
      break;
    LLVM_FALLTHROUGH;
  case swift::TypeKind::BuiltinInteger:
  case swift::TypeKind::BuiltinIntegerLiteral:
  case swift::TypeKind::BuiltinFloat:
  case swift::TypeKind::BuiltinRawPointer:
  case swift::TypeKind::BuiltinNativeObject:
  case swift::TypeKind::BuiltinUnsafeValueBuffer:
  case swift::TypeKind::BuiltinBridgeObject:
  case swift::TypeKind::ElementArchetype:
  case swift::TypeKind::OpaqueTypeArchetype:
  case swift::TypeKind::ExistentialArchetype:
  case swift::TypeKind::PrimaryArchetype:
  case swift::TypeKind::Function:
  case swift::TypeKind::GenericFunction:
  case swift::TypeKind::GenericTypeParam:
  case swift::TypeKind::DependentMember:
  case swift::TypeKind::LValue: {
    uint32_t item_count = 1;
    // A few formats, we might need to modify our size and count for
    // depending on how we are trying to display the value.
    switch (format) {
    default:
    case eFormatBoolean:
    case eFormatBinary:
    case eFormatComplex:
    case eFormatCString: // NULL terminated C strings
    case eFormatDecimal:
    case eFormatEnum:
    case eFormatHex:
    case eFormatHexUppercase:
    case eFormatFloat:
    case eFormatOctal:
    case eFormatOSType:
    case eFormatUnsigned:
    case eFormatPointer:
    case eFormatVectorOfChar:
    case eFormatVectorOfSInt8:
    case eFormatVectorOfUInt8:
    case eFormatVectorOfSInt16:
    case eFormatVectorOfUInt16:
    case eFormatVectorOfSInt32:
    case eFormatVectorOfUInt32:
    case eFormatVectorOfSInt64:
    case eFormatVectorOfUInt64:
    case eFormatVectorOfFloat32:
    case eFormatVectorOfFloat64:
    case eFormatVectorOfUInt128:
      break;

    case eFormatAddressInfo:
      if (byte_size == 0) {
        byte_size = exe_scope->CalculateTarget()
                        ->GetArchitecture()
                        .GetAddressByteSize();
        item_count = 1;
      }
      break;

    case eFormatChar:
    case eFormatCharPrintable:
    case eFormatCharArray:
    case eFormatBytes:
    case eFormatBytesWithASCII:
      item_count = byte_size;
      byte_size = 1;
      break;

    case eFormatUnicode16:
      item_count = byte_size / 2;
      byte_size = 2;
      break;

    case eFormatUnicode32:
      item_count = byte_size / 4;
      byte_size = 4;
      break;
    }
    return DumpDataExtractor(data, &s, byte_offset, format, byte_size,
                             item_count, UINT32_MAX, LLDB_INVALID_ADDRESS,
                             bitfield_bit_size, bitfield_bit_offset, exe_scope);
  } break;

  case swift::TypeKind::UnmanagedStorage:
  case swift::TypeKind::UnownedStorage:
  case swift::TypeKind::WeakStorage:
    return ToCompilerType(swift_can_type->getReferenceStorageReferent())
        .DumpTypeValue(&s, format, data, byte_offset, byte_size,
                       bitfield_bit_size, bitfield_bit_offset, exe_scope,
                       is_base_class);
  case swift::TypeKind::Enum:
  case swift::TypeKind::BoundGenericEnum: {
    SwiftEnumDescriptor *cached_enum_info = GetCachedEnumInfo(type);
    if (cached_enum_info) {
      auto enum_elem_info = cached_enum_info->GetElementFromData(data, true);
      if (!enum_elem_info)
        return false;
      s.Printf("%s", enum_elem_info->name.GetCString());
      return true;
    } else
      s.Printf("<unknown type>");
  } break;

  case swift::TypeKind::Struct:
  case swift::TypeKind::Protocol:
    return false;

    
  case swift::TypeKind::Metatype:
  case swift::TypeKind::ExistentialMetatype: {
    return DumpDataExtractor(data, &s, byte_offset, eFormatPointer, byte_size, 1,
                             UINT32_MAX, LLDB_INVALID_ADDRESS,
                             bitfield_bit_size, bitfield_bit_offset, exe_scope);
  } break;

  case swift::TypeKind::Optional:
  case swift::TypeKind::TypeAlias:
  case swift::TypeKind::Dictionary:
  case swift::TypeKind::ArraySlice:
  case swift::TypeKind::InlineArray:
    assert(false && "Not a canonical type");
    break;
  }

  return false;
}

bool SwiftASTContext::IsImportedType(opaque_compiler_type_t type,
                                     CompilerType *original_type) {
  VALID_OR_RETURN_CHECK_TYPE(type, false);
  bool success = false;

  if (swift::Type swift_can_type = GetSwiftType(type)) {
    do {
      swift::NominalType *nominal_type =
          swift_can_type->getAs<swift::NominalType>();
      if (!nominal_type)
        break;
      swift::NominalTypeDecl *nominal_type_decl = nominal_type->getDecl();
      if (nominal_type_decl && nominal_type_decl->hasClangNode()) {
        const clang::Decl *clang_decl = nominal_type_decl->getClangDecl();
        if (!clang_decl)
          break;
        success = true;
        if (!original_type)
          break;

        // ObjCInterfaceDecl is not a TypeDecl.
        if (const clang::ObjCInterfaceDecl *objc_interface_decl =
                llvm::dyn_cast<clang::ObjCInterfaceDecl>(clang_decl)) {
          *original_type =
              CompilerType(TypeSystemClang::GetASTContext(
                               &objc_interface_decl->getASTContext())
                               ->weak_from_this(),
                           clang::QualType::getFromOpaquePtr(
                               objc_interface_decl->getTypeForDecl())
                               .getAsOpaquePtr());
        } else if (const clang::TypeDecl *type_decl =
                       llvm::dyn_cast<clang::TypeDecl>(clang_decl)) {
          *original_type = CompilerType(
              TypeSystemClang::GetASTContext(&type_decl->getASTContext())
                  ->weak_from_this(),
              clang::QualType::getFromOpaquePtr(type_decl->getTypeForDecl())
                  .getAsOpaquePtr());
        } else {
          // TODO: any more cases that we care about?
          *original_type = CompilerType();
        }
      }
    } while (0);
  }

  return success;
}

std::string SwiftASTContext::GetSwiftName(const clang::Decl *clang_decl,
                                          TypeSystemClang &clang_typesystem) {
  if (auto name_decl = llvm::dyn_cast<clang::NamedDecl>(clang_decl))
    return ImportName(name_decl);
  return {};
}

CompilerType
SwiftASTContext::ConvertClangTypeToSwiftType(CompilerType clang_type) {
  auto ts = GetTypeSystemSwiftTypeRef();
  if (!ts)
    return {};
  auto typeref_type = ts->ConvertClangTypeToSwiftType(clang_type);

  if (!typeref_type)
    return {};

  auto ast_type = ReconstructType(typeref_type.GetMangledTypeName());
  if (!ast_type) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Types), ast_type.takeError(),
                   "[SwiftASTContext::ConvertClangTypeToSwiftType] Could not "
                   "reconstruct type. Error: {0}");
    return {};
  }

  return {this->weak_from_this(), *ast_type};
}

CompilerType SwiftASTContext::GetBuiltinIntType() {
  // FIXME: use target int size!
  return GetTypeFromMangledTypename(ConstString("$sBi64_D"));
}

bool SwiftASTContext::TypeHasArchetype(CompilerType type) {
  auto swift_type = GetSwiftTypeIgnoringErrors(type);
  if (swift_type)
    return swift_type->hasArchetype();
  return false;
}

std::string SwiftASTContext::ImportName(const clang::NamedDecl *clang_decl) {
  if (auto clang_importer = GetClangImporter()) {
    swift::DeclName imported_name = clang_importer->importName(clang_decl, {});
    return imported_name.getBaseName().userFacingName().str();
  }
  return clang_decl->getName().str();
}

void SwiftASTContext::DumpTypeDescription(opaque_compiler_type_t type,
                                          lldb::DescriptionLevel level,
                                          ExecutionContextScope *) {
  StreamFile s(stdout, false);
  DumpTypeDescription(type, s, level);
}

void SwiftASTContext::DumpTypeDescription(opaque_compiler_type_t type,
                                          Stream &s,
                                          lldb::DescriptionLevel level,
                                          ExecutionContextScope *) {
  DumpTypeDescription(type, &s, false, true, level);
}

void SwiftASTContext::DumpTypeDescription(opaque_compiler_type_t type,
                                          bool print_help_if_available,
                                          bool print_extensions_if_available,
                                          lldb::DescriptionLevel level,
                                          ExecutionContextScope *) {
  StreamFile s(stdout, false);
  DumpTypeDescription(type, &s, print_help_if_available,
                      print_extensions_if_available, level);
}

static void PrintSwiftNominalType(swift::NominalTypeDecl *nominal_type_decl,
                                  Stream *s, bool print_help_if_available,
                                  bool print_extensions_if_available) {
  if (nominal_type_decl && s) {
    std::string buffer;
    llvm::raw_string_ostream ostream(buffer);
    const swift::PrintOptions &print_options(
        SwiftASTContext::GetUserVisibleTypePrintingOptions(
            print_help_if_available));
    nominal_type_decl->print(ostream, print_options);
    ostream.flush();
    if (buffer.empty() == false)
      s->Printf("%s\n", buffer.c_str());
    if (print_extensions_if_available) {
      for (auto ext : nominal_type_decl->getExtensions()) {
        if (ext) {
          buffer.clear();
          llvm::raw_string_ostream ext_ostream(buffer);
          ext->print(ext_ostream, print_options);
          ext_ostream.flush();
          if (buffer.empty() == false)
            s->Printf("%s\n", buffer.c_str());
        }
      }
    }
  }
}

void SwiftASTContext::DumpTypeDescription(opaque_compiler_type_t type,
                                          Stream *s,
                                          bool print_help_if_available,
                                          bool print_extensions_if_available,
                                          lldb::DescriptionLevel level,
                                          ExecutionContextScope *) {
  const auto initial_written_bytes = s->GetWrittenBytes();

  if (type) {
    swift::CanType swift_can_type(GetCanonicalSwiftType(type));
    switch (swift_can_type->getKind()) {
    case swift::TypeKind::Module: {
      swift::ModuleType *module_type =
          swift_can_type->castTo<swift::ModuleType>();
      swift::ModuleDecl *module = module_type->getModule();
      llvm::SmallVector<swift::Decl *, 10> decls;
      module->getDisplayDecls(decls);
      for (swift::Decl *decl : decls) {
        swift::DeclKind kind = decl->getKind();
        if (kind >= swift::DeclKind::First_TypeDecl &&
            kind <= swift::DeclKind::Last_TypeDecl) {
          swift::TypeDecl *type_decl =
              llvm::dyn_cast_or_null<swift::TypeDecl>(decl);
          if (type_decl) {
            CompilerType clang_type(ToCompilerType(
                type_decl->getDeclaredInterfaceType().getPointer()));
            if (clang_type) {
              Flags clang_type_flags(clang_type.GetTypeInfo());
              DumpTypeDescription(clang_type.GetOpaqueQualType(), s,
                                  print_help_if_available,
                                  print_extensions_if_available, level);
            }
          }
        } else if (kind == swift::DeclKind::Func ||
                   kind == swift::DeclKind::Var) {
          std::string buffer;
          llvm::raw_string_ostream stream(buffer);
          decl->print(stream,
                      SwiftASTContext::GetUserVisibleTypePrintingOptions(
                          print_help_if_available));
          stream.flush();
          s->Printf("%s\n", buffer.c_str());
        } else if (kind == swift::DeclKind::Import) {
          swift::ImportDecl *import_decl =
              llvm::dyn_cast_or_null<swift::ImportDecl>(decl);
          if (import_decl) {
            switch (import_decl->getImportKind()) {
            case swift::ImportKind::Module: {
              swift::ModuleDecl *imported_module = import_decl->getModule();
              if (imported_module) {
                s->Printf("import %s\n", imported_module->getName().get());
              }
            } break;
            default: {
              for (swift::Decl *imported_decl : import_decl->getDecls()) {
                // All of the non-module things you can import should
                // be a ValueDecl.
                if (swift::ValueDecl *imported_value_decl =
                        llvm::dyn_cast_or_null<swift::ValueDecl>(
                            imported_decl)) {
                  if (swift::TypeBase *decl_type =
                          imported_value_decl->getInterfaceType()
                              .getPointer()) {
                    DumpTypeDescription(decl_type, s, print_help_if_available,
                                        print_extensions_if_available, level);
                  }
                }
              }
            } break;
            }
          }
        }
      }
      break;
    }
    case swift::TypeKind::Metatype: {
      s->PutCString("metatype ");
      swift::MetatypeType *metatype_type =
          swift_can_type->castTo<swift::MetatypeType>();
      DumpTypeDescription(metatype_type->getInstanceType().getPointer(),
                          print_help_if_available,
                          print_extensions_if_available, level);
    } break;
    case swift::TypeKind::UnboundGeneric: {
      swift::UnboundGenericType *unbound_generic_type =
          swift_can_type->castTo<swift::UnboundGenericType>();
      auto nominal_type_decl = llvm::dyn_cast<swift::NominalTypeDecl>(
          unbound_generic_type->getDecl());
      if (nominal_type_decl) {
        PrintSwiftNominalType(nominal_type_decl, s, print_help_if_available,
                              print_extensions_if_available);
      }
    } break;
    case swift::TypeKind::GenericFunction:
    case swift::TypeKind::Function: {
      swift::AnyFunctionType *any_function_type =
          swift_can_type->castTo<swift::AnyFunctionType>();
      std::string buffer;
      llvm::raw_string_ostream ostream(buffer);
      const swift::PrintOptions &print_options(
          SwiftASTContext::GetUserVisibleTypePrintingOptions(
              print_help_if_available));

      any_function_type->print(ostream, print_options);
      ostream.flush();
      if (buffer.empty() == false)
        s->Printf("%s\n", buffer.c_str());
    } break;
    case swift::TypeKind::Tuple: {
      swift::TupleType *tuple_type = swift_can_type->castTo<swift::TupleType>();
      std::string buffer;
      llvm::raw_string_ostream ostream(buffer);
      const swift::PrintOptions &print_options(
          SwiftASTContext::GetUserVisibleTypePrintingOptions(
              print_help_if_available));

      tuple_type->print(ostream, print_options);
      ostream.flush();
      if (buffer.empty() == false)
        s->Printf("%s\n", buffer.c_str());
    } break;
    case swift::TypeKind::BoundGenericClass:
    case swift::TypeKind::BoundGenericEnum:
    case swift::TypeKind::BoundGenericStruct: {
      swift::BoundGenericType *bound_generic_type =
          swift_can_type->castTo<swift::BoundGenericType>();
      swift::NominalTypeDecl *nominal_type_decl = bound_generic_type->getDecl();
      PrintSwiftNominalType(nominal_type_decl, s, print_help_if_available,
                            print_extensions_if_available);
    } break;
    case swift::TypeKind::BuiltinInteger: {
      swift::BuiltinIntegerType *builtin_integer_type =
          swift_can_type->castTo<swift::BuiltinIntegerType>();
      s->Printf("builtin integer type of width %u bits\n",
                builtin_integer_type->getWidth().getGreatestWidth());
      break;
    }
    case swift::TypeKind::BuiltinFloat: {
      swift::BuiltinFloatType *builtin_float_type =
          swift_can_type->castTo<swift::BuiltinFloatType>();
      s->Printf("builtin floating-point type of width %u bits\n",
                builtin_float_type->getBitWidth());
      break;
    }
    case swift::TypeKind::ProtocolComposition: {
      swift::ProtocolCompositionType *protocol_composition_type =
          swift_can_type->castTo<swift::ProtocolCompositionType>();
      std::string buffer;
      llvm::raw_string_ostream ostream(buffer);
      const swift::PrintOptions &print_options(
          SwiftASTContext::GetUserVisibleTypePrintingOptions(
              print_help_if_available));

      protocol_composition_type->print(ostream, print_options);
      ostream.flush();
      if (buffer.empty() == false)
        s->Printf("%s\n", buffer.c_str());
      break;
    }
    case swift::TypeKind::Existential: {
      swift::ExistentialType *existential_type =
          swift_can_type->castTo<swift::ExistentialType>();
      std::string buffer;
      llvm::raw_string_ostream ostream(buffer);
      const swift::PrintOptions &print_options(
          SwiftASTContext::GetUserVisibleTypePrintingOptions(
              print_help_if_available));

        existential_type->print(ostream, print_options);
      ostream.flush();
      if (buffer.empty() == false)
        s->Printf("%s\n", buffer.c_str());
      break;
    }
    default: {
      swift::NominalType *nominal_type =
          llvm::dyn_cast_or_null<swift::NominalType>(
              swift_can_type.getPointer());
      if (nominal_type) {
        swift::NominalTypeDecl *nominal_type_decl = nominal_type->getDecl();
        PrintSwiftNominalType(nominal_type_decl, s, print_help_if_available,
                              print_extensions_if_available);
      }
    } break;
    }
  }

  if (s->GetWrittenBytes() == initial_written_bytes)
    s->Printf("<could not resolve type>");
}

plugin::dwarf::DWARFASTParser *SwiftASTContext::GetDWARFParser() {
  if (auto ts = GetTypeSystemSwiftTypeRef())
    return ts->GetDWARFParser();
  return nullptr;
}

lldb::TargetWP SwiftASTContext::GetTargetWP() const {
  if (auto ts = GetTypeSystemSwiftTypeRef())
    return ts->GetTargetWP();
  return {};
}

std::vector<lldb::DataBufferSP> &
SwiftASTContext::GetASTVectorForModule(const Module *module) {
  return m_ast_file_data_map[const_cast<Module *>(module)];
}

SwiftASTContextForExpressions::SwiftASTContextForExpressions(
    std::string description, ModuleSP module_sp,
    TypeSystemSwiftTypeRefSP typeref_typesystem)
    : SwiftASTContext(std::move(description), module_sp, typeref_typesystem) {
  assert(llvm::isa<TypeSystemSwiftTypeRefForExpressions>(
      m_typeref_typesystem.lock().get()));
}

SwiftASTContextForExpressions::~SwiftASTContextForExpressions() {
  LOG_PRINTF(GetLog(LLDBLog::Types | LLDBLog::Expressions), "tearing down");
  auto swift_context = GetASTContext();
  if (!swift_context)
    return;
  // A RemoteASTContext associated with this swift::ASTContext has
  // to be destroyed before the swift::ASTContext is destroyed.
  if (TargetSP target_sp = GetTargetWP().lock()) {
    if (ProcessSP process_sp = target_sp->GetProcessSP())
      if (auto *runtime = SwiftLanguageRuntime::Get(process_sp))
        runtime->ReleaseAssociatedRemoteASTContext(*swift_context);
  } else {
    LOG_PRINTF(GetLog(LLDBLog::Types | LLDBLog::Expressions),
               "Failed to lock target in ~SwiftASTContextForExpressions().");
  }

  GetASTMap().Erase(*swift_context);
}

PersistentExpressionState *
SwiftASTContextForExpressions::GetPersistentExpressionState() {
  if (auto ts = GetTypeSystemSwiftTypeRef())
    return ts->GetPersistentExpressionState();
  return nullptr;
}

static void DescribeFileUnit(Stream &s, const swift::FileUnit *file_unit) {
  s.PutCString("kind = ");

  switch (file_unit->getKind()) {
  case swift::FileUnitKind::Source: {
    s.PutCString("Source, ");
    if (auto *source_file =
            llvm::dyn_cast<const swift::SourceFile>(file_unit)) {
      s.Printf("filename = \"%s\", ", source_file->getFilename().str().c_str());
      s.PutCString("source file kind = ");
      switch (source_file->Kind) {
      case swift::SourceFileKind::Library:
        s.PutCString("Library");
        break;
      case swift::SourceFileKind::Main:
        s.PutCString("Main");
        break;
      case swift::SourceFileKind::MacroExpansion:
        s.PutCString("Macro Expansion");
        break;
      case swift::SourceFileKind::SIL:
        s.PutCString("SIL");
        break;
      case swift::SourceFileKind::Interface:
        s.PutCString("Interface");
        break;
      case swift::SourceFileKind::DefaultArgument:
        s.PutCString("Default Argument");
        break;
      }
    }
  } break;
  case swift::FileUnitKind::Builtin: {
    s.PutCString("Builtin");
  } break;
  case swift::FileUnitKind::Synthesized: {
    s.PutCString("Synthesized");
  } break;
  case swift::FileUnitKind::SerializedAST:
  case swift::FileUnitKind::ClangModule: {
    if (file_unit->getKind() == swift::FileUnitKind::SerializedAST)
      s.PutCString("Serialized Swift AST, ");
    else
      s.PutCString("Clang module, ");
    auto *loaded_file = swift::cast<const swift::LoadedFile>(file_unit);
    s.Printf("filename = \"%s\"", loaded_file->getFilename().str().c_str());
  } break;
  case swift::FileUnitKind::DWARFModule:
    s.PutCString("DWARF");
  };
  s.PutCString(";");
}

// Gets the full module name from the module passed in.

static void GetNameFromModule(swift::ModuleDecl *module, std::string &result) {
  result.clear();
  if (module) {
    const char *name = module->getName().get();
    if (!name)
      return;
    result.append(name);
    const clang::Module *clang_module = module->findUnderlyingClangModule();

    // At present, there doesn't seem to be any way to get the full module path
    // from the Swift side.
    if (!clang_module)
      return;

    for (const clang::Module *cur_module = clang_module->Parent; cur_module;
         cur_module = cur_module->Parent) {
      if (!cur_module->Name.empty()) {
        result.insert(0, 1, '.');
        result.insert(0, cur_module->Name);
      }
    }
  }
}

static swift::ModuleDecl *LoadOneModule(const SourceModule &module,
                                        SwiftASTContext &swift_ast_context,
                                        lldb::ProcessSP process_sp,
                                        bool import_dylibs,
                                        Status &error) {
  if (!module.path.size())
    return nullptr;

  error.Clear();
  ConstString toplevel = module.path.front();
  const std::string &m_description = swift_ast_context.GetDescription();
  LOG_PRINTF(GetLog(LLDBLog::Types | LLDBLog::Expressions),
             "Importing module %s", toplevel.AsCString());
  swift::ModuleDecl *swift_module = nullptr;
  auto *clangimporter = swift_ast_context.GetClangImporter();
  swift::ModuleDecl *imported_header_module =
      clangimporter ? clangimporter->getImportedHeaderModule() : nullptr;
  if (imported_header_module &&
      toplevel.GetStringRef() == imported_header_module->getName().str())
    swift_module = imported_header_module;
  else if (process_sp)
    swift_module = swift_ast_context.FindAndLoadModule(
        module, *process_sp.get(), import_dylibs, error);
  else {
    auto swift_module_or_err = swift_ast_context.GetModule(module);
    if (!swift_module_or_err)
      llvm::consumeError(swift_module_or_err.takeError());
    else
      swift_module = &*swift_module_or_err;
  }

  if (swift_module && IsDWARFImported(*swift_module)) {
    // This module was "imported" from DWARF. This basically means the
    // import as a Swift or Clang module failed. We have not yet
    // checked that DWARF debug info for this module actually exists
    // and there is no good mechanism to do so ahead of time.
    // We do know that we never load the stdlib from DWARF though.
    HEALTH_LOG_PRINTF(
        "Missing Swift module or Clang module found for \"%s\""
            ", \"imported\" via SwiftDWARFImporterDelegate. Hint: %s",
        toplevel.AsCString(),
        swift_ast_context.GetTriple().isOSDarwin()
            ? "Register Swift modules with the linker using -add_ast_path."
            : "Swift modules can be wrapped in object containers using "
              "-module-wrap and linked.");

    if (toplevel.GetStringRef() == swift::STDLIB_NAME)
      swift_module = nullptr;
  }

  if (!swift_module || !error.Success() || swift_ast_context.HasFatalErrors()) {
    LOG_PRINTF(GetLog(LLDBLog::Types | LLDBLog::Expressions),
               "Couldn't import module %s: %s", toplevel.AsCString(),
               error.AsCString());

    if (!swift_module || swift_ast_context.HasFatalErrors()) {
      return nullptr;
    }
  }

  if (GetLog(LLDBLog::Types | LLDBLog::Expressions)) {
    StreamString ss;
    for (const swift::FileUnit *file_unit : swift_module->getFiles())
      DescribeFileUnit(ss, file_unit);
    LOG_PRINTF(GetLog(LLDBLog::Types | LLDBLog::Expressions),
               "Imported module %s from {%s}", module.path.front().AsCString(),
               ss.GetData());
  }
  return swift_module;
}

bool SwiftASTContextForExpressions::GetImplicitImports(
    SymbolContext &sc, lldb::ProcessSP process_sp,
    llvm::SmallVectorImpl<swift::AttributedImport<swift::ImportedModule>>
        &modules,
    Status &error) {
  if (!GetCompileUnitImports(sc, process_sp, modules, error))
    return false;

  // Get the hand-loaded modules from the SwiftPersistentExpressionState.
  for (auto &module_pair : m_hand_loaded_modules) {
    auto &attributed_import = module_pair.second;

    // If the ImportedModule in the SwiftPersistentExpressionState has a
    // non-null ModuleDecl, add it to the ImplicitImports list.
    if (attributed_import.module.importedModule) {
      modules.emplace_back(attributed_import);
      continue;
    }

    // Otherwise, try reloading the ModuleDecl using the module name.
    SourceModule module_info;
    module_info.path.emplace_back(module_pair.first());
    auto *module = LoadOneModule(module_info, *this, process_sp,
                                 /*import_dylibs=*/false, error);
    if (!module)
      return false;

    attributed_import.module = swift::ImportedModule(module);
    modules.emplace_back(attributed_import);
  }
  return true;
}

void SwiftASTContextForExpressions::LoadImplicitModules(
    TargetSP target, ProcessSP process, ExecutionContextScope &exe_scope) {
  auto load_module = [&](ConstString module_name) {
    SourceModule module_info;
    module_info.path.push_back(module_name);
    Status err;
    LoadOneModule(module_info, *this, process, true, err);
    if (err.Fail()) {
      LOG_PRINTF(GetLog(LLDBLog::Types),
                 "Could not load module %s implicitly, error: %s",
                 module_name.GetCString(), err.AsCString());
      return;
    }

    auto module_or_err = GetModule(module_info);
    if (!module_or_err) {
      std::string error = llvm::toString(module_or_err.takeError());
      LOG_PRINTF(
          GetLog(LLDBLog::Types),
          "Could not add hand loaded module %s to persistent state, error: %s",
          module_name.GetCString(), error.c_str());
      return;
    }
    swift::ModuleDecl *module = &*module_or_err;

    AddHandLoadedModule(module_name, swift::ImportedModule(module));
  };

  load_module(ConstString(swift::SWIFT_STRING_PROCESSING_NAME));
  load_module(ConstString(swift::SWIFT_CONCURRENCY_NAME));
}

bool SwiftASTContextForExpressions::CacheUserImports(
    lldb::ProcessSP process_sp, swift::SourceFile &source_file, Status &error) {
  llvm::SmallString<1> m_description;

  auto src_file_imports = source_file.getImports();

  /// Find all explicit imports in the expression.
  struct UserImportFinder : public swift::ASTWalker {
    llvm::SmallDenseSet<swift::ModuleDecl*, 1> imports;

    PreWalkAction walkToDeclPre(swift::Decl *D) override {
      if (auto *ID = llvm::dyn_cast<swift::ImportDecl>(D))
        if (auto *M = ID->getModule())
          imports.insert(M);
      return Action::Continue();
    }
  };
  UserImportFinder import_finder;
  source_file.walk(import_finder);
  
  for (const auto &attributed_import : src_file_imports) {
    swift::ModuleDecl *module = attributed_import.module.importedModule;
    if (module && import_finder.imports.count(module)) {
      std::string module_name;
      GetNameFromModule(module, module_name);
      if (!module_name.empty()) {
        SourceModule module_info;
        ConstString module_const_str(module_name);
        module_info.path.push_back(module_const_str);
        LOG_PRINTF(GetLog(LLDBLog::Types | LLDBLog::Expressions),
                   "Performing auto import on found module: %s.\n",
                   module_name.c_str());
        auto *module_decl = LoadOneModule(module_info, *this, process_sp,
                                          /*import_dylibs=*/true, error);
        if (!module_decl)
          return false;
        if (IsSerializedAST(*module_decl)) {
          // Parse additional search paths from the module.
          StringRef ast_file = module_decl->getModuleLoadedFilename();
          if (llvm::sys::path::is_absolute(ast_file)) {
            auto file_or_err =
                llvm::MemoryBuffer::getFile(ast_file, /*IsText=*/false,
                                            /*RequiresNullTerminator=*/false);
            if (!file_or_err.getError() && file_or_err->get()) {
              PathMappingList path_remap;
              llvm::SmallString<0> error;
              bool found_swift_modules = false;
              bool got_serialized_options = false;
              llvm::raw_svector_ostream errs(error);
              bool discover_implicit_search_paths = false;
              swift::CompilerInvocation &invocation = GetCompilerInvocation();
              StringRef module_filter;

              LOG_PRINTF(GetLog(LLDBLog::Types),
                         "Scanning for search paths in %s",
                         ast_file.str().c_str());
              if (DeserializeAllCompilerFlags(
                      invocation, ast_file, module_filter,
                      {file_or_err->get()->getBuffer()}, path_remap,
                      discover_implicit_search_paths, m_description.str().str(),
                      errs, got_serialized_options, found_swift_modules,
                      /*search_paths_only = */ true)) {
                LOG_PRINTF(GetLog(LLDBLog::Types), "Could not parse %s: %s",
                           ast_file.str().c_str(), error.str().str().c_str());
              }
              if (got_serialized_options)
                LogConfiguration();
            }
          }
        }
        // How do we tell we are in REPL or playground mode?
        AddHandLoadedModule(module_const_str, attributed_import);
      }
    }
  }
  return true;
}

bool SwiftASTContext::GetCompileUnitImports(
    const SymbolContext &sc, ProcessSP process_sp,
    llvm::SmallVectorImpl<swift::AttributedImport<swift::ImportedModule>>
        &modules,
    Status &error) {
  return GetCompileUnitImportsImpl(sc, process_sp, &modules, error);
}

void SwiftASTContext::PerformCompileUnitImports(const SymbolContext &sc,
                                                lldb::ProcessSP process_sp,
                                                Status &error) {
  GetCompileUnitImportsImpl(sc, process_sp, nullptr, error);
}

static std::pair<Module *, lldb::user_id_t>
GetCUSignature(CompileUnit &compile_unit) {
  return {compile_unit.GetModule().get(), compile_unit.GetID()};
}

bool SwiftASTContext::GetCompileUnitImportsImpl(
    const SymbolContext &sc, lldb::ProcessSP process_sp,
    llvm::SmallVectorImpl<swift::AttributedImport<swift::ImportedModule>>
        *modules,
    Status &error) {

  CompileUnit *compile_unit = sc.comp_unit;
  if (compile_unit && compile_unit->GetModule())
    // Check the cache if this compile unit's imports were previously
    // requested.  If the caller didn't request the list of imported
    // modules then there is nothing left to do for subsequent
    // GetCompileUnitImportsImpl() calls as the previously loaded
    // modules should still be loaded.  The fact the we
    // unconditionally return true does not matter because the only
    // way to get here is through void PerformCompileUnitImports(),
    // which discards the return value.
    if (!m_cu_imports.insert(GetCUSignature(*compile_unit)).second)
      // List of imports isn't requested and we already processed this CU?
      if (!modules)
        return true;

  // Import the Swift standard library and its dependencies.
  SourceModule swift_module;
  swift_module.path.emplace_back(swift::STDLIB_NAME);
  auto *stdlib = LoadOneModule(swift_module, *this, process_sp,
                               /*import_dylibs=*/true, error);
  if (!stdlib)
    return false;

  if (modules)
    modules->emplace_back(swift::ImportedModule(stdlib));

  if (!compile_unit || compile_unit->GetLanguage() != lldb::eLanguageTypeSwift)
    return true;

  auto cu_imports = compile_unit->GetImportedModules();
  if (cu_imports.size() == 0)
    return true;

  LOG_PRINTF(GetLog(LLDBLog::Types), "Importing dependencies of current CU");
  std::string category = "Importing dependencies for ";
  category += compile_unit->GetPrimaryFile().GetFilename().GetString();
  auto module_import_progress_raii = GetModuleImportProgressRAII(category);
 
  for (const SourceModule &module : cu_imports) {
    // When building the Swift stdlib with debug info these will
    // show up in "Swift.o", but we already imported them and
    // manually importing them will fail.
    // Also skip the "std" module, as the C++ standard library will be
    // imported as "CxxStdlib", which should also be imported.
    if (module.path.size() &&
        llvm::StringSwitch<bool>(module.path.front().GetStringRef())
            .Cases("Swift", "SwiftShims", "Builtin", "std", true)
            .Default(false))
      continue;

    auto *loaded_module = LoadOneModule(module, *this, process_sp,
                                        /*import_dylibs=*/false, error);
    if (!loaded_module)
      return false;

    if (modules)
      modules->emplace_back(swift::ImportedModule(loaded_module));
  }
  return true;
}

swift::Mangle::ManglingFlavor SwiftASTContext::GetManglingFlavor() {
  return GetASTContext()->LangOpts.hasFeature(swift::Feature::Embedded)
             ? swift::Mangle::ManglingFlavor::Embedded
             : swift::Mangle::ManglingFlavor::Default;
}
