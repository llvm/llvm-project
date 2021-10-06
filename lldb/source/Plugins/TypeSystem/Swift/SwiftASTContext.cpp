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

// C++ Includes
#include <mutex> // std::once
#include <queue>
#include <set>
#include <sstream>

#include "swift/AST/ASTContext.h"
#include "swift/AST/ASTDemangler.h"
#include "swift/AST/ASTMangler.h"
#include "swift/AST/DebuggerClient.h"
#include "swift/AST/Decl.h"
#include "swift/AST/DiagnosticEngine.h"
#include "swift/AST/DiagnosticsSema.h"
#include "swift/AST/ExistentialLayout.h"
#include "swift/AST/GenericParamList.h"
#include "swift/AST/GenericSignature.h"
#include "swift/AST/IRGenOptions.h"
#include "swift/AST/ImportCache.h"
#include "swift/AST/ModuleLoader.h"
#include "swift/AST/NameLookup.h"
#include "swift/AST/OperatorNameLookup.h"
#include "swift/AST/SearchPathOptions.h"
#include "swift/AST/SubstitutionMap.h"
#include "swift/AST/Type.h"
#include "swift/AST/Types.h"
#include "swift/ASTSectionImporter/ASTSectionImporter.h"
#include "swift/Basic/DiagnosticOptions.h"
#include "swift/Basic/Dwarf.h"
#include "swift/Basic/LangOptions.h"
#include "swift/Basic/Located.h"
#include "swift/Basic/Platform.h"
#include "swift/Basic/PrimarySpecificPaths.h"
#include "swift/Basic/SourceManager.h"
#include "swift/ClangImporter/ClangImporter.h"
#include "swift/Demangling/Demangle.h"
#include "swift/Demangling/ManglingMacros.h"
#include "swift/Frontend/Frontend.h"
#include "swift/Frontend/ModuleInterfaceLoader.h"
#include "swift/Frontend/PrintingDiagnosticConsumer.h"
#include "swift/IRGen/Linking.h"
#include "swift/SIL/SILModule.h"
#include "swift/Sema/IDETypeChecking.h"
#include "swift/Serialization/Validation.h"
#include "swift/SymbolGraphGen/SymbolGraphOptions.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Driver/Driver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TargetRegistry.h"
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
#include "Plugins/ExpressionParser/Clang/ClangUtil.h"
#include "Plugins/ExpressionParser/Swift/SwiftDiagnostic.h"
#include "Plugins/ExpressionParser/Swift/SwiftHost.h"
#include "Plugins/ExpressionParser/Swift/SwiftUserExpression.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/DumpDataExtractor.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/ThreadSafeDenseMap.h"
#include "lldb/Expression/DiagnosticManager.h"
#include "lldb/Expression/IRExecutionUnit.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/StringConvert.h"
#include "lldb/Host/XML.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SourceModule.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/TypeMap.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/ReproducerProvider.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/Timer.h"
#include "lldb/Utility/XcodeSDK.h"

#include "llvm/ADT/ScopeExit.h"

#include "Plugins/LanguageRuntime/Swift/SwiftLanguageRuntime.h"
#include "Plugins/Platform/MacOSX/PlatformDarwin.h"
#include "Plugins/SymbolFile/DWARF/DWARFASTParserClang.h"
#include "Plugins/SymbolFile/DWARF/DWARFASTParserSwift.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"

#define VALID_OR_RETURN(value)                                                 \
  do {                                                                         \
    if (HasFatalErrors()) {                                                    \
      return (value);                                                          \
    }                                                                          \
  } while (0)
#define VALID_OR_RETURN_VOID()                                                 \
  do {                                                                         \
    if (HasFatalErrors()) {                                                    \
      return;                                                                  \
    }                                                                          \
  } while (0);
#define VALID_OR_RETURN_CHECK_TYPE(type, value)                                \
  do {                                                                         \
    if (HasFatalErrors() || !type) {                                           \
      return (value);                                                          \
    }                                                                          \
  } while (0)

namespace {
/// This silly constexpr allows us to filter out the useless __FUNCTION__ name
/// of lambdas in the LOG_PRINTF macro.
bool constexpr IsLambda(const char *name) {
  return name[0] && name[0] == 'o' && name[1] && name[1] == 'p' && name[2] &&
         name[2] == 'e' && name[3] && name[3] == 'r' && name[4] &&
         name[4] == 'a' && name[5] && name[5] == 't' && name[6] &&
         name[6] == 'o' && name[7] && name[7] == 'r' && name[8] &&
         name[8] == '(' && name[9] && name[9] == ')';
}

/// Used to sort the log output.
std::recursive_mutex g_log_mutex;

} // namespace

/// Similar to LLDB_LOG, but with richer contextual information.
#define LOG_PRINTF(CHANNEL, FMT, ...)                                          \
  LOG_PRINTF_IMPL(CHANNEL, false, FMT, ##__VA_ARGS__)
#define LOG_VERBOSE_PRINTF(CHANNEL, FMT, ...)                                  \
  LOG_PRINTF_IMPL(CHANNEL, true, FMT, ##__VA_ARGS__)
#define LOG_PRINTF_IMPL(CHANNEL, VERBOSE, FMT, ...)                            \
  do {                                                                         \
    if (Log *log = lldb_private::GetLogIfAllCategoriesSet(CHANNEL))            \
      if (!(VERBOSE) || log->GetVerbose()) {                                   \
        std::lock_guard<std::recursive_mutex> locker(g_log_mutex);             \
        /* The format string is optimized for code size, not speed. */         \
        log->Printf("%s::%s%s" FMT, m_description.c_str(),                     \
                    IsLambda(__FUNCTION__) ? "" : __FUNCTION__,                \
                    (FMT && FMT[0] == '(') ? "" : "() -- ", ##__VA_ARGS__);    \
      }                                                                        \
  } while (0)

using namespace lldb;
using namespace lldb_private;

char SwiftASTContext::ID;
char SwiftASTContextForExpressions::ID;

CompilerType lldb_private::ToCompilerType(swift::Type qual_type) {
  return CompilerType(
      SwiftASTContext::GetSwiftASTContext(&qual_type->getASTContext()),
      qual_type.getPointer());
}

TypePayloadSwift::TypePayloadSwift(bool is_fixed_value_buffer) {
  SetIsFixedValueBuffer(is_fixed_value_buffer);
}

CompilerType SwiftASTContext::GetCompilerType(ConstString mangled_name) {
  return m_typeref_typesystem.GetTypeFromMangledTypename(mangled_name);
}

CompilerType SwiftASTContext::GetCompilerType(swift::TypeBase *swift_type) {
  return {this, swift_type};
}

swift::Type TypeSystemSwiftTypeRef::GetSwiftType(CompilerType compiler_type) {
  auto *ts = llvm::dyn_cast_or_null<TypeSystemSwiftTypeRef>(
      compiler_type.GetTypeSystem());
  if (!ts)
    return {};

  // FIXME: Suboptimal performance, because the ConstString is looked up again.
  ConstString mangled_name(
      reinterpret_cast<const char *>(compiler_type.GetOpaqueQualType()));
  return ts->m_swift_ast_context->ReconstructType(mangled_name);
}

swift::Type SwiftASTContext::GetSwiftType(CompilerType compiler_type) {
  if (llvm::dyn_cast_or_null<SwiftASTContext>(compiler_type.GetTypeSystem()))
    return reinterpret_cast<swift::TypeBase *>(
        compiler_type.GetOpaqueQualType());
  return {};
}

swift::Type SwiftASTContext::GetSwiftType(opaque_compiler_type_t opaque_type) {
  assert(opaque_type && *reinterpret_cast<const char *>(opaque_type) != '$' &&
         "wrong type system");
  return lldb_private::GetSwiftType(CompilerType(this, opaque_type));
}

swift::CanType
SwiftASTContext::GetCanonicalSwiftType(opaque_compiler_type_t opaque_type) {
  assert(!opaque_type || *reinterpret_cast<const char *>(opaque_type) != '$' &&
         "wrong type system");
  return lldb_private::GetCanonicalSwiftType(CompilerType(this, opaque_type));
}

ConstString SwiftASTContext::GetMangledTypeName(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(
      type, ConstString("<invalid Swift context or opaque type>"));
  return GetMangledTypeName(GetSwiftType({this, type}).getPointer());
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
typedef llvm::DenseMap<const swift::ASTContext *, EnumInfoCacheSP>
    ASTEnumInfoCacheMap;

static EnumInfoCache *GetEnumInfoCache(const swift::ASTContext *a) {
  static ASTEnumInfoCacheMap g_cache;
  static std::mutex g_mutex;
  std::lock_guard<std::mutex> locker(g_mutex);
  ASTEnumInfoCacheMap::iterator pos = g_cache.find(a);
  if (pos == g_cache.end()) {
    g_cache.insert(
        std::make_pair(a, std::shared_ptr<EnumInfoCache>(new EnumInfoCache())));
    return g_cache.find(a)->second.get();
  }
  return pos->second.get();
}

namespace {
bool IsDirectory(const FileSpec &spec) {
  return llvm::sys::fs::is_directory(spec.GetPath());
}
} // namespace

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

  static SwiftEnumDescriptor *CreateDescriptor(swift::ASTContext *ast,
                                               swift::CanType swift_can_type,
                                               swift::EnumDecl *enum_decl);

protected:
  SwiftEnumDescriptor(swift::ASTContext *ast, swift::CanType swift_can_type,
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
  SwiftEmptyEnumDescriptor(swift::ASTContext *ast,
                           swift::CanType swift_can_type,
                           swift::EnumDecl *enum_decl)
      : SwiftEnumDescriptor(ast, swift_can_type, enum_decl,
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
  SwiftCStyleEnumDescriptor(swift::ASTContext *ast,
                            swift::CanType swift_can_type,
                            swift::EnumDecl *enum_decl)
      : SwiftEnumDescriptor(ast, swift_can_type, enum_decl,
                            SwiftEnumDescriptor::Kind::CStyle),
        m_nopayload_elems_bitmask(), m_elements(), m_element_indexes() {
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "doing C-style enum layout for %s",
               GetTypeName().AsCString());

    SwiftASTContext *swift_ast_ctx = SwiftASTContext::GetSwiftASTContext(ast);
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

    LOG_PRINTF(LIBLLDB_LOG_TYPES, "m_nopayload_elems_bitmask = %s",
               Dump(m_nopayload_elems_bitmask).c_str());

    for (auto enum_case : elements_with_no_payload) {
      ConstString case_name(enum_case.decl->getBaseIdentifier().str());
      swift::ClusteredBitVector case_value =
          enum_impl_strategy.getBitPatternForNoPayloadElement(enum_case.decl);

      LOG_PRINTF(LIBLLDB_LOG_TYPES, "case_name = %s, unmasked value = %s",
                 case_name.AsCString(), Dump(case_value).c_str());

      case_value &= m_nopayload_elems_bitmask;

      LOG_PRINTF(LIBLLDB_LOG_TYPES, "case_name = %s, masked value = %s",
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
    LOG_PRINTF(LIBLLDB_LOG_TYPES,
               "C-style enum - inspecting data to find enum case for type %s",
               GetTypeName().AsCString());

    swift::ClusteredBitVector current_payload;
    lldb::offset_t offset = 0;
    for (size_t idx = 0; idx < data.GetByteSize(); idx++) {
      uint64_t byte = data.GetU8(&offset);
      current_payload.add(8, byte);
    }

    LOG_PRINTF(LIBLLDB_LOG_TYPES, "m_nopayload_elems_bitmask        = %s",
               Dump(m_nopayload_elems_bitmask).c_str());
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "current_payload                  = %s",
               Dump(current_payload).c_str());

    if (current_payload.size() != m_nopayload_elems_bitmask.size()) {
      LOG_PRINTF(LIBLLDB_LOG_TYPES,
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

    LOG_PRINTF(LIBLLDB_LOG_TYPES, "masked current_payload           = %s",
               Dump(current_payload).c_str());

    auto iter = m_elements.find(current_payload), end = m_elements.end();
    if (iter == end) {
      LOG_PRINTF(LIBLLDB_LOG_TYPES, "bitmask search failed");
      return nullptr;
    }
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "bitmask search success - found case %s",
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
  SwiftAllPayloadEnumDescriptor(swift::ASTContext *ast,
                                swift::CanType swift_can_type,
                                swift::EnumDecl *enum_decl)
      : SwiftEnumDescriptor(ast, swift_can_type, enum_decl,
                            SwiftEnumDescriptor::Kind::AllPayload) {
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "doing ADT-style enum layout for %s",
               GetTypeName().AsCString());

    SwiftASTContext *swift_ast_ctx = SwiftASTContext::GetSwiftASTContext(ast);
    swift::irgen::IRGenModule &irgen_module = swift_ast_ctx->GetIRGenModule();
    const swift::irgen::EnumImplStrategy &enum_impl_strategy =
        swift::irgen::getEnumImplStrategy(irgen_module, swift_can_type);
    llvm::ArrayRef<swift::irgen::EnumImplStrategy::Element>
        elements_with_payload = enum_impl_strategy.getElementsWithPayload();
    m_tag_bits = enum_impl_strategy.getTagBitsForPayloads();

    LOG_PRINTF(LIBLLDB_LOG_TYPES, "tag_bits = %s", Dump(m_tag_bits).c_str());

    auto module_ctx = enum_decl->getModuleContext();
    const bool has_payload = true;
    for (auto enum_case : elements_with_payload) {
      ConstString case_name(enum_case.decl->getBaseIdentifier().str());

      swift::EnumElementDecl *case_decl = enum_case.decl;
      assert(case_decl);
      auto arg_type = case_decl->getArgumentInterfaceType();
      CompilerType case_type;
      if (arg_type) {
        case_type = ToCompilerType(
            {swift_can_type->getTypeOfMember(module_ctx, case_decl, arg_type)
                 ->getCanonicalType()
                 .getPointer()});
      }

      const bool is_indirect =
          case_decl->isIndirect() || case_decl->getParentEnum()->isIndirect();

      LOG_PRINTF(LIBLLDB_LOG_TYPES,
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
    LOG_PRINTF(LIBLLDB_LOG_TYPES,
               "ADT-style enum - inspecting data to find enum case for type %s",
               GetTypeName().AsCString());

    // No elements, just fail.
    if (m_elements.size() == 0) {
      LOG_PRINTF(LIBLLDB_LOG_TYPES, "enum with no cases. getting out");
      return nullptr;
    }
    // One element, so it's got to be it.
    if (m_elements.size() == 1) {
      LOG_PRINTF(LIBLLDB_LOG_TYPES,
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
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "tag_bits        = %s",
               Dump(m_tag_bits).c_str());
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "current_payload = %s",
               Dump(current_payload).c_str());

    if (current_payload.size() != m_tag_bits.size()) {
      LOG_PRINTF(LIBLLDB_LOG_TYPES,
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
      LOG_PRINTF(LIBLLDB_LOG_TYPES,
                 "discriminator value of %" PRIu64 " too large, getting out",
                 (uint64_t)discriminator);
      return nullptr;
    } else {
      auto ptr = m_elements[discriminator].get();
      if (!ptr)
        LOG_PRINTF(LIBLLDB_LOG_TYPES,
                   "discriminator value of %" PRIu64
                   " acceptable, but null case matched - that's bad",
                   (uint64_t)discriminator);
      else
        LOG_PRINTF(LIBLLDB_LOG_TYPES,
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
  SwiftMixedEnumDescriptor(swift::ASTContext *ast,
                           swift::CanType swift_can_type,
                           swift::EnumDecl *enum_decl)
      : SwiftEnumDescriptor(ast, swift_can_type, enum_decl,
                            SwiftEnumDescriptor::Kind::Mixed),
        m_non_payload_cases(ast, swift_can_type, enum_decl),
        m_payload_cases(ast, swift_can_type, enum_decl) {}

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
  SwiftResilientEnumDescriptor(swift::ASTContext *ast,
                               swift::CanType swift_can_type,
                               swift::EnumDecl *enum_decl)
      : SwiftEnumDescriptor(ast, swift_can_type, enum_decl,
                            SwiftEnumDescriptor::Kind::Resilient) {
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "doing resilient enum layout for %s",
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
SwiftEnumDescriptor::CreateDescriptor(swift::ASTContext *ast,
                                      swift::CanType swift_can_type,
                                      swift::EnumDecl *enum_decl) {
  assert(ast);
  assert(enum_decl);
  assert(swift_can_type.getPointer());
  SwiftASTContext *swift_ast_ctx = SwiftASTContext::GetSwiftASTContext(ast);
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
    return new SwiftResilientEnumDescriptor(ast, swift_can_type, enum_decl);
  if (elements_with_no_payload.size() == 0) {
    // Nothing with no payload.. empty or all payloads?
    if (elements_with_payload.size() == 0)
      return new SwiftEmptyEnumDescriptor(ast, swift_can_type, enum_decl);
    return new SwiftAllPayloadEnumDescriptor(ast, swift_can_type, enum_decl);
  }

  // Something with no payload.. mixed or C-style?
  if (elements_with_payload.size() == 0)
    return new SwiftCStyleEnumDescriptor(ast, swift_can_type, enum_decl);
  return new SwiftMixedEnumDescriptor(ast, swift_can_type, enum_decl);
}

static SwiftEnumDescriptor *
GetEnumInfoFromEnumDecl(swift::ASTContext *ast, swift::CanType swift_can_type,
                        swift::EnumDecl *enum_decl) {
  return SwiftEnumDescriptor::CreateDescriptor(ast, swift_can_type, enum_decl);
}

SwiftEnumDescriptor *
SwiftASTContext::GetCachedEnumInfo(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, nullptr);

  EnumInfoCache *enum_info_cache = GetEnumInfoCache(GetASTContext());
  EnumInfoCache::const_iterator pos = enum_info_cache->find(type);
  if (pos != enum_info_cache->end())
    return pos->second.get();

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  if (!SwiftASTContext::IsFullyRealized(ToCompilerType({swift_can_type})))
    return nullptr;

  SwiftEnumDescriptorSP enum_info_sp;

  if (auto *enum_type = swift_can_type->getAs<swift::EnumType>()) {
    enum_info_sp.reset(GetEnumInfoFromEnumDecl(GetASTContext(), swift_can_type,
                                               enum_type->getDecl()));
  } else if (auto *bound_enum_type =
                 swift_can_type->getAs<swift::BoundGenericEnumType>()) {
    enum_info_sp.reset(GetEnumInfoFromEnumDecl(GetASTContext(), swift_can_type,
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

static bool IsDeviceSupport(const char *path) {
  // The old-style check, which we preserve for safety.
  if (path && strstr(path, "iOS DeviceSupport"))
    return true;

  // The new-style check, which should cover more devices.
  if (path)
    if (const char *Developer_Xcode = strstr(path, "Developer"))
      if (const char *DeviceSupport = strstr(Developer_Xcode, "DeviceSupport"))
        if (strstr(DeviceSupport, "Symbols"))
          return true;

  // Don't look in the simulator runtime frameworks either.  They
  // either duplicate what the SDK has, or for older simulators
  // conflict with them.
  if (path && strstr(path, ".simruntime/Contents/Resources/"))
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

#ifndef NDEBUG
SwiftASTContext::SwiftASTContext() : m_typeref_typesystem(this) {
  llvm::dbgs() << "Initialized mock SwiftASTContext\n";
}
#endif

SwiftASTContext::SwiftASTContext(std::string description, llvm::Triple triple,
                                 Target *target)
    : TypeSystemSwift(), m_typeref_typesystem(this),
      m_compiler_invocation_ap(new swift::CompilerInvocation()) {
  m_description = description;

  // Set the dependency tracker.
  if (auto g = repro::Reproducer::Instance().GetGenerator()) {
    repro::FileProvider &fp = g->GetOrCreate<repro::FileProvider>();
    m_dependency_tracker = std::make_unique<swift::DependencyTracker>(
        swift::IntermoduleDepTrackingMode::IncludeSystem,
        fp.GetFileCollector());
  }

  // Set the clang modules cache path.
  m_compiler_invocation_ap->setClangModuleCachePath(
      GetClangModulesCacheProperty());

  if (target)
    m_target_wp = target->shared_from_this();

  SetTriple(triple);
  swift::IRGenOptions &ir_gen_opts =
      m_compiler_invocation_ap->getIRGenOptions();
  ir_gen_opts.OutputKind = swift::IRGenOutputKind::Module;
  ir_gen_opts.UseJIT = true;
  ir_gen_opts.DWARFVersion = swift::DWARFVersion;
  // Allow deserializing @_implementationOnly dependencies
  // to avoid crashing due to module recovery issues.
  swift::LangOptions &lang_opts = m_compiler_invocation_ap->getLangOptions();
  lang_opts.AllowDeserializingImplementationOnly = true;
}

SwiftASTContext::~SwiftASTContext() {
  if (swift::ASTContext *ctx = m_ast_context_ap.get()) {
    // A RemoteASTContext associated with this swift::ASTContext has
    // to be destroyed before the swift::ASTContext is destroyed.
    if (TargetSP target_sp = m_target_wp.lock())
      if (ProcessSP process_sp = target_sp->GetProcessSP())
        if (auto *runtime = SwiftLanguageRuntime::Get(process_sp))
          runtime->ReleaseAssociatedRemoteASTContext(ctx);

    GetASTMap().Erase(ctx);
  }
}

const std::string &SwiftASTContext::GetDescription() const {
  return m_description;
}

namespace {
struct SDKTypeMinVersion {
  XcodeSDK::Type sdk_type;
  unsigned min_version_major;
  unsigned min_version_minor;
};
} // namespace

/// Return the SDKType (+minimum version needed for Swift support) for
/// the target triple, if that makes sense. Otherwise, return the
/// unknown sdk type.
static SDKTypeMinVersion GetSDKType(const llvm::Triple &target,
                                    const llvm::Triple &host) {
  // Only Darwin platforms know the concept of an SDK.
  auto host_os = host.getOS();
  if (host_os != llvm::Triple::OSType::MacOSX)
    return {XcodeSDK::Type::unknown, 0, 0};

  auto is_simulator = [&]() -> bool {
    return target.getEnvironment() == llvm::Triple::Simulator ||
           !target.getArchName().startswith("arm");
  };

  switch (target.getOS()) {
  case llvm::Triple::OSType::MacOSX:
  case llvm::Triple::OSType::Darwin:
    return {XcodeSDK::Type::MacOSX, 10, 10};
  case llvm::Triple::OSType::IOS:
    if (is_simulator())
      return {XcodeSDK::Type::iPhoneSimulator, 8, 0};
    return {XcodeSDK::Type::iPhoneOS, 8, 0};
  case llvm::Triple::OSType::TvOS:
    if (is_simulator())
      return {XcodeSDK::Type::AppleTVSimulator, 9, 0};
    return {XcodeSDK::Type::AppleTVOS, 9, 0};
  case llvm::Triple::OSType::WatchOS:
    if (is_simulator())
      return {XcodeSDK::Type::WatchSimulator, 2, 0};
    return {XcodeSDK::Type::watchOS, 2, 0};
  default:
    return {XcodeSDK::Type::unknown, 0, 0};
  }
}

/// Return the name of the OS-specific subdirectory containing the
/// Swift stdlib needed for \p target.
std::string SwiftASTContext::GetSwiftStdlibOSDir(const llvm::Triple &target,
                                                 const llvm::Triple &host) {
  auto sdk = GetSDKType(target, host);
  XcodeSDK::Info sdk_info;
  sdk_info.type = sdk.sdk_type;
  std::string sdk_name = XcodeSDK::GetCanonicalName(sdk_info);
  if (!sdk_name.empty())
    return sdk_name;
  return target.getOSName().str();
}

std::string SwiftASTContext::GetResourceDir(const llvm::Triple &triple) {
  static std::mutex g_mutex;
  std::lock_guard<std::mutex> locker(g_mutex);
  StringRef platform_sdk_path = GetPlatformSDKPath();
  std::string swift_stdlib_os_dir =
      GetSwiftStdlibOSDir(triple, HostInfo::GetArchitecture().GetTriple());

  // The resource dir depends on the SDK path and the expected os name.
  llvm::SmallString<128> key(platform_sdk_path);
  key.append(swift_stdlib_os_dir);
  static llvm::StringMap<std::string> g_resource_dir_cache;
  auto it = g_resource_dir_cache.find(key);
  if (it != g_resource_dir_cache.end())
    return it->getValue();

  auto value = GetResourceDir(
      platform_sdk_path, swift_stdlib_os_dir, GetSwiftResourceDir().GetPath(),
      HostInfo::GetXcodeContentsDirectory().GetPath(),
      PlatformDarwin::GetCurrentToolchainDirectory().GetPath(),
      PlatformDarwin::GetCurrentCommandLineToolsDirectory().GetPath());
  g_resource_dir_cache.insert({key, value});
  return g_resource_dir_cache[key];
}

std::string SwiftASTContext::GetResourceDir(StringRef platform_sdk_path,
                                            StringRef swift_stdlib_os_dir,
                                            std::string swift_dir,
                                            std::string xcode_contents_path,
                                            std::string toolchain_path,
                                            std::string cl_tools_path) {
  llvm::SmallString<16> m_description("SwiftASTContext");
  // First, check if there's something in our bundle.
  {
    FileSpec swift_dir_spec(swift_dir);
    if (swift_dir_spec) {
      LOG_PRINTF(LIBLLDB_LOG_TYPES, "trying ePathTypeSwiftDir: %s",
                 swift_dir_spec.GetCString());
      // We can't just check for the Swift directory, because that
      // always exists.  We have to look for "clang" inside that.
      FileSpec swift_clang_dir_spec = swift_dir_spec;
      swift_clang_dir_spec.AppendPathComponent("clang");

      if (IsDirectory(swift_clang_dir_spec)) {
        LOG_PRINTF(LIBLLDB_LOG_TYPES,
                   "found Swift resource dir via ePathTypeSwiftDir': %s",
                   swift_dir_spec.GetCString());
        return swift_dir_spec.GetPath();
      }
    }
  }

  // Nothing in our bundle. Are we in a toolchain that has its own Swift
  // compiler resource dir?

  {
    llvm::SmallString<256> path(toolchain_path);
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "trying toolchain path: %s", path.c_str());

    if (!path.empty()) {
      llvm::sys::path::append(path, "usr/lib/swift");
      LOG_PRINTF(LIBLLDB_LOG_TYPES, "trying toolchain-based lib path: %s",
                 path.c_str());

      if (IsDirectory(FileSpec(path))) {
        LOG_PRINTF(LIBLLDB_LOG_TYPES,
                   "found Swift resource dir via "
                   "toolchain path + 'usr/lib/swift': %s",
                   path.c_str());
        return std::string(path);
      }
    }
  }

  // We're not in a toolchain that has one. Use the Xcode default toolchain.

  {
    llvm::SmallString<256> path(xcode_contents_path);
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "trying Xcode path: %s", path.c_str());

    if (!path.empty()) {
      llvm::sys::path::append(path, "Developer",
                              "Toolchains/XcodeDefault.xctoolchain",
                              "usr/lib/swift");
      LOG_PRINTF(LIBLLDB_LOG_TYPES, "trying Xcode-based lib path: %s",
                 path.c_str());

      if (IsDirectory(FileSpec(path))) {
        StringRef resource_dir = path;
        llvm::sys::path::append(path, swift_stdlib_os_dir);
        std::string s(path);
        if (IsDirectory(FileSpec(path))) {
          LOG_PRINTF(LIBLLDB_LOG_TYPES,
                     "found Swift resource dir via "
                     "Xcode contents path + default toolchain "
                     "relative dir: %s",
                     resource_dir.str().c_str());
          return resource_dir.str();
        } else {
          // Search the SDK for a matching cross-SDK.
          path = platform_sdk_path;
          llvm::sys::path::append(path, "usr/lib/swift");
          StringRef resource_dir = path;
          llvm::sys::path::append(path, swift_stdlib_os_dir);
          if (IsDirectory(FileSpec(path))) {
            LOG_PRINTF(LIBLLDB_LOG_TYPES,
                       "found Swift resource dir via "
                       "Xcode contents path + cross-compilation SDK "
                       "relative dir: %s",
                       resource_dir.str().c_str());
            return resource_dir.str();
          }
        }
      }
    }
  }

  // We're not in Xcode. We might be in the command-line tools.

  {
    llvm::SmallString<256> path(cl_tools_path);
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "trying command-line tools path: %s",
               path.c_str());

    if (!path.empty()) {
      llvm::sys::path::append(path, "usr/lib/swift");
      LOG_PRINTF(LIBLLDB_LOG_TYPES,
                 "trying command-line tools-based lib path: %s", path.c_str());

      if (IsDirectory(FileSpec(path))) {
        LOG_PRINTF(LIBLLDB_LOG_TYPES,
                   "found Swift resource dir via command-line tools "
                   "path + usr/lib/swift: %s",
                   path.c_str());
        return std::string(path);
      }
    }
  }

  // We might be in the build-dir configuration for a
  // build-script-driven LLDB build, which has the Swift build dir as
  // a sibling directory to the lldb build dir.  This looks much
  // different than the install- dir layout that the previous checks
  // would try.
  {
    FileSpec faux_swift_dir_spec(swift_dir);
    if (faux_swift_dir_spec) {
// We can't use a C++11 stdlib regex feature here because it doesn't
// work on Ubuntu 14.04 x86_64.  Once we don't care about supporting
// that anymore, let's pull the code below back in since it is a
// simpler implementation using std::regex.
#if 0
      // Let's try to regex this.
      // We're looking for /some/path/lldb-{os}-{arch}, and want to
      // build the following:
      //    /some/path/swift-{os}-{arch}/lib/swift/{os}/{arch}
      // In a match, these are the following assignments for
      // backrefs:
      //   $1 - first part of path before swift build dir
      //   $2 - the host OS path separator character
      //   $3 - all the stuff that should come after changing
      //        lldb to swift for the lib dir.
      auto match_regex =
          std::regex("^(.+([/\\\\]))lldb-(.+)$");
      const std::string replace_format = "$1swift-$3";
      const std::string faux_swift_dir =
          faux_swift_dir_spec.GetCString();
      const std::string build_tree_resource_dir =
          std::regex_replace(faux_swift_dir, match_regex,
                             replace_format);
#else
      std::string build_tree_resource_dir;
      const std::string faux_swift_dir = faux_swift_dir_spec.GetCString();

      // Find something that matches lldb- (particularly,
      // the last one).
      const std::string lldb_dash("lldb-");
      auto lldb_pos = faux_swift_dir.rfind(lldb_dash);
      if ((lldb_pos != std::string::npos) && (lldb_pos > 0) &&
          ((faux_swift_dir[lldb_pos - 1] == '\\') ||
           (faux_swift_dir[lldb_pos - 1] == '/'))) {
        // We found something that matches ^.+[/\\]lldb-.+$
        std::ostringstream stream;
        // Take everything before lldb- (the path leading up to
        // the lldb dir).
        stream << faux_swift_dir.substr(0, lldb_pos);

        // replace lldb- with swift-.
        stream << "swift-";

        // and now tack on the same components from after
        // the lldb- part.
        stream << faux_swift_dir.substr(lldb_pos + lldb_dash.length());
        const std::string build_tree_resource_dir = stream.str();
        LOG_PRINTF(LIBLLDB_LOG_TYPES,
                   "trying ePathTypeSwiftDir regex-based build dir: %s",
                   build_tree_resource_dir.c_str());
        FileSpec swift_resource_dir_spec(build_tree_resource_dir.c_str());
        if (IsDirectory(swift_resource_dir_spec)) {
          LOG_PRINTF(LIBLLDB_LOG_TYPES,
                     "found Swift resource dir via "
                     "ePathTypeSwiftDir + inferred build-tree dir: %s",
                     swift_resource_dir_spec.GetCString());
          return swift_resource_dir_spec.GetCString();
        }
      }
#endif
    }
  }

  // We failed to find a reasonable Swift resource dir.
  LOG_PRINTF(LIBLLDB_LOG_TYPES, "failed to find a Swift resource dir");

  return {};
}

/// This code comes from CompilerInvocation.cpp (setRuntimeResourcePath).
static void ConfigureResourceDirs(swift::CompilerInvocation &invocation,
                                  FileSpec resource_dir, llvm::Triple triple) {
  // Make sure the triple is right:
  invocation.setTargetTriple(triple.str());
  invocation.setRuntimeResourcePath(resource_dir.GetPath().c_str());
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
  case swift::serialization::Status::SDKMismatch:
    return "The module file was built with a different SDK than the one in use "
           "to build the client.";
  case swift::serialization::Status::RevisionIncompatible:
    return "The module file was built with library evolution enabled by a "
           "different version of the compiler.";
  }
}

/// Initialize the compiler invocation with it the search paths from a
/// serialized AST.
/// \returns true on success.
static bool DeserializeCompilerFlags(swift::CompilerInvocation &invocation,
                                     StringRef section_data_ref, StringRef name,
                                     llvm::raw_ostream &error) {
  auto result = invocation.loadFromSerializedAST(section_data_ref);
  if (result == swift::serialization::Status::Valid)
    return true;

  error << "Could not deserialize " << name << ":\n"
        << getImportFailureString(result) << "\n";
  return false;
}

static void printASTValidationError(
    llvm::raw_ostream &errs,
    const swift::serialization::ValidationInfo &ast_info,
    const swift::serialization::ExtendedValidationInfo &ext_ast_info,
    Module &module, StringRef module_buf, bool invalid_name,
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
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES));
  LLDB_LOG(log, R"(Unable to load Swift AST for module "{0}" from library "{1}".
  {2}
  - targetTriple: {3}
  - shortVersion: {4}
  - bytes: {5} (module_buf bytes: {6})
  - SDK path: {7}
  - Clang Importer Options:
)",
           ast_info.name, module.GetSpecificationDescription(), error,
           ast_info.targetTriple, ast_info.shortVersion, ast_info.bytes,
           module_buf.size(), ext_ast_info.getSDKPath());
  for (StringRef ExtraOpt : ext_ast_info.getExtraClangImporterOptions())
    LLDB_LOG(log, "  -- {0}", ExtraOpt);
}

void SwiftASTContext::DiagnoseWarnings(Process &process, Module &module) const {
  for (const std::string &message : m_module_import_warnings)
    process.PrintWarningCantLoadSwiftModule(module, message);
}

/// Retrieve the serialized AST data blobs and initialize the compiler
/// invocation with the concatenated search paths from the blobs.
/// \returns true if an error was encountered.
static bool DeserializeAllCompilerFlags(swift::CompilerInvocation &invocation,
                                        Module &module,
                                        const std::string &m_description,
                                        llvm::raw_ostream &error,
                                        bool &got_serialized_options,
                                        bool &found_swift_modules) {
  bool found_validation_errors = false;
  got_serialized_options = false;
  auto ast_file_datas = module.GetASTData(eLanguageTypeSwift);
  LOG_PRINTF(LIBLLDB_LOG_TYPES, "Found %d AST file data entries.",
             (int)ast_file_datas.size());

  // If no N_AST symbols exist, this is not an error.
  if (ast_file_datas.empty())
    return false;

  // An AST section consists of one or more AST modules, optionally
  // with headers. Iterate over all AST modules.
  for (auto ast_file_data_sp : ast_file_datas) {
    llvm::StringRef buf((const char *)ast_file_data_sp->GetBytes(),
                        ast_file_data_sp->GetByteSize());
    swift::serialization::ValidationInfo info;
    for (; !buf.empty(); buf = buf.substr(info.bytes)) {
      swift::serialization::ExtendedValidationInfo extended_validation_info;
      info = swift::serialization::validateSerializedAST(
          buf, &extended_validation_info);
      bool invalid_ast = info.status != swift::serialization::Status::Valid;
      bool invalid_size = (info.bytes == 0) || (info.bytes > buf.size());
      bool invalid_name = info.name.empty();
      if (invalid_ast || invalid_size || invalid_name) {
        // Validation errors are diagnosed, but not fatal for the context.
        found_validation_errors = true;
        printASTValidationError(error, info, extended_validation_info, module,
                                buf, invalid_name, invalid_size);
        // If there's a size error, quit the loop early, otherwise try the next.
        if (invalid_size)
          break;
        continue;
      }

      found_swift_modules = true;
      StringRef moduleData = buf.substr(0, info.bytes);
      got_serialized_options |=
          DeserializeCompilerFlags(invocation, moduleData, info.name, error);
      LOG_PRINTF(LIBLLDB_LOG_TYPES, "SDK path from module \"%s\" was \"%s\".",
                 info.name.str().c_str(),
                 invocation.getSDKPath().str().c_str());
      // We will deduce a matching SDK path from DWARF later.
      invocation.setSDKPath("");
    }
  }
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
  return false;
}

std::array<StringRef, 2> macro_flags = { "-D", "-U" };
std::array<StringRef, 5> multi_arg_flags =
    { "-D", "-U", "-I", "-F", "-working-directory" };
std::array<StringRef, 5> args_to_unique =
    { "-D", "-U", "-I", "-F", "-fmodule-map-file=" };

bool IsMultiArgClangFlag(StringRef arg) {
  for (auto &flag : multi_arg_flags)
    if (flag == arg)
      return true;
  return false;
}

bool IsMacroDefinition(StringRef arg) {
  for (auto &flag : macro_flags)
    if (arg.startswith(flag))
      return true;
  return false;
}

bool ShouldUnique(StringRef arg) {
  for (auto &flag : args_to_unique)
    if (arg.startswith(flag))
      return true;
  return false;
}
} // namespace

// static
void SwiftASTContext::AddExtraClangArgs(const std::vector<std::string>& source,
                                        std::vector<std::string>& dest) {
  llvm::StringSet<> unique_flags;
  for (auto &arg : dest)
    unique_flags.insert(arg);

  llvm::SmallString<128> cur_working_dir;
  llvm::SmallString<128> clang_argument;
  for (const std::string &arg : source) {
    // Join multi-arg options for uniquing.
    clang_argument += arg;
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
    if (clang_argument.startswith("-Werror"))
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

void SwiftASTContext::AddExtraClangArgs(const std::vector<std::string> &ExtraArgs) {
  swift::ClangImporterOptions &importer_options = GetClangImporterOptions();
  AddExtraClangArgs(ExtraArgs, importer_options.ExtraArgs);
}

void SwiftASTContext::AddUserClangArgs(TargetProperties &props) {
  Args args(props.GetSwiftExtraClangFlags());
  std::vector<std::string> user_clang_flags;
  for (const auto &arg : args.entries())
    user_clang_flags.push_back(arg.ref().str());
  AddExtraClangArgs(user_clang_flags);
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
  } else if (arg.startswith("-")) {
    // Assume this is a compiler arg and not a path starting with "-".
    return;
  }
  // There is most probably a path in arg now.
  if (!llvm::sys::path::is_relative(arg))
    return;

  llvm::SmallString<128> joined_path;
  llvm::sys::path::append(joined_path, cur_working_dir, arg);
  llvm::sys::path::remove_dots(joined_path);
  clang_argument.resize(prefix.size());
  clang_argument.append(joined_path.begin(), joined_path.end());
}

void SwiftASTContext::ApplyDiagnosticOptions() {
  const auto &opts = GetCompilerInvocation().getDiagnosticOptions();
  if (opts.PrintDiagnosticNames)
    GetDiagnosticEngine().setPrintDiagnosticNames(true);

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
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "remapped %s -> %s",
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
      LOG_PRINTF(LIBLLDB_LOG_TYPES, "remapped %s -> %s%s", arg.str().c_str(),
                 prefix.str().c_str(), remapped.GetCString());
      arg_string = prefix.str() + remapped.GetCString();
    }
  }
}

/// Retrieve the .dSYM bundle for \p module.
static llvm::Optional<StringRef> GetDSYMBundle(Module &module) {
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

/// Detect whether a Swift module was "imported" by DWARFImporter.
/// All this *really* means is that it couldn't be loaded through any
/// other mechanism.
static bool IsDWARFImported(swift::ModuleDecl &module) {
  return std::any_of(module.getFiles().begin(), module.getFiles().end(),
                     [](swift::FileUnit *file_unit) {
                       return (file_unit->getKind() ==
                               swift::FileUnitKind::DWARFModule);
                     });
}

lldb::TypeSystemSP SwiftASTContext::CreateInstance(lldb::LanguageType language,
                                                   Module &module,
                                                   Target *target,
                                                   bool fallback) {
  if (!SwiftASTContextSupportsLanguage(language))
    return lldb::TypeSystemSP();

  std::string m_description;
  {
    llvm::raw_string_ostream ss(m_description);
    ss << "SwiftASTContext";
    if (fallback)
      ss << "ForExpressions";
    ss << '(' << '"';
    module.GetDescription(ss, eDescriptionLevelBrief);
    ss << '"' << ')';
  }
  LLDB_SCOPED_TIMERF("%s::CreateInstance", m_description.c_str());
  std::vector<std::string> module_search_paths;
  std::vector<std::pair<std::string, bool>> framework_search_paths;

  LOG_PRINTF(LIBLLDB_LOG_TYPES, "(Module)");

  auto logError = [&](const char *message) {
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "Failed to create module context - %s",
               message);
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
  assert((!fallback || target) && "fallback context must specify a target");
  std::shared_ptr<SwiftASTContext> swift_ast_sp(
      fallback ? (new SwiftASTContextForExpressions(m_description, *target))
               : (new SwiftASTContext(
                     m_description,
                     target ? target->GetArchitecture().GetTriple() : triple,
                     target)));
  bool suppress_config_log = false;
  auto defer_log = llvm::make_scope_exit([swift_ast_sp, &suppress_config_log] {
    // To avoid spamming the log with useless info, we don't log the
    // configuration if everything went fine and the current module
    // doesn't have any Swift contents (i.e., the shared cache dylibs).
    if (!suppress_config_log)
      swift_ast_sp->LogConfiguration();
  });

  // This is a module AST context, mark it as such.
  swift_ast_sp->m_is_scratch_context = false;
  swift_ast_sp->m_module = &module;
  swift_ast_sp->GetLanguageOptions().DebuggerSupport = true;
  swift_ast_sp->GetLanguageOptions().EnableAccessControl = false;
  swift_ast_sp->GetLanguageOptions().EnableTargetOSChecking = false;

  swift_ast_sp->SetTriple(triple, &module);

  bool set_triple = false;
  bool found_swift_modules = false;
  SymbolFile *sym_file = module.GetSymbolFile();
  std::string target_triple;

  if (sym_file) {
    bool got_serialized_options = false;
    llvm::SmallString<0> error;
    llvm::raw_svector_ostream errs(error);
    if (DeserializeAllCompilerFlags(
            swift_ast_sp->GetCompilerInvocation(), module, m_description, errs,
            got_serialized_options, found_swift_modules)) {
      // Validation errors are not fatal for the context.
      swift_ast_sp->m_module_import_warnings.push_back(std::string(error));
    }

    // Some of the bits in the compiler options we keep separately, so
    // we need to populate them from the serialized options:
    llvm::StringRef serialized_triple =
        swift_ast_sp->GetCompilerInvocation().getTargetTriple();
    if (serialized_triple.empty()) {
      LOG_PRINTF(LIBLLDB_LOG_TYPES, "Serialized triple was empty.");
    } else {
      LOG_PRINTF(LIBLLDB_LOG_TYPES, "Found serialized triple %s.",
                 serialized_triple.str().c_str());
      swift_ast_sp->SetTriple(llvm::Triple(serialized_triple), &module);
      set_triple = true;
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
        LOG_PRINTF(LIBLLDB_LOG_TYPES, "No serialized SDK path.");
      else
        LOG_PRINTF(LIBLLDB_LOG_TYPES, "Serialized SDK path is %s.",
                   serialized_sdk_path.str().c_str());

      // Force parsing of the CUs to extract the SDK info.
      XcodeSDK sdk;
      if (SymbolFile *sym_file = module.GetSymbolFile())
        for (unsigned i = 0; i < sym_file->GetNumCompileUnits(); ++i) {
          auto &cu = *sym_file->GetCompileUnitAtIndex(i);
          if (cu.GetLanguage() == lldb::eLanguageTypeSwift)
            sdk.Merge(sym_file->ParseXcodeSDK(cu));
        }

      std::string sdk_path = HostInfo::GetXcodeSDKPath(sdk).str();
      LOG_PRINTF(LIBLLDB_LOG_TYPES, "Host SDK path for sdk %s is %s.",
                 sdk.GetString().str().c_str(), sdk_path.c_str());
      if (FileSystem::Instance().Exists(sdk_path)) {
        // Note that this is not final. InitializeSearchPathOptions()
        // will set the SDK path based on the triple if this fails.
        swift_ast_sp->SetPlatformSDKPath(sdk_path);
        swift_ast_sp->GetCompilerInvocation().setSDKPath(sdk_path);
      }
    }
  }

  if (!set_triple) {
    llvm::Triple llvm_triple = swift_ast_sp->GetTriple();

    // LLVM wants this to be set to iOS or MacOSX; if we're working on
    // a bare-boards type image, change the triple for LLVM's benefit.
    if (llvm_triple.getVendor() == llvm::Triple::Apple &&
        llvm_triple.getOS() == llvm::Triple::UnknownOS) {
      if (llvm_triple.getArch() == llvm::Triple::arm ||
          llvm_triple.getArch() == llvm::Triple::thumb) {
        llvm_triple.setOS(llvm::Triple::IOS);
      } else {
        llvm_triple.setOS(llvm::Triple::MacOSX);
      }
      swift_ast_sp->SetTriple(llvm_triple, &module);
    }
  }

  triple = swift_ast_sp->GetTriple();
  std::string resource_dir = swift_ast_sp->GetResourceDir(triple);
  ConfigureResourceDirs(swift_ast_sp->GetCompilerInvocation(),
                        FileSpec(resource_dir), triple);

  // Apply the working directory to all relative paths.
  std::vector<std::string> DeserializedArgs = swift_ast_sp->GetClangArguments();
  swift_ast_sp->GetClangImporterOptions().ExtraArgs.clear();
  swift_ast_sp->AddExtraClangArgs(DeserializedArgs);
  if (target)
    swift_ast_sp->AddUserClangArgs(*target);
  else if (auto &global_target_properties = Target::GetGlobalProperties())
    swift_ast_sp->AddUserClangArgs(*global_target_properties);

  // Apply source path remappings found in the module's dSYM.
  swift_ast_sp->RemapClangImporterOptions(module.GetSourceMappingList());

  // Add Swift interfaces in the .dSYM at the end of the search paths.
  // .swiftmodules win over .swiftinterfaces, when they are loaded
  // directly from the .swift_ast section.
  //
  // FIXME: Since these paths also end up in the scratch context, we
  //        would need a mechanism to ensure that and newer versions
  //        (in the library evolution sense, not the date on disk) win
  //        over older versions of the same .swiftinterface.
  if (auto dsym = GetDSYMBundle(module)) {
    llvm::SmallString<256> path(*dsym);
    llvm::Triple triple(swift_ast_sp->GetTriple());
    StringRef arch = llvm::Triple::getArchTypeName(triple.getArch());
    llvm::sys::path::append(path, "Contents", "Resources", "Swift", arch);
    bool exists = false;
    llvm::sys::fs::is_directory(path, exists);
    if (exists)
      module_search_paths.push_back(std::string(path));
  }

  swift_ast_sp->InitializeSearchPathOptions(module_search_paths,
                                            framework_search_paths);
  if (!swift_ast_sp->GetClangImporter()) {
    LOG_PRINTF(LIBLLDB_LOG_TYPES,
               "(\"%s\") returning NULL - couldn't create a ClangImporter",
               module.GetFileSpec().GetFilename().AsCString("<anonymous>"));
    return {};
  }

  std::vector<std::string> module_names;
  swift_ast_sp->RegisterSectionModules(module, module_names);
  if (!module_names.size()) {
    // This dylib has no Swift contents; logging the configuration is pointless.
    suppress_config_log = true;
  } else {
    swift_ast_sp->ValidateSectionModules(module, module_names);
    if (lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES)) {
      std::lock_guard<std::recursive_mutex> locker(g_log_mutex);
      LOG_PRINTF(LIBLLDB_LOG_TYPES, "((Module*)%p, \"%s\") = %p",
                 static_cast<void *>(&module),
                 module.GetFileSpec().GetFilename().AsCString("<anonymous>"),
                 static_cast<void *>(swift_ast_sp.get()));
    }
  }

  if (swift_ast_sp->HasFatalErrors()) {
    logError(swift_ast_sp->GetFatalErrors().AsCString());
    return {};
  }

  {
    LLDB_SCOPED_TIMERF("%s (getStdlibModule)", m_description.c_str());
    const bool can_create = true;
    swift::ModuleDecl *stdlib =
        swift_ast_sp->m_ast_context_ap->getStdlibModule(can_create);
    if (!stdlib || IsDWARFImported(*stdlib)) {
      logError("couldn't load the Swift stdlib");
      return {};
    }
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

/// Scan a newly added lldb::Module fdor Swift modules and report any errors in
/// its module SwiftASTContext to Target.
static void
ProcessModule(ModuleSP module_sp, std::string m_description,
              bool use_all_compiler_flags, Target &target,
              std::vector<std::string> &module_search_paths,
              std::vector<std::pair<std::string, bool>> &framework_search_paths,
              std::vector<std::string> &extra_clang_args) {
  {
    llvm::raw_string_ostream ss(m_description);
    ss << "::ProcessModule(" << '"';
    module_sp->GetDescription(ss, eDescriptionLevelBrief);
    ss << '"' << ')';
  }

  const FileSpec &module_file = module_sp->GetFileSpec();
  std::string module_path = module_file.GetPath();

  // Add the containing framework to the framework search path.
  // Don't do that if this is the executable module, since it
  // might be buried in some framework that we don't care about.
  if (use_all_compiler_flags &&
      target.GetExecutableModulePointer() != module_sp.get()) {
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
        LOG_PRINTF(LIBLLDB_LOG_TYPES,
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

          // Never add framework paths pointing into the
          // system. These modules must be imported from the
          // SDK instead.
          if (!StringRef(parent_path).startswith("/System/Library") &&
              !IsDeviceSupport(parent_path.c_str())) {
            LOG_PRINTF(LIBLLDB_LOG_TYPES, "adding framework path \"%s\".",
                       framework_path.c_str());
            framework_search_paths.push_back(
                {std::move(parent_path), /*system*/ false});
          }
        }
      }
    }
  }

  // Skip images without a serialized Swift AST.
  if (!HasSwiftModules(*module_sp))
    return;

  auto type_system_or_err =
      module_sp->GetTypeSystemForLanguage(lldb::eLanguageTypeSwift);
  if (!type_system_or_err) {
    llvm::consumeError(type_system_or_err.takeError());
    return;
  }

  SwiftASTContext *ast_context =
      llvm::dyn_cast_or_null<SwiftASTContext>(&*type_system_or_err);

  if (!ast_context || ast_context->HasFatalErrors() ||
      !ast_context->GetClangImporter()) {
    // Make sure we warn about this module load failure, the one
    // that comes from loading types often gets swallowed up and not
    // seen, this is the only reliable point where we can show this.
    // But only do it once per UUID so we don't overwhelm the user
    // with warnings.
    UUID module_uuid(module_sp->GetUUID());
    bool unique_message =
        target.RegisterSwiftContextMessageKey(module_uuid.GetAsString());
    if (!unique_message)
      return;
    std::string buf;
    {
      llvm::raw_string_ostream ss(buf);
      module_sp->GetDescription(ss, eDescriptionLevelBrief);
      if (ast_context && ast_context->HasFatalErrors())
        ss << ": " << ast_context->GetFatalErrors().AsCString("unknown error");
    }
    target.GetDebugger().GetErrorStreamSP()->Printf(
        "Error while loading Swift module:\n%s\n"
        "Debug info from this module will be unavailable in the "
        "debugger.\n\n",
        buf.c_str());
    return;
  }

  if (ast_context->HasErrors())
    return;
  if (use_all_compiler_flags ||
      target.GetExecutableModulePointer() == module_sp.get()) {

    const auto &opts = ast_context->GetSearchPathOptions();
    module_search_paths.insert(module_search_paths.end(),
                               opts.ImportSearchPaths.begin(),
                               opts.ImportSearchPaths.end());
    for (const auto &fwsp : opts.FrameworkSearchPaths)
      framework_search_paths.push_back({fwsp.Path, fwsp.IsSystem});
    for (const std::string &arg : ast_context->GetClangArguments()) {
      extra_clang_args.push_back(arg);
      LOG_VERBOSE_PRINTF(LIBLLDB_LOG_TYPES, "adding Clang argument \"%s\".",
                         arg.c_str());
    }
  }
}

lldb::TypeSystemSP SwiftASTContext::CreateInstance(lldb::LanguageType language,
                                                   Target &target,
                                                   const char *extra_options) {
  if (!SwiftASTContextSupportsLanguage(language))
    return lldb::TypeSystemSP();

  LLDB_SCOPED_TIMER();
  std::string m_description = "SwiftASTContextForExpressions";
  std::vector<std::string> module_search_paths;
  std::vector<std::pair<std::string, bool>> framework_search_paths;

  // Make an AST but don't set the triple yet. We need to try and
  // detect if we have a iOS simulator.
  std::shared_ptr<SwiftASTContextForExpressions> swift_ast_sp(
      new SwiftASTContextForExpressions(m_description, target));
  auto defer_log = llvm::make_scope_exit(
      [swift_ast_sp] { swift_ast_sp->LogConfiguration(); });

  LOG_PRINTF(LIBLLDB_LOG_TYPES, "(Target)");

  auto logError = [&](const char *message) {
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "Failed to create scratch context - %s",
               message);
    // Avoid spamming the user with errors.
    if (!target.UseScratchTypesystemPerModule()) {
      StreamSP errs_sp = target.GetDebugger().GetAsyncErrorStream();
      errs_sp->Printf("Cannot create Swift scratch context (%s)", message);
    }
  };

  ArchSpec arch = target.GetArchitecture();
  if (!arch.IsValid()) {
    logError("invalid target architecture");
    return {};
  }

  // This is a scratch AST context, mark it as such.
  swift_ast_sp->m_is_scratch_context = true;

  swift_ast_sp->GetLanguageOptions().EnableTargetOSChecking = false;

  bool handled_sdk_path = false;
  const size_t num_images = target.GetImages().GetSize();

  // Set the SDK path prior to doing search paths.  Otherwise when we
  // create search path options we put in the wrong SDK path.
  FileSpec &target_sdk_spec = target.GetSDKPath();
  if (target_sdk_spec && FileSystem::Instance().Exists(target_sdk_spec)) {
    swift_ast_sp->SetPlatformSDKPath(target_sdk_spec.GetPath());
    handled_sdk_path = true;
  }

  auto warmup_astcontexts = [&]() {
    if (target.GetSwiftCreateModuleContextsInParallel()) {
      // The first call to GetTypeSystemForLanguage() on a module will
      // trigger the import (and thus most likely the rebuild) of all
      // the Clang modules that were imported in this module. This can
      // be a lot of work (potentially ten seconds per module), but it
      // can be performed in parallel.
      const unsigned threads =
          repro::Reproducer::Instance().IsReplaying() ? 1 : 0;
      llvm::ThreadPool pool(llvm::hardware_concurrency(threads));
      for (size_t mi = 0; mi != num_images; ++mi) {
        auto module_sp = target.GetImages().GetModuleAtIndex(mi);
        pool.async([=] {
          auto val_or_err =
              module_sp->GetTypeSystemForLanguage(lldb::eLanguageTypeSwift);
          if (!val_or_err) {
            llvm::consumeError(val_or_err.takeError());
          }
        });
      }
      pool.wait();
    }
  };

  if (!handled_sdk_path) {
    warmup_astcontexts();
    for (size_t mi = 0; mi != num_images; ++mi) {
      ModuleSP module_sp = target.GetImages().GetModuleAtIndex(mi);
      if (!HasSwiftModules(*module_sp))
        continue;

      auto type_system_or_err =
          module_sp->GetTypeSystemForLanguage(lldb::eLanguageTypeSwift);
      if (!type_system_or_err) {
        llvm::consumeError(type_system_or_err.takeError());
        continue;
      }

      auto *module_swift_ast =
          llvm::dyn_cast_or_null<SwiftASTContext>(&*type_system_or_err);
      if (!module_swift_ast || module_swift_ast->HasFatalErrors() ||
          !module_swift_ast->GetClangImporter())
        continue;

      StringRef platform_sdk_path = module_swift_ast->GetPlatformSDKPath();
      if (platform_sdk_path.empty())
        continue;

      handled_sdk_path = true;
      swift_ast_sp->SetPlatformSDKPath(platform_sdk_path);
      break;
    }
  }

  // First, prime the compiler with the options from the main executable:
  bool got_serialized_options = false;
  ModuleSP exe_module_sp(target.GetExecutableModule());

  // If we're debugging a testsuite, then treat the main test bundle
  // as the executable.
  if (exe_module_sp && IsUnitTestExecutable(*exe_module_sp)) {
    ModuleSP unit_test_module = GetUnitTestModule(target.GetImages());

    if (unit_test_module) {
      exe_module_sp = unit_test_module;
    }
  }

  {
    auto get_executable_triple = [&]() -> llvm::Triple {
      if (!exe_module_sp)
        return {};
      auto type_system_or_err =
          exe_module_sp->GetTypeSystemForLanguage(lldb::eLanguageTypeSwift);
      if (!type_system_or_err) {
        llvm::consumeError(type_system_or_err.takeError());
        return {};
      }
      auto *exe_ast_ctx =
          llvm::dyn_cast_or_null<SwiftASTContext>(&type_system_or_err.get());
      if (!exe_ast_ctx)
        return {};
      return exe_ast_ctx->GetLanguageOptions().Target;
    };

    llvm::Triple computed_triple;
    llvm::Triple target_triple = target.GetArchitecture().GetTriple();

    if (target.GetArchitecture().IsFullySpecifiedTriple()) {
      // If a fully specified triple was passed in, for example
      // through CreateTargetWithFileAndTargetTriple(), prefer that.
      LOG_PRINTF(LIBLLDB_LOG_TYPES, "Fully specified target triple %s.",
                 target_triple.str().c_str());
      computed_triple = target_triple;
    } else {
      LOG_PRINTF(LIBLLDB_LOG_TYPES, "Underspecified target triple %s.",
                 target_triple.str().c_str());
      PlatformSP platform_sp(target.GetPlatform());
      if (platform_sp && !target_triple.hasEnvironment()) {
        llvm::VersionTuple version =
            platform_sp->GetOSVersion(target.GetProcessSP().get());
        std::string buffer;
        llvm::raw_string_ostream(buffer)
            << target_triple.getArchName() << '-'
            << target_triple.getVendorName() << '-'
            << llvm::Triple::getOSTypeName(target_triple.getOS())
            << version.getAsString();
        computed_triple = llvm::Triple(buffer);
      } else {
        computed_triple = get_executable_triple();
      }
    }

    if (computed_triple.getOS() == llvm::Triple::MacOSX) {
      // Handle the case where an apparent macOS binary has been
      // force-loaded as a macCatalyst process. The Xcode test
      // runner works this way.
      llvm::Triple exe_triple = get_executable_triple();
      if (exe_triple.getOS() == llvm::Triple::IOS &&
          exe_triple.getEnvironment() == llvm::Triple::MacABI) {
        LOG_PRINTF(LIBLLDB_LOG_TYPES, "Adjusting triple to macCatalyst.");
        computed_triple.setOSAndEnvironmentName(
            exe_triple.getOSAndEnvironmentName());
      }
    }
    if (computed_triple == llvm::Triple()) {
      LOG_PRINTF(LIBLLDB_LOG_TYPES, "Failed to compute triple.");
      return {};
    }
    swift_ast_sp->SetTriple(computed_triple);
  }

  llvm::Triple triple = swift_ast_sp->GetTriple();
  std::string resource_dir = swift_ast_sp->GetResourceDir(triple);
  ConfigureResourceDirs(swift_ast_sp->GetCompilerInvocation(),
                        FileSpec(resource_dir), triple);

  const bool use_all_compiler_flags =
      !got_serialized_options || target.GetUseAllCompilerFlags();

  warmup_astcontexts();
  for (size_t mi = 0; mi != num_images; ++mi) {
    std::vector<std::string> extra_clang_args;
    ProcessModule(target.GetImages().GetModuleAtIndex(mi), m_description,
                  use_all_compiler_flags, target, module_search_paths,
                  framework_search_paths, extra_clang_args);
    swift_ast_sp->AddExtraClangArgs(extra_clang_args);
  }

  FileSpecList target_module_paths = target.GetSwiftModuleSearchPaths();
  for (size_t mi = 0, me = target_module_paths.GetSize(); mi != me; ++mi)
    module_search_paths.push_back(
        target_module_paths.GetFileSpecAtIndex(mi).GetPath());

  FileSpecList target_framework_paths = target.GetSwiftFrameworkSearchPaths();
  for (size_t fi = 0, fe = target_framework_paths.GetSize(); fi != fe; ++fi)
    framework_search_paths.push_back(
        {target_framework_paths.GetFileSpecAtIndex(fi).GetPath(),
         /*is_system*/ false});

  // Now fold any extra options we were passed. This has to be done
  // BEFORE the ClangImporter is made by calling GetClangImporter or
  // these options will be ignored.

  swift_ast_sp->AddUserClangArgs(target);

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

  // Apply source path remappings found in the target settings.
  swift_ast_sp->RemapClangImporterOptions(target.GetSourcePathMap());

  // This needs to happen once all the import paths are set, or
  // otherwise no modules will be found.
  swift_ast_sp->InitializeSearchPathOptions(module_search_paths,
                                            framework_search_paths);
  if (!swift_ast_sp->GetClangImporter()) {
    logError("couldn't create a ClangImporter");
    return {};
  }

  for (size_t mi = 0; mi != num_images; ++mi) {
    std::vector<std::string> module_names;
    auto module_sp = target.GetImages().GetModuleAtIndex(mi);
    swift_ast_sp->RegisterSectionModules(*module_sp, module_names);
  }

  LOG_PRINTF(LIBLLDB_LOG_TYPES, "((Target*)%p) = %p",
             static_cast<void *>(&target),
             static_cast<void *>(swift_ast_sp.get()));

  if (swift_ast_sp->HasFatalErrors()) {
    logError(swift_ast_sp->GetFatalErrors().AsCString());
    return {};
  }

  {
    LLDB_SCOPED_TIMERF("%s (getStdlibModule)", m_description.c_str());
    const bool can_create = true;
    swift::ModuleDecl *stdlib =
        swift_ast_sp->m_ast_context_ap->getStdlibModule(can_create);
    if (!stdlib || IsDWARFImported(*stdlib)) {
      logError("couldn't load the Swift stdlib");
      return {};
    }
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

Status SwiftASTContext::GetFatalErrors() {
  Status error;
  if (HasFatalErrors()) {
    error = m_fatal_errors;
    if (error.Success()) {
      // Retrieve the error message from the DiagnosticConsumer.
      DiagnosticManager diagnostic_manager;
      PrintDiagnostics(diagnostic_manager);
      error.SetErrorString(diagnostic_manager.GetString());
    }
  }
  return error;
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

bool SwiftASTContext::SetTriple(const llvm::Triple triple, Module *module) {
  VALID_OR_RETURN(false);
  if (triple.str().empty())
    return false;

  // The triple may change up until a swift::irgen::IRGenModule is created.
  if (m_ir_gen_module_ap.get()) {
    LOG_PRINTF(LIBLLDB_LOG_TYPES,
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
      TargetSP target_sp(m_target_wp.lock());
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
    LOG_PRINTF(LIBLLDB_LOG_TYPES,
               "Could not set pointer byte size using triple: %s",
               triple.str().c_str());
    m_pointer_byte_size = 0;
  }

  if (llvm::Triple(triple).getOS() == llvm::Triple::UnknownOS) {
    // This case triggers an llvm_unreachable() in the Swift compiler.
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "Cannot initialize Swift with an unknown OS");
    return false;
  }
  LOG_PRINTF(LIBLLDB_LOG_TYPES, "(\"%s\") setting to \"%s\"",
             triple.str().c_str(), adjusted_triple.str().c_str());

  m_compiler_invocation_ap->setTargetTriple(adjusted_triple);

  assert(GetTriple() == adjusted_triple);
  assert(!m_ast_context_ap ||
         (llvm::Triple(m_ast_context_ap->LangOpts.Target.getTriple()) ==
          adjusted_triple));

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

    FileSpec clang_dir_spec;
    clang_dir_spec = GetClangResourceDir();
    if (FileSystem::Instance().Exists(clang_dir_spec))
      clang_importer_options.OverrideResourceDir = clang_dir_spec.GetPath();
    clang_importer_options.DebuggerSupport = true;

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
    llvm::ArrayRef<std::string> extra_module_search_paths,
    llvm::ArrayRef<std::pair<std::string, bool>> extra_framework_search_paths) {
  LLDB_SCOPED_TIMER();
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
  std::string resource_dir = GetResourceDir(triple);
  ConfigureResourceDirs(GetCompilerInvocation(), FileSpec(resource_dir),
                        triple);

  std::string sdk_path = GetPlatformSDKPath().str();
  if (TargetSP target_sp = m_target_wp.lock())
    if (FileSpec &manual_override_sdk = target_sp->GetSDKPath()) {
      set_sdk = false;
      sdk_path = manual_override_sdk.GetPath();
      LOG_PRINTF(LIBLLDB_LOG_TYPES, "Override target.sdk-path \"%s\"",
                 sdk_path.c_str());
    }

  if (!set_sdk) {
    if (sdk_path.empty()) {
      XcodeSDK::Info info;
      info.type = XcodeSDK::GetSDKTypeForTriple(triple);
      XcodeSDK sdk(info);
      sdk_path = HostInfo::GetXcodeSDKPath(sdk).str();
    }
    if (sdk_path.empty()) {
      // This fallback is questionable. Perhaps it should be removed.
      XcodeSDK::Info info;
      info.type = XcodeSDK::GetSDKTypeForTriple(
          HostInfo::GetArchitecture().GetTriple());
      XcodeSDK sdk(info);
      sdk_path = std::string(HostInfo::GetXcodeSDKPath(sdk));
    }
    if (!sdk_path.empty()) {
      // Note that calling setSDKPath() also recomputes all paths that
      // depend on the SDK path including the
      // RuntimeLibraryImportPaths, which are *only* initialized
      // through this mechanism.
      LOG_PRINTF(LIBLLDB_LOG_TYPES, "Setting SDK path \"%s\"",
                 sdk_path.c_str());
      invocation.setSDKPath(sdk_path);
    }

    std::vector<std::string> &lpaths =
        invocation.getSearchPathOptions().LibrarySearchPaths;
    lpaths.insert(lpaths.begin(), "/usr/lib/swift");
  }

  llvm::StringMap<bool> processed;
  std::vector<std::string> &invocation_import_paths =
      invocation.getSearchPathOptions().ImportSearchPaths;
  // Add all deserialized paths to the map.
  for (const auto &path : invocation_import_paths)
    processed.insert({path, false});

  // Add/unique all extra paths.
  for (const auto &path : extra_module_search_paths) {
    auto it_notseen = processed.insert({path, false});
    if (it_notseen.second)
      invocation_import_paths.push_back(path);
  }

  // This preserves the IsSystem bit, but deduplicates entries ignoring it.
  processed.clear();
  auto &invocation_framework_paths =
      invocation.getSearchPathOptions().FrameworkSearchPaths;
  // Add all deserialized paths to the map.
  for (const auto &path : invocation_framework_paths)
    processed.insert({path.Path, path.IsSystem});

  // Add/unique all extra paths.
  for (const auto &path : extra_framework_search_paths) {
    auto it_notseen = processed.insert(path);
    if (it_notseen.second)
      invocation_framework_paths.push_back({path.first, path.second});
  }
}

namespace lldb_private {

class ANSIColorStringStream : public llvm::raw_string_ostream {
public:
  ANSIColorStringStream(bool colorize)
      : llvm::raw_string_ostream(m_buffer), m_colorize(colorize) {}
  /// Changes the foreground color of text that will be output from
  /// this point forward.
  /// \param Color ANSI color to use, the special SAVEDCOLOR can be
  ///        used to change only the bold attribute, and keep colors
  ///        untouched.
  /// \param Bold bold/brighter text, default false
  /// \param BG if true change the background,
  ///        default: change foreground
  /// \returns itself so it can be used within << invocations.
  raw_ostream &changeColor(enum Colors colors, bool bold = false,
                           bool bg = false) override {
    if (llvm::sys::Process::ColorNeedsFlush())
      flush();
    const char *colorcode;
    if (colors == SAVEDCOLOR)
      colorcode = llvm::sys::Process::OutputBold(bg);
    else
      colorcode =
          llvm::sys::Process::OutputColor(static_cast<char>(colors), bold, bg);
    if (colorcode) {
      size_t len = strlen(colorcode);
      write(colorcode, len);
    }
    return *this;
  }

  /// Resets the colors to terminal defaults. Call this when you are
  /// done outputting colored text, or before program exit.
  raw_ostream &resetColor() override {
    if (llvm::sys::Process::ColorNeedsFlush())
      flush();
    const char *colorcode = llvm::sys::Process::ResetColor();
    if (colorcode) {
      size_t len = strlen(colorcode);
      write(colorcode, len);
    }
    return *this;
  }

  /// Reverses the forground and background colors.
  raw_ostream &reverseColor() override {
    if (llvm::sys::Process::ColorNeedsFlush())
      flush();
    const char *colorcode = llvm::sys::Process::OutputReverse();
    if (colorcode) {
      size_t len = strlen(colorcode);
      write(colorcode, len);
    }
    return *this;
  }

  /// This function determines if this stream is connected to a "tty"
  /// or "console" window. That is, the output would be displayed to
  /// the user rather than being put on a pipe or stored in a file.
  bool is_displayed() const override { return m_colorize; }

  /// This function determines if this stream is displayed and
  /// supports colors.
  bool has_colors() const override { return m_colorize; }

protected:
  std::string m_buffer;
  bool m_colorize;
};

class StoringDiagnosticConsumer : public swift::DiagnosticConsumer {
public:
  StoringDiagnosticConsumer(SwiftASTContext &ast_context)
      : m_ast_context(ast_context), m_raw_diagnostics(), m_diagnostics(),
        m_num_errors(0), m_colorize(false) {
    m_ast_context.GetDiagnosticEngine().resetHadAnyError();
    m_ast_context.GetDiagnosticEngine().addConsumer(*this);
  }

  ~StoringDiagnosticConsumer() {
    m_ast_context.GetDiagnosticEngine().takeConsumers();
  }

  void handleDiagnostic(swift::SourceManager &source_mgr,
                        const swift::DiagnosticInfo &info) override {
    llvm::StringRef bufferName = "<anonymous>";
    unsigned bufferID = 0;
    std::pair<unsigned, unsigned> line_col = {0, 0};

    llvm::SmallString<256> text;
    {
      llvm::raw_svector_ostream out(text);
      swift::DiagnosticEngine::formatDiagnosticText(out, info.FormatString,
                                                    info.FormatArgs);
    }

    swift::SourceLoc source_loc = info.Loc;
    if (source_loc.isValid()) {
      bufferID = source_mgr.findBufferContainingLoc(source_loc);
      bufferName = source_mgr.getDisplayNameForLoc(source_loc);
      line_col = source_mgr.getPresumedLineAndColumnForLoc(source_loc);
    }

    if (line_col.first != 0) {
      ANSIColorStringStream os(m_colorize);

      // Determine what kind of diagnostic we're emitting, and whether
      // we want to use its fixits:
      bool use_fixits = false;
      llvm::SourceMgr::DiagKind source_mgr_kind;
      switch (info.Kind) {
      default:
      case swift::DiagnosticKind::Error:
        source_mgr_kind = llvm::SourceMgr::DK_Error;
        use_fixits = true;
        break;
      case swift::DiagnosticKind::Warning:
        source_mgr_kind = llvm::SourceMgr::DK_Warning;
        break;

      case swift::DiagnosticKind::Note:
        source_mgr_kind = llvm::SourceMgr::DK_Note;
        break;
      }

      // Swift may insert note diagnostics after an error diagnostic with fixits
      // related to that error. Check if the latest inserted diagnostic is an
      // error one, and that the diagnostic being processed is a note one that
      // points to the same error, and if so, copy the fixits from the note
      // diagnostic to the error one. There may be subsequent notes with fixits
      // related to the same error, but we only copy the first one as the fixits
      // are mutually exclusive (for example, one may suggest inserting a '?'
      // and the next may suggest inserting '!')
      if (!m_raw_diagnostics.empty() &&
          info.Kind == swift::DiagnosticKind::Note) {
        auto &last_diagnostic = m_raw_diagnostics.back();
        if (last_diagnostic.kind == swift::DiagnosticKind::Error &&
            last_diagnostic.fixits.empty() &&
            last_diagnostic.bufferID == bufferID &&
            last_diagnostic.column == line_col.second &&
            last_diagnostic.line == line_col.first)
          last_diagnostic.fixits.insert(last_diagnostic.fixits.end(),
                                        info.FixIts.begin(), info.FixIts.end());
      }

      // Translate ranges.
      llvm::SmallVector<llvm::SMRange, 2> ranges;
      for (auto R : info.Ranges)
        ranges.push_back(getRawRange(source_mgr, R));

      // Translate fix-its.
      llvm::SmallVector<llvm::SMFixIt, 2> fix_its;
      for (swift::DiagnosticInfo::FixIt F : info.FixIts)
        fix_its.push_back(getRawFixIt(source_mgr, F));

      // Display the diagnostic.
      auto message = source_mgr.GetMessage(source_loc, source_mgr_kind, text,
                                           ranges, fix_its);
      source_mgr.getLLVMSourceMgr().PrintMessage(os, message);

      // Use the llvm::raw_string_ostream::str() accessor as it will
      // flush the stream into our "message" and return us a reference
      // to "message".
      std::string &message_ref = os.str();

      if (message_ref.empty())
        m_raw_diagnostics.push_back(RawDiagnostic(
            std::string(text), info.Kind, bufferName, bufferID, line_col.first,
            line_col.second,
            use_fixits ? info.FixIts
                       : llvm::ArrayRef<swift::Diagnostic::FixIt>()));
      else
        m_raw_diagnostics.push_back(RawDiagnostic(
            message_ref, info.Kind, bufferName, bufferID, line_col.first,
            line_col.second,
            use_fixits ? info.FixIts
                       : llvm::ArrayRef<swift::Diagnostic::FixIt>()));
    } else {
      m_raw_diagnostics.push_back(RawDiagnostic(
          std::string(text), info.Kind, bufferName, bufferID, line_col.first,
          line_col.second, llvm::ArrayRef<swift::Diagnostic::FixIt>()));
    }

    if (info.Kind == swift::DiagnosticKind::Error)
      m_num_errors++;
  }

  void Clear() {
    m_ast_context.GetDiagnosticEngine().resetHadAnyError();
    m_raw_diagnostics.clear();
    m_diagnostics.clear();
    m_num_errors = 0;
  }

  unsigned NumErrors() {
    if (m_num_errors)
      return m_num_errors;
    else if (m_ast_context.GetASTContext()->hadError())
      return 1;
    else
      return 0;
  }

  static DiagnosticSeverity SeverityForKind(swift::DiagnosticKind kind) {
    switch (kind) {
    case swift::DiagnosticKind::Error:
      return eDiagnosticSeverityError;
    case swift::DiagnosticKind::Warning:
      return eDiagnosticSeverityWarning;
    case swift::DiagnosticKind::Note:
      return eDiagnosticSeverityRemark;
    case swift::DiagnosticKind::Remark:
      break;
    }

    llvm_unreachable("Unhandled DiagnosticKind in switch.");
  }

  void PrintDiagnostics(DiagnosticManager &diagnostic_manager,
                        uint32_t bufferID = UINT32_MAX, uint32_t first_line = 0,
                        uint32_t last_line = UINT32_MAX) {
    bool added_one_diagnostic = !m_diagnostics.empty();

    for (std::unique_ptr<Diagnostic> &diagnostic : m_diagnostics) {
      diagnostic_manager.AddDiagnostic(std::move(diagnostic));
    }

    for (const RawDiagnostic &diagnostic : m_raw_diagnostics) {
      // We often make expressions and wrap them in some code.  When
      // we see errors we want the line numbers to be correct so we
      // correct them below. LLVM stores in SourceLoc objects as
      // character offsets so there is no way to get LLVM to move its
      // error line numbers around by adjusting the source location,
      // we must do it manually. We also want to use the same error
      // formatting as LLVM and Clang, so we must muck with the
      // string.

      const DiagnosticSeverity severity = SeverityForKind(diagnostic.kind);
      const DiagnosticOrigin origin = eDiagnosticOriginSwift;

      if (first_line > 0 && bufferID != UINT32_MAX) {
        // Make sure the error line is in range or in another file.
        if (diagnostic.bufferID == bufferID && !diagnostic.bufferName.empty() &&
            (diagnostic.line < first_line || diagnostic.line > last_line))
          continue;
        // Need to remap the error/warning to a different line.
        StreamString match;
        match.Printf("%s:%u:", diagnostic.bufferName.str().c_str(),
                     diagnostic.line);
        const size_t match_len = match.GetString().size();
        size_t match_pos = diagnostic.description.find(match.GetString().str());
        if (match_pos != std::string::npos) {
          // We have some <file>:<line>:" instances that need to be updated.
          StreamString fixed_description;
          size_t start_pos = 0;
          do {
            if (match_pos > start_pos)
              fixed_description.Printf(
                  "%s",
                  diagnostic.description.substr(start_pos, match_pos).c_str());
            fixed_description.Printf(
                "%s:%u:", diagnostic.bufferName.str().c_str(),
                diagnostic.line - first_line + 1);
            start_pos = match_pos + match_len;
            match_pos =
                diagnostic.description.find(match.GetString().str(), start_pos);
          } while (match_pos != std::string::npos);

          // Append any last remaining text.
          if (start_pos < diagnostic.description.size())
            fixed_description.Printf(
                "%s", diagnostic.description
                          .substr(start_pos,
                                  diagnostic.description.size() - start_pos)
                          .c_str());

          auto new_diagnostic = std::make_unique<SwiftDiagnostic>(
              fixed_description.GetData(), severity, origin, bufferID);
          for (auto fixit : diagnostic.fixits)
            new_diagnostic->AddFixIt(fixit);

          diagnostic_manager.AddDiagnostic(std::move(new_diagnostic));
          if (diagnostic.kind == swift::DiagnosticKind::Error)
            added_one_diagnostic = true;

        }
      }
    }

    // In general, we don't want to see diagnostics from outside of
    // the source text range of the actual user expression. But if we
    // didn't find any diagnostics in the text range, it's probably
    // because the source range was not specified correctly, and we
    // don't want to lose legit errors because of that. So in that
    // case we'll add them all here:
    if (!added_one_diagnostic) {
      // This will report diagnostic errors from outside the
      // expression's source range. Those are not interesting to
      // users, so we only emit them in debug builds.
      for (const RawDiagnostic &diagnostic : m_raw_diagnostics) {
        const DiagnosticSeverity severity = SeverityForKind(diagnostic.kind);
        const DiagnosticOrigin origin = eDiagnosticOriginSwift;
        diagnostic_manager.AddDiagnostic(diagnostic.description.c_str(),
                                         severity, origin);
      }
    }
  }

  bool GetColorize() const { return m_colorize; }

  bool SetColorize(bool b) {
    const bool old = m_colorize;
    m_colorize = b;
    return old;
  }

  void AddDiagnostic(std::unique_ptr<Diagnostic> diagnostic) {
    m_diagnostics.push_back(std::move(diagnostic));
  }

private:
  // We don't currently use lldb_private::Diagostic or any of the lldb
  // DiagnosticManager machinery to store diagnostics as they
  // occur. Instead, we store them in raw form using this struct, then
  // transcode them to SwiftDiagnostics in PrintDiagnostic.
  struct RawDiagnostic {
    RawDiagnostic(std::string in_desc, swift::DiagnosticKind in_kind,
                  llvm::StringRef in_bufferName, unsigned in_bufferID,
                  uint32_t in_line, uint32_t in_column,
                  llvm::ArrayRef<swift::Diagnostic::FixIt> in_fixits)
        : description(in_desc), kind(in_kind), bufferName(in_bufferName),
          bufferID(in_bufferID), line(in_line), column(in_column) {
      for (auto fixit : in_fixits) {
        fixits.push_back(fixit);
      }
    }
    std::string description;
    swift::DiagnosticKind kind;
    const llvm::StringRef bufferName;
    unsigned bufferID;
    uint32_t line;
    uint32_t column;
    std::vector<swift::DiagnosticInfo::FixIt> fixits;
  };
  typedef std::vector<RawDiagnostic> RawDiagnosticBuffer;
  typedef std::vector<std::unique_ptr<Diagnostic>> DiagnosticList;

  SwiftASTContext &m_ast_context;
  RawDiagnosticBuffer m_raw_diagnostics;
  DiagnosticList m_diagnostics;

  unsigned m_num_errors = 0;
  bool m_colorize;
};

/// Implements a swift::DWARFImporterDelegate to look up Clang types in DWARF.
///
/// During compile time, ClangImporter-imported Clang modules are compiled with
/// -gmodules, which emits a DWARF rendition of all types defined in the module
/// into the .pcm file. On Darwin, these types can be collected by
/// dsymutil. This delegate allows DWARFImporter to ask LLDB to look up a Clang
/// type by name, synthesize a Clang AST from it. DWARFImporter then hands this
/// Clang AST to ClangImporter to import the type into Swift.
class SwiftDWARFImporterDelegate : public swift::DWARFImporterDelegate {
  SwiftASTContext &m_swift_ast_ctx;
  using ModuleAndName = std::pair<const char *, const char *>;
  std::string m_description;

  /// Used to filter out types with mismatching kinds.
  bool HasTypeKind(TypeSP clang_type_sp, swift::ClangTypeKind kind) {
    CompilerType fwd_type = clang_type_sp->GetForwardCompilerType();
    clang::QualType qual_type = ClangUtil::GetQualType(fwd_type);
    switch (kind) {
    case swift::ClangTypeKind::Typedef:
      /*=swift::ClangTypeKind::ObjCClass:*/
      return !qual_type->isObjCObjectOrInterfaceType() &&
             !qual_type->getAs<clang::TypedefType>();
    case swift::ClangTypeKind::Tag:
      return !qual_type->isStructureOrClassType() &&
             !qual_type->isEnumeralType() && !qual_type->isUnionType();
    case swift::ClangTypeKind::ObjCProtocol:
      // Not implemented since Objective-C protocols aren't yet
      // described in DWARF.
      return true;
    }
  }

  clang::Decl *GetDeclForTypeAndKind(clang::QualType qual_type,
                                     swift::ClangTypeKind kind) {
    switch (kind) {
    case swift::ClangTypeKind::Typedef:
      /*=swift::ClangTypeKind::ObjCClass:*/
      if (auto *obj_type = qual_type->getAsObjCInterfaceType())
        return obj_type->getInterface();
      if (auto *typedef_type = qual_type->getAs<clang::TypedefType>())
        return typedef_type->getDecl();
      break;
    case swift::ClangTypeKind::Tag:
      return qual_type->getAsTagDecl();
    case swift::ClangTypeKind::ObjCProtocol:
      // Not implemented since Objective-C protocols aren't yet
      // described in DWARF.
      break;
    }
    return nullptr;
  }

  static CompilerContextKind
  GetCompilerContextKind(llvm::Optional<swift::ClangTypeKind> kind) {
    if (!kind)
      return CompilerContextKind::AnyType;
    switch (*kind) {
    case swift::ClangTypeKind::Typedef:
      /*=swift::ClangTypeKind::ObjCClass:*/
      return (CompilerContextKind)((uint16_t)CompilerContextKind::Any |
                                   (uint16_t)CompilerContextKind::Typedef |
                                   (uint16_t)CompilerContextKind::Struct);
      break;
    case swift::ClangTypeKind::Tag:
      return (CompilerContextKind)((uint16_t)CompilerContextKind::Any |
                                   (uint16_t)CompilerContextKind::Class |
                                   (uint16_t)CompilerContextKind::Struct |
                                   (uint16_t)CompilerContextKind::Union |
                                   (uint16_t)CompilerContextKind::Enum);
      // case swift::ClangTypeKind::ObjCProtocol:
      // Not implemented since Objective-C protocols aren't yet
      // described in DWARF.
    default:
      return CompilerContextKind::Invalid;
    }
  }

  /// Import \p qual_type from one clang ASTContext to another and
  /// add it to \p results if successful.
  void importType(clang::QualType qual_type, clang::ASTContext &from_ctx,
                  clang::ASTContext &to_ctx,
                  llvm::Optional<swift::ClangTypeKind> kind,
                  llvm::SmallVectorImpl<clang::Decl *> &results) {
    clang::ASTImporter importer(to_ctx,
                                to_ctx.getSourceManager().getFileManager(),
                                from_ctx,
                                from_ctx.getSourceManager().getFileManager(),
                                false);
    llvm::Expected<clang::QualType> clang_type(importer.Import(qual_type));
    if (!clang_type) {
      llvm::consumeError(clang_type.takeError());
      return;
    }

    // Retrieve the imported type's Decl.
    if (kind) {
      if (clang::Decl *clang_decl = GetDeclForTypeAndKind(*clang_type, *kind))
        results.push_back(clang_decl);
    } else {
      swift::ClangTypeKind kinds[] = {
          swift::ClangTypeKind::Typedef, // =swift::ClangTypeKind::ObjCClass,
          swift::ClangTypeKind::Tag, swift::ClangTypeKind::ObjCProtocol};
      for (auto kind : kinds)
        if (clang::Decl *clang_decl = GetDeclForTypeAndKind(*clang_type, kind))
          results.push_back(clang_decl);
    }
  }

public:
  SwiftDWARFImporterDelegate(SwiftASTContext &swift_ast_ctx)
      : m_swift_ast_ctx(swift_ast_ctx),
        m_description(swift_ast_ctx.GetDescription() +
                      "::SwiftDWARFImporterDelegate") {}

  /// Look up a clang::Decl by name.
  ///
  /// There are two primary ways that this delegate method is called:
  ///
  ///    1. When resolving a type from a mangled name. In this case \p
  ///       kind will be known, but the owning module of a Clang type
  ///       in a mangled name is always __ObjC or __C.
  ///
  ///    2. When resolving a type from a serialized module
  ///       cross reference. In this case \c kind will be unspecified,
  ///       but the (top-level) module that the type is defined in
  ///       will be known.
  ///
  /// The following diagram shows how the various components
  /// interact. All paths lead to a call to the function
  /// \c ClangImporter::Implementation::importDeclReal(), which turns
  /// a \c clang::Decl into a \c swift::Decl.  The return paths leading
  /// back from \c importDeclReal() are omitted from the diagram. Also
  /// some auxiliary intermediate function calls are be omitted for
  /// brevity.
  ///
  /// \verbatim
  /// ╔═LLDB═════════════════════════════════════════════════════════════════╗
  /// ║                                                                      ║
  /// ║  ┌─DWARFASTParserSwift──────────┐   ┌─DWARFImporterDelegate──────┐   ║
  /// ║  │                              │   │                            │   ║
  /// ║  │ GetTypeFromMangledTypename() │ ┌─├→lookupValue()─────┐        │   ║
  /// ║  │             │                │ │ │                   │        │   ║
  /// ║  └─────────────┬────────────────┘ │ └───────────────────┬────────┘   ║
  /// ║                │                  │                     │            ║
  /// ╚════════════════╤══════════════════╧═════════════════════╤════════════╝
  ///                  │                  │                     │
  /// ╔═Swift Compiler═╤══════════════════╧═════════════════════╤════════════╗
  /// ║                │                  │                     │            ║
  /// ║  ┌─ASTDemangler┬─────────────┐    │ ┌─ClangImporter─────┬────┐       ║
  /// ║  │             ↓             │    │ │                   │    │       ║
  /// ║  │ findForeignTypeDecl()─────├──────├→lookupTypeDecl()  │    │       ║
  /// ║  │                           │    │ │      ⇣            ↓    │       ║
  /// ║  └───────────────────────────┘    └─┤─lookupTypeDeclDWARF()  │       ║
  /// ║                                     │      ↓                 │       ║
  /// ║                                     │ *importDeclReal()*     │       ║
  /// ║                                     │      ↑                 │       ║
  /// ║                                     │ lookupValueDWARF()     │       ║
  /// ║                                     │      ↑                 │       ║
  /// ║                                     └──────┴─────────────────┘       ║
  /// ║                                            │                         ║
  /// ║  ┌─Deserialization─────────┐               └──────────────────────┐  ║
  /// ║  │ loadAllMembers()        │                                      │  ║
  /// ║  │        ↓                │  ┌─ModuleDecl────┐ ┌─DWARFModuleUnit─┴┐ ║
  /// ║  │ resolveCrossReference()─├──├→lookupValue()─├─├→lookupValue()───┘│ ║
  /// ║  │                         │  └───────────────┘ └──────────────────┘ ║
  /// ║  └─────────────────────────┘                                         ║
  /// ╚══════════════════════════════════════════════════════════════════════╝
  /// \endverbatim
  void lookupValue(StringRef name, llvm::Optional<swift::ClangTypeKind> kind,
                   StringRef inModule,
                   llvm::SmallVectorImpl<clang::Decl *> &results) override {
    LLDB_SCOPED_TIMER();
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "(\"%s\")", name.str().c_str());

    // We will not find any Swift types in the Clang compile units.
    ConstString name_cs(name);
    if (SwiftLanguageRuntime::IsSwiftMangledName(name_cs.GetStringRef()))
      return;

    auto clang_importer = m_swift_ast_ctx.GetClangImporter();
    if (!clang_importer)
      return;

    // Find the type in the debug info.
    TypeMap clang_types;
    ConstString module_cs(inModule);

    llvm::SmallVector<CompilerContext, 3> decl_context;
    // Perform a lookup in a specific module, if requested.
    if (!inModule.empty())
      decl_context.push_back({CompilerContextKind::Module, module_cs});
    // Swift doesn't keep track of submodules.
    decl_context.push_back({CompilerContextKind::AnyModule, ConstString()});
    decl_context.push_back({GetCompilerContextKind(kind), name_cs});
    llvm::DenseSet<SymbolFile *> searched_symbol_files;
    auto search = [&](Module &module) {
      module.FindTypes(decl_context,
                       TypeSystemClang::GetSupportedLanguagesForTypes(),
                       searched_symbol_files, clang_types);
    };
    if (Module *module = m_swift_ast_ctx.GetModule())
      search(*module);
    else if (TargetSP target_sp = m_swift_ast_ctx.GetTarget().lock()) {
      // In a scratch context, check the module's DWARFImporterDelegates first.
      //
      // It's a common pattern that a type is revisited immediately
      // after looking it up in a per-module context in the scratch
      // context for dynamic type resolution.
      auto images = target_sp->GetImages();
      for (size_t i = 0; i != images.GetSize(); ++i) {
        auto module_sp = images.GetModuleAtIndex(i);
        auto ts = module_sp->GetTypeSystemForLanguage(lldb::eLanguageTypeSwift);
        if (!ts) {
          llvm::consumeError(ts.takeError());
          continue;
        }
        auto *swift_ast_ctx = static_cast<SwiftASTContext *>(&*ts);
        auto *dwarf_imp = static_cast<SwiftDWARFImporterDelegate *>(
            swift_ast_ctx->GetDWARFImporterDelegate());
        if (!dwarf_imp || dwarf_imp == this)
          continue;

        llvm::SmallVector<clang::Decl *, 2> module_results;
        dwarf_imp->lookupValue(name, kind, inModule, module_results);
        if (!module_results.size())
          continue;

        auto *from_clang_importer = swift_ast_ctx->GetClangImporter();
        if (!from_clang_importer)
          continue;
        auto &from_ctx = from_clang_importer->getClangASTContext();
        auto &to_ctx = clang_importer->getClangASTContext();
        for (clang::Decl *decl : module_results) {
          clang::QualType qual_type;
          if (auto *interface = llvm::dyn_cast<clang::ObjCInterfaceDecl>(decl))
            qual_type = {interface->getTypeForDecl(), 0};
          if (auto *type = llvm::dyn_cast<clang::TypeDecl>(decl))
            qual_type = {type->getTypeForDecl(), 0};
          importType(qual_type, from_ctx, to_ctx, kind, results);
        }
        // Cut the search short after we found the first result.
        if (results.size())
          break;
      }
      LOG_PRINTF(LIBLLDB_LOG_TYPES, "%zu types collected.", results.size());
      return;
    }

    clang_types.ForEach([&](lldb::TypeSP &clang_type_sp) {
      if (!clang_type_sp)
        return true;

      // Filter out types with a mismatching type kind.
      if (kind && HasTypeKind(clang_type_sp, *kind))
        return true;

      // Realize the full type.
      CompilerType compiler_type = clang_type_sp->GetFullCompilerType();

      // Filter our non-Clang types.
      auto *type_system = llvm::dyn_cast_or_null<TypeSystemClang>(
          compiler_type.GetTypeSystem());
      if (!type_system)
        return true;

      // Import the type into the DWARFImporter's context.
      clang::ASTContext &to_ctx = clang_importer->getClangASTContext();
      clang::ASTContext &from_ctx = type_system->getASTContext();

      clang::QualType qual_type = ClangUtil::GetQualType(compiler_type);
      importType(qual_type, from_ctx, to_ctx, kind, results);

      return true;
    });

    LOG_PRINTF(LIBLLDB_LOG_TYPES, "%zu types from debug info.", results.size());
  }
};
} // namespace lldb_private

swift::ASTContext *SwiftASTContext::GetASTContext() {
  assert(m_initialized_search_path_options &&
         m_initialized_clang_importer_options &&
         "search path options must be initialized before ClangImporter");

  if (m_ast_context_ap.get())
    return m_ast_context_ap.get();

  LLDB_SCOPED_TIMER();
  m_ast_context_ap.reset(swift::ASTContext::get(
      GetLanguageOptions(), GetTypeCheckerOptions(), GetSearchPathOptions(),
      GetClangImporterOptions(), GetSymbolGraphOptions(),
      GetSourceManager(), GetDiagnosticEngine()));
  m_diagnostic_consumer_ap.reset(new StoringDiagnosticConsumer(*this));

  if (getenv("LLDB_SWIFT_DUMP_DIAGS")) {
    // NOTE: leaking a swift::PrintingDiagnosticConsumer() here, but
    // this only gets enabled when the above environment variable is
    // set.
    GetDiagnosticEngine().addConsumer(*new swift::PrintingDiagnosticConsumer());
  }

  // Create the ClangImporter and determine the Clang module cache path.
  std::string moduleCachePath = "";
  std::unique_ptr<swift::ClangImporter> clang_importer_ap;
  auto &clang_importer_options = GetClangImporterOptions();
  if (!m_ast_context_ap->SearchPathOpts.SDKPath.empty() || TargetHasNoSDK()) {
    if (!clang_importer_options.OverrideResourceDir.empty()) {
      // Create the DWARFImporterDelegate.
      const auto &props = ModuleList::GetGlobalModuleListProperties();
      if (props.GetUseSwiftDWARFImporter())
        m_dwarf_importer_delegate_up =
            std::make_unique<SwiftDWARFImporterDelegate>(*this);
      clang_importer_ap = swift::ClangImporter::create(
          *m_ast_context_ap, "",
          m_dependency_tracker.get(), m_dwarf_importer_delegate_up.get());

      // Handle any errors.
      if (!clang_importer_ap || HasErrors()) {
        std::string message;
        if (!HasErrors()) {
          message = "failed to create ClangImporter.";
          m_module_import_warnings.push_back(message);
        } else {
          DiagnosticManager diagnostic_manager;
          PrintDiagnostics(diagnostic_manager);
          std::string underlying_error = diagnostic_manager.GetString();
          message = "failed to initialize ClangImporter: ";
          message += underlying_error;
          m_module_import_warnings.push_back(underlying_error);
        }
        LOG_PRINTF(LIBLLDB_LOG_TYPES, "%s", message.c_str());
      }
      if (clang_importer_ap)
        moduleCachePath = swift::getModuleCachePathFromClang(
            clang_importer_ap->getClangInstance());
    }
  }

  if (moduleCachePath.empty()) {
    moduleCachePath = GetClangModulesCacheProperty();
    // Even though it is initialized to the default Clang location at startup a
    // user could have overwritten it with an empty path.
    if (moduleCachePath.empty()) {
      llvm::SmallString<0> path;
      std::error_code ec =
          llvm::sys::fs::createUniqueDirectory("ModuleCache", path);
      if (!ec)
        moduleCachePath = std::string(path);
      else
        moduleCachePath = "/tmp/lldb-ModuleCache";
    }
  }
  LOG_PRINTF(LIBLLDB_LOG_TYPES, "Using Clang module cache path: %s",
             moduleCachePath.c_str());

  // Compute the prebuilt module cache path to use:
  // <resource-dir>/<platform>/prebuilt-modules/<version>
  llvm::Triple triple(GetTriple());
  llvm::Optional<llvm::VersionTuple> sdk_version =
      m_ast_context_ap->LangOpts.SDKVersion;
  if (!sdk_version) {
    auto SDKInfoOrErr = clang::parseDarwinSDKInfo(
        *llvm::vfs::getRealFileSystem(),
        m_ast_context_ap->SearchPathOpts.SDKPath);
    if (SDKInfoOrErr) {
      if (auto SDKInfo = *SDKInfoOrErr)
        sdk_version = swift::getTargetSDKVersion(*SDKInfo, triple);
    } else
      llvm::consumeError(SDKInfoOrErr.takeError());
  }
  std::string prebuiltModuleCachePath =
      swift::CompilerInvocation::computePrebuiltCachePath(
          GetResourceDir(triple), triple, sdk_version);
  if (sdk_version)
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "SDK version: %s",
               sdk_version->getAsString().c_str());
  LOG_PRINTF(LIBLLDB_LOG_TYPES, "Using prebuilt Swift module cache path: %s",
             prebuiltModuleCachePath.c_str());

  // Determine the Swift module loading mode to use.
  const auto &props = ModuleList::GetGlobalModuleListProperties();
  swift::ModuleLoadingMode loading_mode;
  switch (props.GetSwiftModuleLoadingMode()) {
  case eSwiftModuleLoadingModePreferSerialized:
    loading_mode = swift::ModuleLoadingMode::PreferSerialized;
    break;
  case eSwiftModuleLoadingModePreferInterface:
    loading_mode = swift::ModuleLoadingMode::PreferInterface;
    break;
  case eSwiftModuleLoadingModeOnlySerialized:
    loading_mode = swift::ModuleLoadingMode::OnlySerialized;
    break;
  case eSwiftModuleLoadingModeOnlyInterface:
    loading_mode = swift::ModuleLoadingMode::OnlyInterface;
    break;
  }

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
      swift::ModuleInterfaceLoaderOptions(),
      swift::RequireOSSAModules_t(GetSILOptions())));

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
    m_clang_importer = (swift::ClangImporter *)clang_importer_ap.get();
    m_ast_context_ap->addModuleLoader(std::move(clang_importer_ap),
                                      /*isClang=*/true);
  }

  // Set up the required state for the evaluator in the TypeChecker.
  registerIDERequestFunctions(m_ast_context_ap->evaluator);
  registerParseRequestFunctions(m_ast_context_ap->evaluator);
  registerTypeCheckerRequestFunctions(m_ast_context_ap->evaluator);
  registerSILGenRequestFunctions(m_ast_context_ap->evaluator);
  registerSILOptimizerRequestFunctions(m_ast_context_ap->evaluator);
  registerTBDGenRequestFunctions(m_ast_context_ap->evaluator);
  registerIRGenRequestFunctions(m_ast_context_ap->evaluator);
  registerIRGenSILTransforms(*m_ast_context_ap);

  GetASTMap().Insert(m_ast_context_ap.get(), this);

  VALID_OR_RETURN(nullptr);
  return m_ast_context_ap.get();
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
  return m_clang_importer;
}

swift::DWARFImporterDelegate *SwiftASTContext::GetDWARFImporterDelegate() {
  VALID_OR_RETURN(nullptr);

  return m_dwarf_importer_delegate_up.get();
}

const swift::SearchPathOptions *SwiftASTContext::GetSearchPathOptions() const {
  VALID_OR_RETURN(0);

  if (!m_ast_context_ap)
    return nullptr;
  return &m_ast_context_ap->SearchPathOpts;
}

const std::vector<std::string> &SwiftASTContext::GetClangArguments() {
  return GetClangImporterOptions().ExtraArgs;
}

swift::ModuleDecl *
SwiftASTContext::GetCachedModule(const SourceModule &module) {
  VALID_OR_RETURN(nullptr);
  if (!module.path.size())
    return nullptr;

  SwiftModuleMap::const_iterator iter =
      m_swift_module_cache.find(module.path.front().GetStringRef());

  if (iter != m_swift_module_cache.end())
    return iter->second;
  return nullptr;
}

swift::ModuleDecl *
SwiftASTContext::CreateModule(const SourceModule &module, Status &error,
                              swift::ImplicitImportInfo importInfo) {
  VALID_OR_RETURN(nullptr);
  if (!module.path.size()) {
    error.SetErrorStringWithFormat("invalid module name (empty)");
    return nullptr;
  }

  if (swift::ModuleDecl *module_decl = GetCachedModule(module)) {
    error.SetErrorStringWithFormat("module already exists for \"%s\"",
                                   module.path.front().GetCString());
    return nullptr;
  }

  swift::ASTContext *ast = GetASTContext();
  if (!ast) {
    error.SetErrorStringWithFormat("invalid swift AST (nullptr)");
    return nullptr;
  }

  swift::Identifier module_id(
      ast->getIdentifier(module.path.front().GetCString()));
  auto *module_decl = swift::ModuleDecl::create(module_id, *ast, importInfo);
  if (!module_decl) {
    error.SetErrorStringWithFormat("failed to create module for \"%s\"",
                                   module.path.front().GetCString());
    return nullptr;
  }

  m_swift_module_cache.insert(
      {module.path.front().GetStringRef(), module_decl});
  return module_decl;
}

void SwiftASTContext::CacheModule(swift::ModuleDecl *module) {
  VALID_OR_RETURN_VOID();

  if (!module)
    return;
  auto ID = module->getName().get();
  if (!ID || !ID[0])
    return;
  if (m_swift_module_cache.find(ID) != m_swift_module_cache.end())
    return;
  m_swift_module_cache.insert({ID, module});
}

swift::ModuleDecl *SwiftASTContext::GetModule(const SourceModule &module,
                                              Status &error) {
  VALID_OR_RETURN(nullptr);
  if (!module.path.size())
    return nullptr;

  LOG_PRINTF(LIBLLDB_LOG_TYPES, "(\"%s\")",
             module.path.front().AsCString("<no name>"));

  if (module.path.front().IsEmpty()) {
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "empty module name");
    error.SetErrorString("invalid module name (empty)");
    return nullptr;
  }

  if (swift::ModuleDecl *module_decl = GetCachedModule(module))
    return module_decl;

  LLDB_SCOPED_TIMER();
  swift::ASTContext *ast = GetASTContext();
  if (!ast) {
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "(\"%s\") invalid ASTContext",
               module.path.front().GetCString());

    error.SetErrorString("invalid swift::ASTContext");
    return nullptr;
  }

  typedef swift::Located<swift::Identifier> ModuleNameSpec;
  llvm::StringRef module_basename_sref = module.path.front().GetStringRef();
  ModuleNameSpec name_pair(ast->getIdentifier(module_basename_sref),
                           swift::SourceLoc());

  if (HasFatalErrors()) {
    error.SetErrorStringWithFormat("failed to get module \"%s\" from AST "
                                   "context:\nAST context is in a fatal "
                                   "error state",
                                   module.path.front().GetCString());
    return nullptr;
  }

  ClearDiagnostics();
  swift::ModuleDecl *module_decl = ast->getModuleByName(module_basename_sref);
  if (HasErrors()) {
    DiagnosticManager diagnostic_manager;
    PrintDiagnostics(diagnostic_manager);
    std::string diagnostic = diagnostic_manager.GetString();
    error.SetErrorStringWithFormat(
        "failed to get module \"%s\" from AST context:\n%s",
        module.path.front().GetCString(), diagnostic.c_str());

    LOG_PRINTF(LIBLLDB_LOG_TYPES, "(\"%s\") -- %s",
               module.path.front().GetCString(), diagnostic.c_str());
    return nullptr;
  }

  if (!module_decl) {
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "failed with no error");

    error.SetErrorStringWithFormat(
        "failed to get module \"%s\" from AST context",
        module.path.front().GetCString());
    return nullptr;
  }
  LOG_PRINTF(LIBLLDB_LOG_TYPES, "(\"%s\") -- found %s",
             module.path.front().GetCString(),
             module_decl->getName().str().str().c_str());

  m_swift_module_cache[module.path.front().GetStringRef()] = module_decl;
  return module_decl;
}

swift::ModuleDecl *SwiftASTContext::GetModule(const FileSpec &module_spec,
                                              Status &error) {
  VALID_OR_RETURN(nullptr);

  ConstString module_basename(module_spec.GetFileNameStrippingExtension());

  LOG_PRINTF(LIBLLDB_LOG_TYPES, "(\"%s\")", module_spec.GetPath().c_str());

  if (module_basename) {
    SwiftModuleMap::const_iterator iter =
        m_swift_module_cache.find(module_basename.GetCString());

    if (iter != m_swift_module_cache.end())
      return iter->second;

    if (FileSystem::Instance().Exists(module_spec)) {
      swift::ASTContext *ast = GetASTContext();
      if (!GetClangImporter()) {
        LOG_PRINTF(LIBLLDB_LOG_TYPES,
                   "((FileSpec)\"%s\") -- no ClangImporter so giving up",
                   module_spec.GetPath().c_str());
        error.SetErrorStringWithFormat("couldn't get a ClangImporter");
        return nullptr;
      }

      std::string module_directory(module_spec.GetDirectory().GetCString());
      bool add_search_path = true;
      for (auto path : ast->SearchPathOpts.ImportSearchPaths) {
        if (path == module_directory) {
          add_search_path = false;
          break;
        }
      }
      // Add the search path if needed so we can find the module by basename.
      if (add_search_path)
        ast->SearchPathOpts.ImportSearchPaths.push_back(
            std::move(module_directory));

      typedef swift::Located<swift::Identifier> ModuleNameSpec;
      llvm::StringRef module_basename_sref(module_basename.GetCString());
      ModuleNameSpec name_pair(ast->getIdentifier(module_basename_sref),
                               swift::SourceLoc());
      swift::ModuleDecl *module =
          ast->getModule(llvm::ArrayRef<ModuleNameSpec>(name_pair));
      if (module) {
        LOG_PRINTF(LIBLLDB_LOG_TYPES, "((FileSpec)\"%s\") -- found %s",
                   module_spec.GetPath().c_str(),
                   module->getName().str().str().c_str());

        m_swift_module_cache[module_basename.GetCString()] = module;
        return module;
      } else {
        LOG_PRINTF(LIBLLDB_LOG_TYPES,
                   "((FileSpec)\"%s\") -- couldn't get from AST context",
                   module_spec.GetPath().c_str());

        error.SetErrorStringWithFormat(
            "failed to get module \"%s\" from AST context",
            module_basename.GetCString());
      }
    } else {
      LOG_PRINTF(LIBLLDB_LOG_TYPES, "((FileSpec)\"%s\") -- doesn't exist",
                 module_spec.GetPath().c_str());

      error.SetErrorStringWithFormat("module \"%s\" doesn't exist",
                                     module_spec.GetPath().c_str());
    }
  } else {
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "((FileSpec)\"%s\") -- no basename",
               module_spec.GetPath().c_str());

    error.SetErrorStringWithFormat("no module basename in \"%s\"",
                                   module_spec.GetPath().c_str());
  }
  return NULL;
}

swift::ModuleDecl *
SwiftASTContext::FindAndLoadModule(const SourceModule &module, Process &process,
                                   Status &error) {
  VALID_OR_RETURN(nullptr);

  swift::ModuleDecl *swift_module = GetModule(module, error);
  if (!swift_module)
    return nullptr;
  LoadModule(swift_module, process, error);
  return swift_module;
}

swift::ModuleDecl *
SwiftASTContext::FindAndLoadModule(const FileSpec &module_spec,
                                   Process &process, Status &error) {
  VALID_OR_RETURN(nullptr);

  swift::ModuleDecl *swift_module = GetModule(module_spec, error);
  if (!swift_module)
    return nullptr;
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
  return paths;
}

void SwiftASTContext::LoadModule(swift::ModuleDecl *swift_module,
                                 Process &process, Status &error) {
  VALID_OR_RETURN_VOID();
  LLDB_SCOPED_TIMER();

  Status current_error;
  auto addLinkLibrary = [&](swift::LinkLibrary link_lib) {
    Status load_image_error;
    StreamString all_dlopen_errors;
    std::string library_name = link_lib.getName().str();

    if (library_name.empty()) {
      error.SetErrorString("Empty library name passed to addLinkLibrary");
      return;
    }

    SwiftLanguageRuntime *runtime = SwiftLanguageRuntime::Get(&process);
    if (runtime && runtime->IsInLibraryNegativeCache(library_name))
      return;

    swift::LibraryKind library_kind = link_lib.getKind();

    LOG_PRINTF(LIBLLDB_LOG_TYPES, "Loading link library \"%s\" of kind: %d.",
               library_name.c_str(), library_kind);

    switch (library_kind) {
    case swift::LibraryKind::Framework: {

      // First make sure the library isn't already loaded. Since this
      // is a framework, we make sure the file name and the framework
      // name are the same, and that we are contained in
      // FileName.framework with no other intervening frameworks.  We
      // can get more restrictive if this gives false positives.
      ConstString library_cstr(library_name);

      std::string framework_name(library_name);
      framework_name.append(".framework");

      // Lookup the module by file basename and make sure that
      // basename has "<basename>.framework" in the path.
      ModuleSpec module_spec;
      module_spec.GetFileSpec().GetFilename() = library_cstr;
      lldb_private::ModuleList matching_module_list;
      bool module_already_loaded = false;
      process.GetTarget().GetImages().FindModules(module_spec,
                                                  matching_module_list);
      if (!matching_module_list.IsEmpty()) {
        matching_module_list.ForEach(
            [&module_already_loaded, &module_spec,
             &framework_name](const ModuleSP &module_sp) -> bool {
              module_already_loaded = module_spec.GetFileSpec().GetPath().find(
                                          framework_name) != std::string::npos;
              return module_already_loaded ==
                     false; // Keep iterating if we didn't find the right module
            });
      }
      // If we already have this library loaded, don't try and load it again.
      if (module_already_loaded) {
        LOG_PRINTF(LIBLLDB_LOG_TYPES,
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
            LOG_PRINTF(LIBLLDB_LOG_TYPES,
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
        LOG_PRINTF(LIBLLDB_LOG_TYPES, "Found framework at: %s.",
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
           swift_module->getASTContext().SearchPathOpts.FrameworkSearchPaths) {
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
        LOG_PRINTF(LIBLLDB_LOG_TYPES, "Found framework at: %s.",
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

    current_error.SetErrorStringWithFormat(
        "Failed to load linked library %s of module %s - errors:\n%s\n",
        library_name.c_str(), swift_module->getName().str().str().c_str(),
        all_dlopen_errors.GetData());
  };

  for (auto import : swift::namelookup::getAllImports(swift_module)) {
    import.importedModule->collectLinkLibraries(addLinkLibrary);
  }
  error = current_error;
}

bool SwiftASTContext::LoadLibraryUsingPaths(
    Process &process, llvm::StringRef library_name,
    std::vector<std::string> &search_paths, bool check_rpath,
    StreamString &all_dlopen_errors) {
  VALID_OR_RETURN(false);
  LLDB_SCOPED_TIMER();

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
  module_spec.GetFileSpec().GetFilename().SetCString(library_fullname.c_str());

  if (process.GetTarget().GetImages().FindFirstModule(module_spec)) {
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "Skipping module %s as it is already loaded.",
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
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "Found library at: %s.",
               found_library.GetCString());
    return true;
  } else {
    all_dlopen_errors.Printf("Failed to find \"%s\" in paths:\n,",
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
      LOG_PRINTF(LIBLLDB_LOG_TYPES, "Found library using RPATH at: %s.",
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
  VALID_OR_RETURN_VOID();

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
        error.SetErrorString(errors.GetData());
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
  VALID_OR_RETURN_VOID();
  LLDB_SCOPED_TIMER();

  swift::MemoryBufferSerializedModuleLoader *loader =
      GetMemoryBufferModuleLoader();
  if (!loader)
    return;

  SectionList *section_list = module.GetSectionList();
  if (!section_list)
    return;

  auto parse_ast_section = [&](llvm::StringRef section_data_ref, size_t n,
                               size_t total) {
    llvm::SmallVector<std::string, 4> swift_modules;
    if (!swift::parseASTSection(*loader, section_data_ref, swift_modules)) {
      LOG_PRINTF(LIBLLDB_LOG_TYPES,
                 "failed to parse AST section %zu/%zu in image \"%s\".", n,
                 total, module.GetFileSpec().GetFilename().GetCString());
      return;
    }

    // Collect the Swift module names referenced by the AST.
    for (auto module_name : swift_modules) {
      module_names.push_back(module_name);
      LOG_PRINTF(LIBLLDB_LOG_TYPES,
                 "parsed module \"%s\" from Swift AST section %zu/%zu in "
                 "image \"%s\".",
                 module_name.c_str(), n, total,
                 module.GetFileSpec().GetFilename().GetCString());
    }
  };

  if (auto section_sp =
          section_list->FindSectionByType(eSectionTypeSwiftModules, true)) {
    DataExtractor section_data;

    if (section_sp->GetSectionData(section_data)) {
      llvm::StringRef section_data_ref(
          (const char *)section_data.GetDataStart(),
          section_data.GetByteSize());
      parse_ast_section(section_data_ref, 1, 1);
    }
  } else {
    if (m_ast_file_data_map.find(&module) != m_ast_file_data_map.end())
      return;

    // Grab all the AST blobs from the symbol vendor.
    auto ast_file_datas = module.GetASTData(eLanguageTypeSwift);
    LOG_PRINTF(LIBLLDB_LOG_TYPES,
               "(\"%s\") retrieved %zu AST Data blobs from the symbol vendor.",
               GetBriefModuleName(module).c_str(), ast_file_datas.size());

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
      llvm::StringRef section_data_ref(
          (const char *)ast_file_data_sp->GetBytes(),
          ast_file_data_sp->GetByteSize());
      parse_ast_section(section_data_ref, ++i, ast_file_datas.size());
    }
  }
}

void SwiftASTContext::ValidateSectionModules(
    Module &module, const std::vector<std::string> &module_names) {
  VALID_OR_RETURN_VOID();
  LLDB_SCOPED_TIMER();

  Status error;

  for (const std::string &module_name : module_names) {
    SourceModule module_info;
    module_info.path.push_back(ConstString(module_name));
    if (!GetModule(module_info, error))
      module.ReportWarning("unable to load swift module \"%s\" (%s)",
                           module_name.c_str(), error.AsCString());
  }
}

swift::Identifier SwiftASTContext::GetIdentifier(const llvm::StringRef &name) {
  VALID_OR_RETURN(swift::Identifier());

  return GetASTContext()->getIdentifier(name);
}

ConstString SwiftASTContext::GetMangledTypeName(swift::TypeBase *type_base) {
  VALID_OR_RETURN(ConstString());
  LLDB_SCOPED_TIMER();

  auto iter = m_type_to_mangled_name_map.find(type_base),
       end = m_type_to_mangled_name_map.end();
  if (iter != end)
    return ConstString(iter->second);

  swift::Type swift_type(type_base);

  assert(!swift_type->hasArchetype() &&
         "type has not been mapped out of context");
  swift::Mangle::ASTMangler mangler(true);
  std::string s = mangler.mangleTypeForDebugger(swift_type, nullptr);
  if (s.empty())
    return {};

  ConstString mangled_cs{StringRef(s)};
  CacheDemangledType(mangled_cs, type_base);
  return mangled_cs;
}

void SwiftASTContext::CacheDemangledType(ConstString name,
                                         swift::TypeBase *found_type) {
  VALID_OR_RETURN_VOID();

  m_type_to_mangled_name_map.insert({found_type, name.AsCString()});
  m_mangled_name_to_type_map.insert({name.AsCString(), found_type});
}

void SwiftASTContext::CacheDemangledTypeFailure(ConstString name) {
  VALID_OR_RETURN_VOID();

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
  return t.transform([](swift::Type t) -> swift::Type {
    if (auto *silFn = t->getAs<swift::SILFunctionType>()) {
      // FIXME: Verify ExtInfo state is correct, not working by accident.
      swift::FunctionType::ExtInfo info;
      return swift::FunctionType::get({}, t->getASTContext().TheEmptyTupleType,
                                      info);
    }
    return t;
  });
}

CompilerType
SwiftASTContext::GetTypeFromMangledTypename(ConstString mangled_typename) {
  if (llvm::isa<SwiftASTContextForExpressions>(this))
    return GetCompilerType(ReconstructType(mangled_typename));
  return GetCompilerType(mangled_typename);
}

CompilerType SwiftASTContext::GetAsClangType(ConstString mangled_name) {
  LLDB_SCOPED_TIMER();
  if (!swift::Demangle::isObjCSymbol(mangled_name.GetStringRef()))
    return {};

  // When we failed to look up the type because no .swiftmodule is
  // present or it couldn't be read, fall back to presenting objects
  // that look like they might be come from Objective-C (or C) as
  // Clang types. LLDB's Objective-C part is very robust against
  // malformed object pointers, so this isn't very risky.
  Module *module = GetModule();
  if (!module)
    return {};
  auto type_system_or_err =
      module->GetTypeSystemForLanguage(eLanguageTypeObjC);
  if (!type_system_or_err) {
    llvm::consumeError(type_system_or_err.takeError());
    return {};
  }

  auto *clang_ctx =
      llvm::dyn_cast_or_null<TypeSystemClang>(&*type_system_or_err);
  if (!clang_ctx)
    return {};
  DWARFASTParserClang *clang_ast_parser =
      static_cast<DWARFASTParserClang *>(clang_ctx->GetDWARFParser());
  CompilerType clang_type;
  CompilerType imported_type = GetCompilerType(mangled_name);
  if (auto *ts = llvm::dyn_cast_or_null<TypeSystemSwift>(
          imported_type.GetTypeSystem()))
    ts->IsImportedType(imported_type.GetOpaqueQualType(), &clang_type);

  // Import the Clang type into the Clang context.
  if (!clang_type)
    return {};

  if (clang_type.GetTypeSystem() != clang_ctx)
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
SwiftASTContext::ReconstructType(ConstString mangled_typename) {
  Status error;

  auto reconstructed_type = this->ReconstructType(mangled_typename, error);
  if (!error.Success()) {
    this->AddErrorStatusAsGenericDiagnostic(error);
  }
  return reconstructed_type;
}

swift::TypeBase *SwiftASTContext::ReconstructType(ConstString mangled_typename,
                                                  Status &error) {
  VALID_OR_RETURN(nullptr);

  const char *mangled_cstr = mangled_typename.AsCString();
  if (mangled_typename.IsEmpty() ||
      !SwiftLanguageRuntime::IsSwiftMangledName(mangled_typename.GetStringRef())) {
    error.SetErrorStringWithFormat(
        "typename \"%s\" is not a valid Swift mangled name", mangled_cstr);
    return {};
  }

  LOG_PRINTF(LIBLLDB_LOG_TYPES, "(\"%s\")", mangled_cstr);

  swift::ASTContext *ast_ctx = GetASTContext();
  if (!ast_ctx) {
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "(\"%s\") -- null Swift AST Context",
               mangled_cstr);
    error.SetErrorString("null Swift AST Context");
    return {};
  }

  error.Clear();

  // If we were to crash doing this, remember what type caused it.
  llvm::PrettyStackTraceFormat PST("error finding type for %s", mangled_cstr);
  swift::TypeBase *found_type = m_mangled_name_to_type_map.lookup(mangled_cstr);
  if (found_type) {
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "(\"%s\") -- found in the positive cache",
               mangled_cstr);
    assert(&found_type->getASTContext() == ast_ctx);
    return found_type;
  }

  if (m_negative_type_cache.Lookup(mangled_cstr)) {
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "(\"%s\") -- found in the negative cache",
               mangled_cstr);
    return {};
  }

  LOG_PRINTF(LIBLLDB_LOG_TYPES, "(\"%s\") -- not cached, searching",
             mangled_cstr);

  LLDB_SCOPED_TIMERF("%s (not cached)", LLVM_PRETTY_FUNCTION);
  found_type = swift::Demangle::getTypeForMangling(
                   *ast_ctx, mangled_typename.GetStringRef())
                   .getPointer();

  // Objective-C classes sometimes have private subclasses that are invisible to
  // the Swift compiler because they are declared and defined in a .m file. If
  // we can't reconstruct an ObjC type, walk up the type hierarchy until we find
  // something we can import, or until we run out of types
  while (!found_type) {
    CompilerType clang_type = GetAsClangType(mangled_typename);
    if (!clang_type)
      break;

    auto *clang_ctx =
        llvm::dyn_cast_or_null<TypeSystemClang>(clang_type.GetTypeSystem());
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
                     *ast_ctx, super_mangled_typename.GetStringRef())
                     .getPointer();
  }

  if (found_type) {
    swift::TypeBase *ast_type =
        ConvertSILFunctionTypesToASTFunctionTypes(found_type).getPointer();
    // This transformation is lossy: all SILFunction types are mapped
    // to the same AST type. We thus cannot cache the result, since
    // the mapping isn't bijective.
    if (ast_type == found_type)
      CacheDemangledType(mangled_typename, ast_type);
    CompilerType result_type = ToCompilerType(ast_type);
    assert(&ast_type->getASTContext() == ast_ctx);
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "(\"%s\") -- found %s", mangled_cstr,
               result_type.GetTypeName().GetCString());
    return ast_type;
  }

  LOG_PRINTF(LIBLLDB_LOG_TYPES, "(\"%s\") -- not found", mangled_cstr);

  error.SetErrorStringWithFormat("type for typename \"%s\" was not found",
                                 mangled_cstr);
  CacheDemangledTypeFailure(mangled_typename);
  return {};
}

CompilerType SwiftASTContext::GetAnyObjectType() {
  VALID_OR_RETURN(CompilerType());
  swift::ASTContext *ast = GetASTContext();
  return ToCompilerType({ast->getAnyObjectType()});
}

CompilerType SwiftASTContext::GetVoidFunctionType() {
  VALID_OR_RETURN(CompilerType());

  if (!m_void_function_type) {
    swift::ASTContext *ast = GetASTContext();
    swift::Type empty_tuple_type(swift::TupleType::getEmpty(*ast));
    // FIXME: Verify ExtInfo state is correct, not working by accident.
    swift::FunctionType::ExtInfo info;
    m_void_function_type =
        ToCompilerType({swift::FunctionType::get({}, empty_tuple_type, info)});
  }
  return m_void_function_type;
}

static CompilerType ValueDeclToType(swift::ValueDecl *decl,
                                    swift::ASTContext *ast) {
  if (decl) {
    switch (decl->getKind()) {
    case swift::DeclKind::TypeAlias: {
      swift::TypeAliasDecl *alias_decl =
          swift::cast<swift::TypeAliasDecl>(decl);
      swift::Type swift_type = swift::TypeAliasType::get(
          alias_decl, swift::Type(), swift::SubstitutionMap(),
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

static CompilerType DeclToType(swift::Decl *decl, swift::ASTContext *ast) {
  if (swift::ValueDecl *value_decl =
          swift::dyn_cast_or_null<swift::ValueDecl>(decl))
    return ValueDeclToType(value_decl, ast);
  return {};
}

static SwiftASTContext::TypeOrDecl DeclToTypeOrDecl(swift::ASTContext *ast,
                                                    swift::Decl *decl) {
  if (decl) {
    switch (decl->getKind()) {
    case swift::DeclKind::Import:
    case swift::DeclKind::Extension:
    case swift::DeclKind::PatternBinding:
    case swift::DeclKind::TopLevelCode:
    case swift::DeclKind::GenericTypeParam:
    case swift::DeclKind::AssociatedType:
    case swift::DeclKind::EnumElement:
    case swift::DeclKind::EnumCase:
    case swift::DeclKind::IfConfig:
    case swift::DeclKind::Param:
    case swift::DeclKind::Module:
    case swift::DeclKind::MissingMember:
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
          alias_decl, swift::Type(), swift::SubstitutionMap(),
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
    case swift::DeclKind::PoundDiagnostic:
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
  LLDB_SCOPED_TIMER();

  if (!append)
    results.clear();
  size_t size_before = results.size();

  CompilerType container_type = container_type_or_decl.Apply<CompilerType>(
      [](CompilerType type) -> CompilerType { return type; },
      [this](swift::Decl *decl) -> CompilerType {
        return DeclToType(decl, GetASTContext());
      });

  if (false == name.empty() &&
      llvm::dyn_cast_or_null<TypeSystemSwift>(container_type.GetTypeSystem())) {
    swift::Type swift_type = GetSwiftType(container_type);
    if (!swift_type)
      return 0;
    swift::CanType swift_can_type(swift_type->getCanonicalType());
    swift::NominalType *nominal_type =
        swift_can_type->getAs<swift::NominalType>();
    if (!nominal_type)
      return 0;
    swift::NominalTypeDecl *nominal_decl = nominal_type->getDecl();
    llvm::ArrayRef<swift::ValueDecl *> decls = nominal_decl->lookupDirect(
        swift::DeclName(m_ast_context_ap->getIdentifier(name)));
    for (auto decl : decls)
      results.emplace(DeclToTypeOrDecl(GetASTContext(), decl));
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

llvm::Optional<SwiftASTContext::TypeOrDecl>
SwiftASTContext::FindTypeOrDecl(const char *name,
                                swift::ModuleDecl *swift_module) {
  VALID_OR_RETURN(llvm::Optional<SwiftASTContext::TypeOrDecl>());

  TypesOrDecls search_results;

  FindTypesOrDecls(name, swift_module, search_results, false);

  if (search_results.empty())
    return llvm::Optional<SwiftASTContext::TypeOrDecl>();
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
  LLDB_SCOPED_TIMER();

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
        results.emplace(DeclToTypeOrDecl(GetASTContext(), op));

      if (auto *op = swift_module->lookupInfixOperator(identifier).getSingle())
        results.emplace(DeclToTypeOrDecl(GetASTContext(), op));

      if (auto *op = swift_module->lookupPostfixOperator(identifier))
        results.emplace(DeclToTypeOrDecl(GetASTContext(), op));
    }
    if (auto *pg = swift_module->lookupPrecedenceGroup(identifier).getSingle())
      results.emplace(DeclToTypeOrDecl(GetASTContext(), pg));

    for (auto decl : value_decls)
      results.emplace(DeclToTypeOrDecl(GetASTContext(), decl));
  }

  return results.size() - before;
}

size_t SwiftASTContext::FindType(const char *name,
                                 std::set<CompilerType> &results, bool append) {
  VALID_OR_RETURN(0);
  LLDB_SCOPED_TIMER();

  if (!append)
    results.clear();
  auto iter = m_swift_module_cache.begin(), end = m_swift_module_cache.end();

  size_t count = 0;

  std::function<void(swift::ModuleDecl *)> lookup_func =
      [this, name, &results, &count](swift::ModuleDecl *module) -> void {
    CompilerType candidate(this->FindType(name, module));
    if (candidate) {
      ++count;
      results.insert(candidate);
    }
  };

  for (; iter != end; iter++)
    lookup_func(iter->second);

  if (m_scratch_module)
    lookup_func(m_scratch_module);

  return count;
}

CompilerType SwiftASTContext::ImportType(CompilerType &type, Status &error) {
  VALID_OR_RETURN(CompilerType());
  LLDB_SCOPED_TIMER();

  if (m_ast_context_ap.get() == NULL)
    return CompilerType();

  auto *ts = type.GetTypeSystem();
  SwiftASTContext *swift_ast_ctx = llvm::dyn_cast_or_null<SwiftASTContext>(ts);

  if (swift_ast_ctx == nullptr && (!ts || !llvm::isa<TypeSystemSwift>(ts))) {
    error.SetErrorString("Can't import clang type into a Swift ASTContext.");
    return CompilerType();
  } else if (swift_ast_ctx == this) {
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
  if (llvm::isa<TypeSystemSwiftTypeRef>(ts))
    return m_typeref_typesystem.GetTypeFromMangledTypename(mangled_name);
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
  print_options.PrintDocumentationComments =
      print_options.PrintRegularClangComments = print_help_if_available;
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

  if (m_scratch_module == nullptr)
    m_scratch_module = swift::ModuleDecl::create(
        GetASTContext()->getIdentifier("__lldb_scratch_module"),
        *GetASTContext());
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
  LLDB_SCOPED_TIMER();

  llvm::call_once(m_ir_gen_module_once, [this]() {
    // Make sure we have a good ClangImporter.
    GetClangImporter();

    swift::IRGenOptions &ir_gen_opts = GetIRGenOptions();

    std::string error_str;
    llvm::Triple llvm_triple = GetTriple();
    const llvm::Target *llvm_target =
        llvm::TargetRegistry::lookupTarget(llvm_triple.str(), error_str);

    llvm::CodeGenOpt::Level optimization_level = llvm::CodeGenOpt::Level::None;

    // Create a target machine.
    llvm::TargetMachine *target_machine = llvm_target->createTargetMachine(
        llvm_triple.str(),
        "generic", // cpu
        "",        // features
        *getTargetOptions(),
        llvm::Reloc::Static, // TODO verify with Sean, Default went away
        llvm::None, optimization_level);
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
        llvm_module->setTargetTriple(llvm_triple.str());
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
      if (auto swift_type = GetSwiftType(element.element_type)) {
        if (element.element_name.IsEmpty())
          tuple_elems.push_back(swift::TupleTypeElt(swift_type));
        else
          tuple_elems.push_back(swift::TupleTypeElt(
              swift_type, m_ast_context_ap->getIdentifier(
                              element.element_name.GetCString())));
      } else
        return {};
    }
    llvm::ArrayRef<swift::TupleTypeElt> fields(tuple_elems);
    return ToCompilerType(
        {swift::TupleType::get(fields, *GetASTContext()).getPointer()});
  }
}

bool SwiftASTContext::IsTupleType(lldb::opaque_compiler_type_t type) {
  VALID_OR_RETURN(false);

  auto swift_type = GetSwiftType(type);
  return llvm::isa<::swift::TupleType>(swift_type);
}

CompilerType SwiftASTContext::GetErrorType() {
  VALID_OR_RETURN(CompilerType());

  swift::ASTContext *swift_ctx = GetASTContext();
  if (swift_ctx) {
    // Getting the error type requires the Stdlib module be loaded,
    // but doesn't cause it to be loaded.  Do that here.
    swift_ctx->getStdlibModule(true);
    swift::NominalTypeDecl *error_type_decl = GetASTContext()->getErrorDecl();
    if (error_type_decl) {
      auto error_type = error_type_decl->getDeclaredType().getPointer();
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

bool SwiftASTContext::HasErrors() {
  if (m_diagnostic_consumer_ap.get())
    return (
        static_cast<StoringDiagnosticConsumer *>(m_diagnostic_consumer_ap.get())
            ->NumErrors() != 0);
  else
    return false;
}

bool SwiftASTContext::HasFatalErrors(swift::ASTContext *ast_context) {
  return (ast_context && ast_context->Diags.hasFatalErrorOccurred());
}

void SwiftASTContext::ClearDiagnostics() {
  assert(!HasFatalErrors() && "Never clear a fatal diagnostic!");
  if (m_diagnostic_consumer_ap.get())
    static_cast<StoringDiagnosticConsumer *>(m_diagnostic_consumer_ap.get())
        ->Clear();
}

bool SwiftASTContext::SetColorizeDiagnostics(bool b) {
  if (m_diagnostic_consumer_ap.get())
    return static_cast<StoringDiagnosticConsumer *>(
               m_diagnostic_consumer_ap.get())
        ->SetColorize(b);
  return false;
}

void SwiftASTContext::AddErrorStatusAsGenericDiagnostic(Status error) {
  assert(!error.Success() && "status should be in an error state");

  auto diagnostic = std::make_unique<Diagnostic>(
      error.AsCString(), eDiagnosticSeverityError, eDiagnosticOriginLLDB,
      LLDB_INVALID_COMPILER_ID);
  if (m_diagnostic_consumer_ap.get())
    static_cast<StoringDiagnosticConsumer *>(m_diagnostic_consumer_ap.get())
        ->AddDiagnostic(std::move(diagnostic));
}

void SwiftASTContext::PrintDiagnostics(DiagnosticManager &diagnostic_manager,
                                       uint32_t bufferID, uint32_t first_line,
                                       uint32_t last_line) {
  LLDB_SCOPED_TIMER();
  // If this is a fatal error, copy the error into the AST context's
  // fatal error field, and then put it to the stream, otherwise just
  // dump the diagnostics to the stream.

  // N.B. you cannot use VALID_OR_RETURN_VOID here since that exits if
  // you have fatal errors, which are what we are trying to print
  // here.
  if (!m_ast_context_ap.get()) {
    SymbolFile *sym_file = GetSymbolFile();
    if (sym_file) {
      ConstString name =
          sym_file->GetObjectFile()->GetModule()->GetObjectName();
      m_fatal_errors.SetErrorStringWithFormat("Null context for %s.",
                                              name.AsCString());
    } else {
      m_fatal_errors.SetErrorString("Unknown fatal error occurred.");
    }
    return;
  }

  if (m_ast_context_ap->Diags.hasFatalErrorOccurred() &&
      !m_reported_fatal_error) {
    DiagnosticManager fatal_diagnostics;

    if (m_diagnostic_consumer_ap.get())
      static_cast<StoringDiagnosticConsumer *>(m_diagnostic_consumer_ap.get())
          ->PrintDiagnostics(fatal_diagnostics, bufferID, first_line,
                             last_line);
    if (fatal_diagnostics.Diagnostics().size())
      m_fatal_errors.SetErrorString(fatal_diagnostics.GetString().c_str());
    else
      m_fatal_errors.SetErrorString("Unknown fatal error occurred.");

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
    if (m_diagnostic_consumer_ap.get())
      static_cast<StoringDiagnosticConsumer *>(m_diagnostic_consumer_ap.get())
          ->PrintDiagnostics(diagnostic_manager, bufferID, first_line,
                             last_line);
  }
}

void SwiftASTContext::ModulesDidLoad(ModuleList &module_list) {
  ClearModuleDependentCaches();

  // Scan the new modules for Swift contents and try to import it if
  // safe, otherwise poison this context.
  TargetSP target_sp = GetTarget().lock();
  if (!target_sp)
    return;

  bool use_all_compiler_flags = target_sp->GetUseAllCompilerFlags();
  unsigned num_images = module_list.GetSize();
  for (size_t mi = 0; mi != num_images; ++mi) {
    std::vector<std::string> module_search_paths;
    std::vector<std::pair<std::string, bool>> framework_search_paths;
    std::vector<std::string> extra_clang_args;
    lldb::ModuleSP module_sp = module_list.GetModuleAtIndex(mi);
    ProcessModule(module_sp, m_description, use_all_compiler_flags, *target_sp,
                  module_search_paths, framework_search_paths,
                  extra_clang_args);
    // If the use-all-compiler-flags setting is enabled, the expression
    // context is supposed to merge all search paths form all dylibs.
    if (use_all_compiler_flags && !extra_clang_args.empty()) {
      // We cannot reconfigure ClangImporter after its creation.
      // Instead poison the SwiftASTContext so it gets recreated.
      m_fatal_errors.SetErrorStringWithFormat(
          "New Swift image added: %s",
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

void SwiftASTContext::LogConfiguration() {
  // It makes no sense to call VALID_OR_RETURN here. We specifically
  // want the logs in the error case!
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES));
  if (!log)
    return;

  LOG_PRINTF(LIBLLDB_LOG_TYPES,
             "(SwiftASTContext*)%p:", static_cast<void *>(this));

  if (!m_ast_context_ap) {
    log->Printf("  (no AST context)");
    return;
  }

  log->Printf("  Architecture                 : %s",
              m_ast_context_ap->LangOpts.Target.getTriple().c_str());
  log->Printf("  SDK path                     : %s",
              m_ast_context_ap->SearchPathOpts.SDKPath.c_str());
  log->Printf("  Runtime resource path        : %s",
              m_ast_context_ap->SearchPathOpts.RuntimeResourcePath.c_str());
  log->Printf("  Runtime library paths        : (%llu items)",
              (unsigned long long)
                  m_ast_context_ap->SearchPathOpts.RuntimeLibraryPaths.size());

  for (const auto &runtime_library_path :
       m_ast_context_ap->SearchPathOpts.RuntimeLibraryPaths) {
    log->Printf("    %s", runtime_library_path.c_str());
  }

  log->Printf("  Runtime library import paths : (%llu items)",
              (unsigned long long)m_ast_context_ap->SearchPathOpts
                  .RuntimeLibraryImportPaths.size());

  for (const auto &runtime_import_path :
       m_ast_context_ap->SearchPathOpts.RuntimeLibraryImportPaths) {
    log->Printf("    %s", runtime_import_path.c_str());
  }

  log->Printf("  Framework search paths       : (%llu items)",
              (unsigned long long)
                  m_ast_context_ap->SearchPathOpts.FrameworkSearchPaths.size());
  for (const auto &framework_search_path :
       m_ast_context_ap->SearchPathOpts.FrameworkSearchPaths) {
    log->Printf("    %s", framework_search_path.Path.c_str());
  }

  log->Printf("  Import search paths          : (%llu items)",
              (unsigned long long)
                  m_ast_context_ap->SearchPathOpts.ImportSearchPaths.size());
  for (std::string &import_search_path :
       m_ast_context_ap->SearchPathOpts.ImportSearchPaths) {
    log->Printf("    %s", import_search_path.c_str());
  }

  swift::ClangImporterOptions &clang_importer_options =
      GetClangImporterOptions();

  log->Printf("  Extra clang arguments        : (%llu items)",
              (unsigned long long)clang_importer_options.ExtraArgs.size());
  for (std::string &extra_arg : clang_importer_options.ExtraArgs) {
    log->Printf("    %s", extra_arg.c_str());
  }
}

bool SwiftASTContext::HasTarget() const {
  lldb::TargetWP empty_wp;

  // If either call to "std::weak_ptr::owner_before(...) value returns
  // true, this indicates that m_section_wp once contained (possibly
  // still does) a reference to a valid shared pointer. This helps us
  // know if we had a valid reference to a target which is now invalid
  // because the target was deleted.
  return empty_wp.owner_before(m_target_wp) ||
         m_target_wp.owner_before(empty_wp);
}

bool SwiftASTContext::CheckProcessChanged() {
  if (HasTarget()) {
    TargetSP target_sp(m_target_wp.lock());
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
  return !swift_type || &swift_type->getASTContext() == GetASTContext();
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
      return {this, param.getParameterType().getPointer()};
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
  LLDB_SCOPED_TIMER();

  auto can_type = GetCanonicalSwiftType(type);
  if (!can_type)
    return false;

  if (can_type->getClassOrBoundGenericClass() ||
      can_type->isAnyExistentialType())
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
  if (swift::Type swift_type = ::GetSwiftType(compiler_type))
    return swift_type->hasTypeParameter(); // is<swift::ArchetypeType>();
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
  if (auto bound = runtime->BindGenericTypeParameters(*frame, type))
    return bound;
  return type;
}

bool SwiftASTContext::IsErrorType(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, false);
  ProtocolInfo protocol_info;
  if (GetProtocolTypeInfo({this, type}, protocol_info))
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
    return {this, type};

  auto ref_type = swift_type->getReferenceStorageReferent();
  return ToCompilerType({ref_type});
}

bool SwiftASTContext::IsFullyRealized(const CompilerType &compiler_type) {
  if (swift::CanType swift_can_type = ::GetCanonicalSwiftType(compiler_type)) {
    if (swift::isa<swift::MetatypeType>(swift_can_type))
      return true;
    return !swift_can_type->hasArchetype() &&
           !swift_can_type->hasTypeParameter();
  }

  return false;
}

bool SwiftASTContext::GetProtocolTypeInfo(const CompilerType &type,
                                          ProtocolInfo &protocol_info) {
  LLDB_SCOPED_TIMER();
  if (swift::CanType swift_can_type = ::GetCanonicalSwiftType(type)) {
    if (!swift_can_type.isExistentialType())
      return false;

    swift::ExistentialLayout layout = swift_can_type.getExistentialLayout();
    protocol_info.m_is_class_only = layout.requiresClass();
    protocol_info.m_num_protocols = layout.getProtocols().size();
    protocol_info.m_is_objc = layout.isObjC();
    protocol_info.m_is_anyobject = layout.isAnyObject();
    protocol_info.m_is_errortype = layout.isErrorExistential();

    if (auto superclass = layout.explicitSuperclass) {
      protocol_info.m_superclass = ToCompilerType({superclass.getPointer()});
    }

    unsigned num_witness_tables = 0;
    for (auto protoTy : layout.getProtocols()) {
      if (!protoTy->getDecl()->isObjC())
        num_witness_tables++;
    }

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

TypeSystemSwift::TypeAllocationStrategy
SwiftASTContext::GetAllocationStrategy(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, TypeAllocationStrategy::eUnknown);

  swift::Type swift_type = GetSwiftType(type);
  if (!swift_type)
    return TypeAllocationStrategy::eUnknown;
  const swift::irgen::TypeInfo *type_info =
      GetSwiftTypeInfo(swift_type.getPointer());
  if (!type_info)
    return TypeAllocationStrategy::eUnknown;
  switch (type_info->getFixedPacking(GetIRGenModule())) {
  case swift::irgen::FixedPacking::OffsetZero:
    return TypeAllocationStrategy::eInline;
  case swift::irgen::FixedPacking::Allocate:
    return TypeAllocationStrategy::ePointer;
  case swift::irgen::FixedPacking::Dynamic:
    return TypeAllocationStrategy::eDynamic;
  }
  return TypeAllocationStrategy::eUnknown;
}

CompilerType
SwiftASTContext::GetTypeRefType(lldb::opaque_compiler_type_t type) {
  return m_typeref_typesystem.GetTypeFromMangledTypename(
      GetMangledTypeName(type));
}

//----------------------------------------------------------------------
// Type Completion
//----------------------------------------------------------------------

ConstString SwiftASTContext::GetTypeName(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(
      type, ConstString("<invalid Swift context or opaque type>"));
  LLDB_SCOPED_TIMER();
  std::string type_name;
  swift::Type swift_type(GetSwiftType(type));

  swift::Type normalized_type =
      swift_type.transform([](swift::Type type) -> swift::Type {
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
                  const SymbolContext *sc) {
  LLDB_SCOPED_TIMER();
  llvm::DenseMap<swift::CanType, swift::Identifier> dict;

  assert(&swift_type->getASTContext() == &ast_ctx);
  if (!sc)
    return dict;

  llvm::DenseMap<std::pair<uint64_t, uint64_t>, StringRef> names;
  SwiftLanguageRuntime::GetGenericParameterNamesForFunction(*sc, names);
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
  LLDB_SCOPED_TIMER();
  std::string type_name(GetTypeName(type).AsCString(""));
  if (type) {
    swift::Type swift_type(GetSwiftType(type));
    swift::PrintOptions print_options;
    print_options.FullyQualifiedTypes = false;
    print_options.SynthesizeSugarOnTypes = true;
    print_options.FullyQualifiedTypesIfAmbiguous = true;
    auto dict = GetArchetypeNames(swift_type, *GetASTContext(), sc);
    print_options.AlternativeTypeNames = &dict;
    type_name = swift_type.getString(print_options);
  }
  return ConstString(type_name);
}

uint32_t
SwiftASTContext::GetTypeInfo(opaque_compiler_type_t type,
                             CompilerType *pointee_or_element_clang_type) {
  VALID_OR_RETURN_CHECK_TYPE(type, 0);
  LLDB_SCOPED_TIMER();

  if (pointee_or_element_clang_type)
    pointee_or_element_clang_type->Clear();

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  const swift::TypeKind type_kind = swift_can_type->getKind();
  uint32_t swift_flags = eTypeIsSwift;
  switch (type_kind) {
  case swift::TypeKind::BuiltinDefaultActorStorage:
  case swift::TypeKind::BuiltinExecutor:
  case swift::TypeKind::BuiltinJob:
  case swift::TypeKind::BuiltinRawUnsafeContinuation:
  case swift::TypeKind::Error:
  case swift::TypeKind::Module:
  case swift::TypeKind::NestedArchetype:
  case swift::TypeKind::OpenedArchetype:
  case swift::TypeKind::OpaqueTypeArchetype:
  case swift::TypeKind::InOut:
  case swift::TypeKind::VariadicSequence:
  case swift::TypeKind::Placeholder:
  case swift::TypeKind::PrimaryArchetype:
  case swift::TypeKind::SILBlockStorage:
  case swift::TypeKind::SILBox:
  case swift::TypeKind::SILFunction:
  case swift::TypeKind::SILToken:
  case swift::TypeKind::TypeVariable:
  case swift::TypeKind::Unresolved:
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "Unexpected type: %s",
               swift_can_type.getString().c_str());
    assert(false && "Internal compiler type");
    break;
  case swift::TypeKind::UnboundGeneric:
    swift_flags |= eTypeIsGeneric;
    break;

  case swift::TypeKind::GenericFunction:
    swift_flags |= eTypeIsGeneric;
    LLVM_FALLTHROUGH;
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
    swift_flags |= eTypeIsGeneric | eTypeIsBound;
    LLVM_FALLTHROUGH;
  case swift::TypeKind::Enum: {
    SwiftEnumDescriptor *cached_enum_info = GetCachedEnumInfo(type);
    if (cached_enum_info)
      swift_flags |= eTypeHasValue | eTypeIsEnumeration | eTypeHasChildren;
    else
      swift_flags |= eTypeIsEnumeration;
  } break;

  case swift::TypeKind::BoundGenericStruct:
    swift_flags |= eTypeIsGeneric | eTypeIsBound;
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
    swift_flags |= eTypeIsGeneric | eTypeIsBound;
    LLVM_FALLTHROUGH;
  case swift::TypeKind::Class:
    swift_flags |= eTypeHasChildren | eTypeIsClass | eTypeHasValue |
                   eTypeInstanceIsPointer;
    break;

  case swift::TypeKind::Protocol:
  case swift::TypeKind::ProtocolComposition:
    swift_flags |= eTypeHasChildren | eTypeIsStructUnion | eTypeIsProtocol;
    break;
  case swift::TypeKind::ExistentialMetatype:
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
    swift_flags |= eTypeIsGeneric | eTypeIsBound | eTypeHasValue;
    break;

  case swift::TypeKind::Optional:
  case swift::TypeKind::TypeAlias:
  case swift::TypeKind::Paren:
  case swift::TypeKind::Dictionary:
  case swift::TypeKind::ArraySlice:
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
  case swift::TypeKind::BuiltinExecutor:
  case swift::TypeKind::BuiltinJob:
  case swift::TypeKind::Placeholder:
  case swift::TypeKind::SILBlockStorage:
  case swift::TypeKind::SILBox:
  case swift::TypeKind::SILToken:
  case swift::TypeKind::Unresolved:
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
  case swift::TypeKind::Metatype:
  case swift::TypeKind::Module:
  case swift::TypeKind::PrimaryArchetype:
  case swift::TypeKind::NestedArchetype:
  case swift::TypeKind::OpaqueTypeArchetype:
  case swift::TypeKind::OpenedArchetype:
  case swift::TypeKind::UnboundGeneric:
  case swift::TypeKind::TypeVariable:
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
  case swift::TypeKind::Paren:
  case swift::TypeKind::Dictionary:
  case swift::TypeKind::ArraySlice:
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
        assert(GetASTContext() == &args[0].getPointer()->getASTContext());
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

CompilerType SwiftASTContext::GetInstanceType(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, CompilerType());

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  assert((&swift_can_type->getASTContext() == GetASTContext()) &&
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
  LLDB_SCOPED_TIMER();

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

CompilerType
SwiftASTContext::GetLValueReferenceType(opaque_compiler_type_t type) {
  return {};
}

CompilerType
SwiftASTContext::GetRValueReferenceType(opaque_compiler_type_t type) {
  return {};
}

CompilerType SwiftASTContext::GetNonReferenceType(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, CompilerType());

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));

  swift::LValueType *lvalue = swift_can_type->getAs<swift::LValueType>();
  if (lvalue)
    return ToCompilerType({lvalue->getObjectType().getPointer()});
  return {};
}

CompilerType SwiftASTContext::GetPointeeType(opaque_compiler_type_t type) {
  return {};
}

CompilerType SwiftASTContext::GetPointerType(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, CompilerType());

  auto swift_type = GetSwiftType({this, type});
  auto pointer_type =
      swift_type->wrapInPointer(swift::PointerTypeKind::PTK_UnsafePointer);
  if (pointer_type)
    return ToCompilerType(pointer_type);

  return {};
}

CompilerType SwiftASTContext::GetTypedefedType(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, CompilerType());

  swift::Type swift_type(GetSwiftType({this, type}));
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

llvm::Optional<uint64_t>
SwiftASTContext::GetBitSize(opaque_compiler_type_t type,
                            ExecutionContextScope *exe_scope) {
  VALID_OR_RETURN_CHECK_TYPE(type, llvm::None);
  LLDB_SCOPED_TIMER();

  // If the type has type parameters, bind them first.
  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  if (swift_can_type->hasTypeParameter()) {
    if (!exe_scope)
      return {};
    ExecutionContext exe_ctx;
    exe_scope->CalculateExecutionContext(exe_ctx);
    auto swift_scratch_ctx_lock = SwiftASTContextLock(&exe_ctx);
    CompilerType bound_type = BindGenericTypeParameters({this, type}, exe_scope);

    // Check that the type has been bound successfully -- and if not,
    // log the event and bail out to avoid an infinite loop.
    swift::CanType swift_bound_type(::GetCanonicalSwiftType(bound_type));
    if (swift_bound_type->hasTypeParameter()) {
      LOG_PRINTF(LIBLLDB_LOG_TYPES, "GetBitSize: Can't bind type: %s",
                 bound_type.GetTypeName().AsCString());
      return {};
    }

    // Note thay the bound type may be in a different AST context.
    return bound_type.GetBitSize(exe_scope);
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
    return {};
  if (auto *runtime = SwiftLanguageRuntime::Get(exe_scope->CalculateProcess()))
    return runtime->GetBitSize({this, type}, exe_scope);
  return {};
}

llvm::Optional<uint64_t>
SwiftASTContext::GetByteStride(opaque_compiler_type_t type,
                               ExecutionContextScope *exe_scope) {
  VALID_OR_RETURN_CHECK_TYPE(type, llvm::None);
  LLDB_SCOPED_TIMER();

  // If the type has type parameters, bind them first.
  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  if (swift_can_type->hasTypeParameter()) {
    if (!exe_scope)
      return {};
    ExecutionContext exe_ctx;
    exe_scope->CalculateExecutionContext(exe_ctx);
    auto swift_scratch_ctx_lock = SwiftASTContextLock(&exe_ctx);
    CompilerType bound_type = BindGenericTypeParameters({this, type}, exe_scope);
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
    return runtime->GetByteStride({this, type});
  return {};
}

llvm::Optional<size_t>
SwiftASTContext::GetTypeBitAlign(opaque_compiler_type_t type,
                                 ExecutionContextScope *exe_scope) {
  VALID_OR_RETURN_CHECK_TYPE(type, llvm::None);
  LLDB_SCOPED_TIMER();

  // If the type has type parameters, bind them first.
  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  if (swift_can_type->hasTypeParameter()) {
    if (!exe_scope)
      return {};
    ExecutionContext exe_ctx;
    exe_scope->CalculateExecutionContext(exe_ctx);
    auto swift_scratch_ctx_lock = SwiftASTContextLock(&exe_ctx);
    CompilerType bound_type = BindGenericTypeParameters({this, type}, exe_scope);
    if (bound_type.GetOpaqueQualType() == type)
      return {};
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
    return runtime->GetBitAlignment({this, type}, exe_scope);
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
  case swift::TypeKind::BuiltinExecutor:
  case swift::TypeKind::BuiltinJob:
  case swift::TypeKind::BuiltinRawUnsafeContinuation:
  case swift::TypeKind::Error:
  case swift::TypeKind::Module:
  case swift::TypeKind::InOut:
  case swift::TypeKind::VariadicSequence:
  case swift::TypeKind::Placeholder:
  case swift::TypeKind::SILBlockStorage:
  case swift::TypeKind::SILBox:
  case swift::TypeKind::SILFunction:
  case swift::TypeKind::SILToken:
  case swift::TypeKind::TypeVariable:
  case swift::TypeKind::Unresolved:
    break;
  case swift::TypeKind::BuiltinInteger:
  case swift::TypeKind::BuiltinIntegerLiteral:
    return lldb::eEncodingSint; // TODO: detect if an integer is unsigned
  case swift::TypeKind::BuiltinFloat:
    return lldb::eEncodingIEEE754; // TODO: detect if an integer is unsigned

  case swift::TypeKind::PrimaryArchetype:
  case swift::TypeKind::OpenedArchetype:
  case swift::TypeKind::OpaqueTypeArchetype:
  case swift::TypeKind::NestedArchetype:
  case swift::TypeKind::BuiltinRawPointer:
  case swift::TypeKind::BuiltinNativeObject:
  case swift::TypeKind::BuiltinUnsafeValueBuffer:
  case swift::TypeKind::BuiltinBridgeObject:
  case swift::TypeKind::Class: // Classes are pointers in swift...
  case swift::TypeKind::BoundGenericClass:
  case swift::TypeKind::GenericTypeParam:
  case swift::TypeKind::DependentMember:
    return lldb::eEncodingUint;

  case swift::TypeKind::BuiltinVector:
    break;
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
    break;
  case swift::TypeKind::LValue:
    return lldb::eEncodingUint;
  case swift::TypeKind::UnboundGeneric:
  case swift::TypeKind::BoundGenericStruct:
  case swift::TypeKind::DynamicSelf:
    break;

  case swift::TypeKind::Optional:
  case swift::TypeKind::TypeAlias:
  case swift::TypeKind::Paren:
  case swift::TypeKind::Dictionary:
  case swift::TypeKind::ArraySlice:
    assert(false && "Not a canonical type");
    break;
  }
  count = 0;
  return lldb::eEncodingInvalid;
}

lldb::Format SwiftASTContext::GetFormat(opaque_compiler_type_t type) {
  VALID_OR_RETURN_CHECK_TYPE(type, lldb::eFormatInvalid);

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));

  const swift::TypeKind type_kind = swift_can_type->getKind();
  switch (type_kind) {
  case swift::TypeKind::BuiltinDefaultActorStorage:
  case swift::TypeKind::BuiltinExecutor:
  case swift::TypeKind::BuiltinJob:
  case swift::TypeKind::BuiltinRawUnsafeContinuation:
  case swift::TypeKind::Error:
  case swift::TypeKind::Module:
  case swift::TypeKind::InOut:
  case swift::TypeKind::VariadicSequence:
  case swift::TypeKind::Placeholder:
  case swift::TypeKind::SILBlockStorage:
  case swift::TypeKind::SILBox:
  case swift::TypeKind::SILFunction:
  case swift::TypeKind::SILToken:
  case swift::TypeKind::TypeVariable:
  case swift::TypeKind::Unresolved:
    break;
  case swift::TypeKind::BuiltinIntegerLiteral:
  case swift::TypeKind::BuiltinInteger:
    return eFormatDecimal; // TODO: detect if an integer is unsigned
  case swift::TypeKind::BuiltinFloat:
    return eFormatFloat; // TODO: detect if an integer is unsigned

  case swift::TypeKind::BuiltinRawPointer:
  case swift::TypeKind::BuiltinNativeObject:
  case swift::TypeKind::BuiltinUnsafeValueBuffer:
  case swift::TypeKind::BuiltinBridgeObject:
  case swift::TypeKind::PrimaryArchetype:
  case swift::TypeKind::OpenedArchetype:
  case swift::TypeKind::OpaqueTypeArchetype:
  case swift::TypeKind::NestedArchetype:
  case swift::TypeKind::GenericTypeParam:
  case swift::TypeKind::DependentMember:
    return eFormatAddressInfo;

  // Classes are always pointers in swift.
  case swift::TypeKind::Class:
  case swift::TypeKind::BoundGenericClass:
    return eFormatHex;

  case swift::TypeKind::BuiltinVector:
    break;
  case swift::TypeKind::Tuple:
    break;
  case swift::TypeKind::UnmanagedStorage:
  case swift::TypeKind::UnownedStorage:
  case swift::TypeKind::WeakStorage:
    return ToCompilerType(swift_can_type->getReferenceStorageReferent())
        .GetFormat();
    break;

  case swift::TypeKind::Enum:
  case swift::TypeKind::BoundGenericEnum:
    return eFormatUnsigned;

  case swift::TypeKind::GenericFunction:
  case swift::TypeKind::Function:
    return lldb::eFormatAddressInfo;

  case swift::TypeKind::Struct:
  case swift::TypeKind::Protocol:
  case swift::TypeKind::Metatype:
  case swift::TypeKind::ProtocolComposition:
    break;
  case swift::TypeKind::LValue:
    return lldb::eFormatHex;
  case swift::TypeKind::UnboundGeneric:
  case swift::TypeKind::BoundGenericStruct:
  case swift::TypeKind::ExistentialMetatype:
  case swift::TypeKind::DynamicSelf:
    break;

  case swift::TypeKind::Optional:
  case swift::TypeKind::TypeAlias:
  case swift::TypeKind::Paren:
  case swift::TypeKind::Dictionary:
  case swift::TypeKind::ArraySlice:
    assert(false && "Not a canonical type");
    break;
  }
  // We don't know hot to display this type.
  return lldb::eFormatBytes;
}

uint32_t SwiftASTContext::GetNumChildren(opaque_compiler_type_t type,
                                         bool omit_empty_base_classes,
                                         const ExecutionContext *exe_ctx) {
  VALID_OR_RETURN_CHECK_TYPE(type, 0);
  LLDB_SCOPED_TIMER();

  uint32_t num_children = 0;

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));

  const swift::TypeKind type_kind = swift_can_type->getKind();
  switch (type_kind) {
  case swift::TypeKind::BuiltinDefaultActorStorage:
  case swift::TypeKind::BuiltinExecutor:
  case swift::TypeKind::BuiltinJob:
  case swift::TypeKind::BuiltinRawUnsafeContinuation:
  case swift::TypeKind::Error:
  case swift::TypeKind::Module:
  case swift::TypeKind::InOut:
  case swift::TypeKind::VariadicSequence:
  case swift::TypeKind::Placeholder:
  case swift::TypeKind::SILBlockStorage:
  case swift::TypeKind::SILBox:
  case swift::TypeKind::SILFunction:
  case swift::TypeKind::SILToken:
  case swift::TypeKind::TypeVariable:
  case swift::TypeKind::Unresolved:
    break;
  case swift::TypeKind::BuiltinInteger:
  case swift::TypeKind::BuiltinIntegerLiteral:
  case swift::TypeKind::BuiltinFloat:
  case swift::TypeKind::BuiltinRawPointer:
  case swift::TypeKind::BuiltinNativeObject:
  case swift::TypeKind::BuiltinUnsafeValueBuffer:
  case swift::TypeKind::BuiltinBridgeObject:
  case swift::TypeKind::BuiltinVector:
  case swift::TypeKind::Function:
  case swift::TypeKind::GenericFunction:
  case swift::TypeKind::DynamicSelf:
    break;
  case swift::TypeKind::UnmanagedStorage:
  case swift::TypeKind::UnownedStorage:
  case swift::TypeKind::WeakStorage:
    return ToCompilerType(swift_can_type->getReferenceStorageReferent())
        .GetNumChildren(omit_empty_base_classes, exe_ctx);
  case swift::TypeKind::GenericTypeParam:
  case swift::TypeKind::DependentMember:
    break;

  case swift::TypeKind::Enum:
  case swift::TypeKind::BoundGenericEnum: {
    SwiftEnumDescriptor *cached_enum_info = GetCachedEnumInfo(type);
    if (cached_enum_info)
      return cached_enum_info->GetNumElementsWithPayload();
  } break;

  case swift::TypeKind::Tuple:
  case swift::TypeKind::Struct:
  case swift::TypeKind::BoundGenericStruct:
    return GetNumFields(type);

  case swift::TypeKind::Class:
  case swift::TypeKind::BoundGenericClass: {
    auto class_decl = swift_can_type->getClassOrBoundGenericClass();
    return (class_decl->hasSuperclass() ? 1 : 0) + GetNumFields(type);
  }

  case swift::TypeKind::Protocol:
  case swift::TypeKind::ProtocolComposition: {
    ProtocolInfo protocol_info;
    if (!GetProtocolTypeInfo(ToCompilerType(GetSwiftType(type)), protocol_info))
      break;

    return protocol_info.m_num_storage_words;
  }

  case swift::TypeKind::ExistentialMetatype:
  case swift::TypeKind::Metatype:
  case swift::TypeKind::PrimaryArchetype:
  case swift::TypeKind::OpenedArchetype:
  case swift::TypeKind::OpaqueTypeArchetype:
  case swift::TypeKind::NestedArchetype:
    return 0;

  case swift::TypeKind::LValue: {
    swift::LValueType *lvalue_type =
        swift_can_type->castTo<swift::LValueType>();
    swift::TypeBase *deref_type = lvalue_type->getObjectType().getPointer();

    uint32_t num_pointee_children =
        ToCompilerType(deref_type)
            .GetNumChildren(omit_empty_base_classes, exe_ctx);
    // If this type points to a simple type (or to a class), then it
    // has 1 child.
    if (num_pointee_children == 0 || deref_type->getClassOrBoundGenericClass())
      num_children = 1;
    else
      num_children = num_pointee_children;
  } break;

  case swift::TypeKind::UnboundGeneric:
    break;

  case swift::TypeKind::Optional:
  case swift::TypeKind::TypeAlias:
  case swift::TypeKind::Paren:
  case swift::TypeKind::Dictionary:
  case swift::TypeKind::ArraySlice:
    assert(false && "Not a canonical type");
    break;
  }

  return num_children;
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
  case swift::TypeKind::BuiltinDefaultActorStorage:
  case swift::TypeKind::BuiltinExecutor:
  case swift::TypeKind::BuiltinJob:
  case swift::TypeKind::BuiltinRawUnsafeContinuation:
  case swift::TypeKind::Error:
  case swift::TypeKind::Module:
  case swift::TypeKind::InOut:
  case swift::TypeKind::VariadicSequence:
  case swift::TypeKind::Placeholder:
  case swift::TypeKind::SILBlockStorage:
  case swift::TypeKind::SILBox:
  case swift::TypeKind::SILFunction:
  case swift::TypeKind::SILToken:
  case swift::TypeKind::TypeVariable:
  case swift::TypeKind::Unresolved:
    break;
  case swift::TypeKind::BuiltinInteger:
  case swift::TypeKind::BuiltinIntegerLiteral:
  case swift::TypeKind::BuiltinFloat:
  case swift::TypeKind::BuiltinRawPointer:
  case swift::TypeKind::BuiltinNativeObject:
  case swift::TypeKind::BuiltinUnsafeValueBuffer:
  case swift::TypeKind::BuiltinBridgeObject:
  case swift::TypeKind::BuiltinVector:
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
  case swift::TypeKind::BoundGenericEnum: {
    SwiftEnumDescriptor *cached_enum_info = GetCachedEnumInfo(type);
    if (cached_enum_info)
      return cached_enum_info->GetNumElementsWithPayload();
  } break;

  case swift::TypeKind::Tuple:
    return cast<swift::TupleType>(swift_can_type)->getNumElements();

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
    return GetNumChildren(type, /*omit_empty_base_classes=*/false, nullptr);

  case swift::TypeKind::ExistentialMetatype:
  case swift::TypeKind::Metatype:
    return 0;

  case swift::TypeKind::PrimaryArchetype:
  case swift::TypeKind::OpenedArchetype:
  case swift::TypeKind::OpaqueTypeArchetype:
  case swift::TypeKind::NestedArchetype:
  case swift::TypeKind::Function:
  case swift::TypeKind::GenericFunction:
  case swift::TypeKind::LValue:
  case swift::TypeKind::UnboundGeneric:
  case swift::TypeKind::DynamicSelf:
    break;

  case swift::TypeKind::Optional:
  case swift::TypeKind::TypeAlias:
  case swift::TypeKind::Paren:
  case swift::TypeKind::Dictionary:
  case swift::TypeKind::ArraySlice:
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
static std::pair<CompilerType, std::string>
GetExistentialTypeChild(swift::ASTContext *swift_ast_ctx, CompilerType type,
                        const SwiftASTContext::ProtocolInfo &protocol_info,
                        unsigned idx) {
  assert(idx < protocol_info.m_num_storage_words &&
         "caller is responsible for validating index");

  // A payload word for a non-class, non-error existential.
  if (idx < protocol_info.m_num_payload_words) {
    std::string name;
    llvm::raw_string_ostream(name) << "payload_data_" << idx;

    auto raw_pointer = swift_ast_ctx->TheRawPointerType;
    return {ToCompilerType(raw_pointer.getPointer()), std::move(name)};
  }

  // The instance for a class-bound existential.
  if (idx == 0 && protocol_info.m_is_class_only) {
    CompilerType class_type;
    // FIXME: Remove this comment once the TypeSystemSwiftTyperef
    // transition is complete.
    //
    // There is not enough data available to support this in
    // TypeSystemSwiftTypeRef, but there is also notuser-visible
    // feature affected by this apart from the --raw-types output, so
    // this was removed to match TypeSystemSwiftTyperef:
    /* if (protocol_info.m_superclass) {
      class_type = protocol_info.m_superclass;
      } else */ {
      auto raw_pointer = swift_ast_ctx->TheRawPointerType;
      class_type = ToCompilerType(raw_pointer.getPointer());
    }

    return {class_type, "object"};
  }

  // The instance for an error existential.
  if (idx == 0 && protocol_info.m_is_errortype) {
    auto raw_pointer = swift_ast_ctx->TheRawPointerType;
    return {ToCompilerType(raw_pointer.getPointer()), "error"};
  }

  // The metatype for a non-class, non-error existential.
  if (idx && idx == protocol_info.m_num_payload_words) {
    // The metatype for a non-class, non-error existential.
    auto any_metatype =
        swift::ExistentialMetatypeType::get(swift_ast_ctx->TheAnyType);
    return {ToCompilerType(any_metatype), "metadata"};
  }

  // A witness table. Figure out which protocol it corresponds to.
  unsigned witness_table_idx = idx - protocol_info.m_num_payload_words - 1;
  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  swift::ExistentialLayout layout = swift_can_type.getExistentialLayout();

  std::string name;
  for (auto protoType : layout.getProtocols()) {
    auto proto = protoType->getDecl();
    if (proto->isObjC())
      continue;

    if (witness_table_idx == 0) {
      name = "wtable";
      break;
    }
    --witness_table_idx;
  }

  auto raw_pointer = swift_ast_ctx->TheRawPointerType;
  return {ToCompilerType(raw_pointer.getPointer()), std::move(name)};
}

CompilerType SwiftASTContext::GetFieldAtIndex(opaque_compiler_type_t type,
                                              size_t idx, std::string &name,
                                              uint64_t *bit_offset_ptr,
                                              uint32_t *bitfield_bit_size_ptr,
                                              bool *is_bitfield_ptr) {
  VALID_OR_RETURN_CHECK_TYPE(type, CompilerType());
  LLDB_SCOPED_TIMER();

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));

  const swift::TypeKind type_kind = swift_can_type->getKind();
  switch (type_kind) {
  case swift::TypeKind::BuiltinDefaultActorStorage:
  case swift::TypeKind::BuiltinExecutor:
  case swift::TypeKind::BuiltinJob:
  case swift::TypeKind::BuiltinRawUnsafeContinuation:
  case swift::TypeKind::Error:
  case swift::TypeKind::Module:
  case swift::TypeKind::InOut:
  case swift::TypeKind::VariadicSequence:
  case swift::TypeKind::Placeholder:
  case swift::TypeKind::SILBlockStorage:
  case swift::TypeKind::SILBox:
  case swift::TypeKind::SILFunction:
  case swift::TypeKind::SILToken:
  case swift::TypeKind::TypeVariable:
  case swift::TypeKind::Unresolved:
    break;
  case swift::TypeKind::BuiltinInteger:
  case swift::TypeKind::BuiltinIntegerLiteral:
  case swift::TypeKind::BuiltinFloat:
  case swift::TypeKind::BuiltinRawPointer:
  case swift::TypeKind::BuiltinNativeObject:
  case swift::TypeKind::BuiltinUnsafeValueBuffer:
  case swift::TypeKind::BuiltinBridgeObject:
  case swift::TypeKind::BuiltinVector:
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
    auto tuple_type = cast<swift::TupleType>(swift_can_type);
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
        nominal->getModuleContext(), property);
    return ToCompilerType(child_swift_type.getPointer());
  }

  case swift::TypeKind::Protocol:
  case swift::TypeKind::ProtocolComposition: {
    ProtocolInfo protocol_info;
    if (!GetProtocolTypeInfo(ToCompilerType(GetSwiftType(type)), protocol_info))
      break;

    if (idx >= protocol_info.m_num_storage_words)
      break;

    CompilerType compiler_type = ToCompilerType(GetSwiftType(type));
    CompilerType child_type;
    std::tie(child_type, name) = GetExistentialTypeChild(
        GetASTContext(), compiler_type, protocol_info, idx);

    llvm::Optional<uint64_t> child_size = child_type.GetByteSize(nullptr);
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

  case swift::TypeKind::PrimaryArchetype:
  case swift::TypeKind::OpenedArchetype:
  case swift::TypeKind::OpaqueTypeArchetype:
  case swift::TypeKind::NestedArchetype:
  case swift::TypeKind::Function:
  case swift::TypeKind::GenericFunction:
  case swift::TypeKind::LValue:
  case swift::TypeKind::UnboundGeneric:
  case swift::TypeKind::DynamicSelf:
    break;

  case swift::TypeKind::Optional:
  case swift::TypeKind::TypeAlias:
  case swift::TypeKind::Paren:
  case swift::TypeKind::Dictionary:
  case swift::TypeKind::ArraySlice:
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
  case swift::TypeKind::BuiltinDefaultActorStorage:
  case swift::TypeKind::BuiltinExecutor:
  case swift::TypeKind::BuiltinJob:
  case swift::TypeKind::BuiltinRawUnsafeContinuation:
  case swift::TypeKind::Error:
  case swift::TypeKind::Module:
  case swift::TypeKind::InOut:
  case swift::TypeKind::VariadicSequence:
  case swift::TypeKind::Placeholder:
  case swift::TypeKind::SILBlockStorage:
  case swift::TypeKind::SILBox:
  case swift::TypeKind::SILFunction:
  case swift::TypeKind::SILToken:
  case swift::TypeKind::TypeVariable:
  case swift::TypeKind::Unresolved:
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
  case swift::TypeKind::Tuple:
  case swift::TypeKind::GenericTypeParam:
  case swift::TypeKind::DependentMember:
  case swift::TypeKind::Enum:
  case swift::TypeKind::Struct:
  case swift::TypeKind::Class:
  case swift::TypeKind::Protocol:
  case swift::TypeKind::Metatype:
  case swift::TypeKind::PrimaryArchetype:
  case swift::TypeKind::OpenedArchetype:
  case swift::TypeKind::OpaqueTypeArchetype:
  case swift::TypeKind::NestedArchetype:
  case swift::TypeKind::Function:
  case swift::TypeKind::GenericFunction:
  case swift::TypeKind::ProtocolComposition:
    return 0;
  case swift::TypeKind::LValue:
    return 1;
  case swift::TypeKind::UnboundGeneric:
  case swift::TypeKind::BoundGenericClass:
  case swift::TypeKind::BoundGenericEnum:
  case swift::TypeKind::BoundGenericStruct:
  case swift::TypeKind::ExistentialMetatype:
  case swift::TypeKind::DynamicSelf:
    return 0;

  case swift::TypeKind::Optional:
  case swift::TypeKind::TypeAlias:
  case swift::TypeKind::Paren:
  case swift::TypeKind::Dictionary:
  case swift::TypeKind::ArraySlice:
    assert(false && "Not a canonical type");
    break;
  }

  return 0;
}

static llvm::Optional<uint64_t> GetInstanceVariableOffset_Metadata(
    ValueObject *valobj, ExecutionContext *exe_ctx, const CompilerType &type,
    StringRef ivar_name, const CompilerType &ivar_type) {
  llvm::SmallString<1> m_description;
  LOG_PRINTF(LIBLLDB_LOG_TYPES, "ivar_name = %s, type = %s",
             ivar_name.str().c_str(), type.GetTypeName().AsCString());

  Process *process = exe_ctx->GetProcessPtr();
  if (!process) {
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "no process");
    return {};
  }

  SwiftLanguageRuntime *runtime = SwiftLanguageRuntime::Get(process);
  if (!runtime) {
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "no runtime");
    return {};
  }

  Status error;
  llvm::Optional<uint64_t> offset =
      runtime->GetMemberVariableOffset(type, valobj, ivar_name, &error);
  if (offset)
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "for %s: %llu", ivar_name.str().c_str(),
               *offset);
  else
    LOG_PRINTF(LIBLLDB_LOG_TYPES, "resolver failure: %s", error.AsCString());

  return offset;
}

static llvm::Optional<uint64_t>
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

bool SwiftASTContext::IsNonTriviallyManagedReferenceType(
    const CompilerType &type, NonTriviallyManagedReferenceStrategy &strategy,
    CompilerType *underlying_type) {
  if (swift::CanType swift_can_type = ::GetCanonicalSwiftType(type)) {
    const swift::TypeKind type_kind = swift_can_type->getKind();
    switch (type_kind) {
    default:
      break;
    case swift::TypeKind::UnmanagedStorage: {
      strategy = NonTriviallyManagedReferenceStrategy::eUnmanaged;
      if (underlying_type)
        *underlying_type = ToCompilerType(
            swift_can_type->getReferenceStorageReferent().getPointer());
    }
      return true;
    case swift::TypeKind::UnownedStorage: {
      strategy = NonTriviallyManagedReferenceStrategy::eUnowned;
      if (underlying_type)
        *underlying_type = ToCompilerType(
            swift_can_type->getReferenceStorageReferent().getPointer());
    }
      return true;
    case swift::TypeKind::WeakStorage: {
      strategy = NonTriviallyManagedReferenceStrategy::eWeak;
      if (underlying_type)
        *underlying_type = ToCompilerType(
            swift_can_type->getReferenceStorageReferent().getPointer());
    }
      return true;
    }
  }
  return false;
}

CompilerType SwiftASTContext::GetChildCompilerTypeAtIndex(
    opaque_compiler_type_t type, ExecutionContext *exe_ctx, size_t idx,
    bool transparent_pointers, bool omit_empty_base_classes,
    bool ignore_array_bounds, std::string &child_name,
    uint32_t &child_byte_size, int32_t &child_byte_offset,
    uint32_t &child_bitfield_bit_size, uint32_t &child_bitfield_bit_offset,
    bool &child_is_base_class, bool &child_is_deref_of_parent,
    ValueObject *valobj, uint64_t &language_flags) {
  VALID_OR_RETURN_CHECK_TYPE(type, CompilerType());
  LLDB_SCOPED_TIMER();

  auto get_type_size = [&exe_ctx](uint32_t &result, CompilerType type) {
    auto *exe_scope =
        exe_ctx ? exe_ctx->GetBestExecutionContextScope() : nullptr;
    llvm::Optional<uint64_t> size = type.GetByteSize(exe_scope);
    if (!size)
      return false;
    result = *size;
    return true;
  };

  language_flags = 0;
  swift::CanType swift_can_type(GetCanonicalSwiftType(type));
  assert(&swift_can_type->getASTContext() == GetASTContext());

  const swift::TypeKind type_kind = swift_can_type->getKind();
  switch (type_kind) {
  case swift::TypeKind::BuiltinDefaultActorStorage:
  case swift::TypeKind::BuiltinExecutor:
  case swift::TypeKind::BuiltinJob:
  case swift::TypeKind::BuiltinRawUnsafeContinuation:
  case swift::TypeKind::Error:
  case swift::TypeKind::Module:
  case swift::TypeKind::InOut:
  case swift::TypeKind::VariadicSequence:
  case swift::TypeKind::Placeholder:
  case swift::TypeKind::SILBlockStorage:
  case swift::TypeKind::SILBox:
  case swift::TypeKind::SILFunction:
  case swift::TypeKind::SILToken:
  case swift::TypeKind::TypeVariable:
  case swift::TypeKind::Unresolved:
    break;
  case swift::TypeKind::BuiltinInteger:
  case swift::TypeKind::BuiltinIntegerLiteral:
  case swift::TypeKind::BuiltinFloat:
  case swift::TypeKind::BuiltinRawPointer:
  case swift::TypeKind::BuiltinNativeObject:
  case swift::TypeKind::BuiltinUnsafeValueBuffer:
  case swift::TypeKind::BuiltinBridgeObject:
  case swift::TypeKind::BuiltinVector:
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
  case swift::TypeKind::GenericTypeParam:
  case swift::TypeKind::DependentMember:
    break;

  case swift::TypeKind::Enum:
  case swift::TypeKind::BoundGenericEnum: {
    SwiftEnumDescriptor *cached_enum_info = GetCachedEnumInfo(type);
    if (cached_enum_info &&
        idx < cached_enum_info->GetNumElementsWithPayload()) {
      const SwiftEnumDescriptor::ElementInfo *element_info =
          cached_enum_info->GetElementWithPayloadAtIndex(idx);
      child_name.assign(element_info->name.GetCString());
      if (!get_type_size(child_byte_size, element_info->payload_type))
        return {};
      child_byte_offset = 0;
      child_bitfield_bit_size = 0;
      child_bitfield_bit_offset = 0;
      child_is_base_class = false;
      child_is_deref_of_parent = false;
      if (element_info->is_indirect) {
        language_flags |= LanguageFlags::eIsIndirectEnumCase;
        return ToCompilerType(GetASTContext()->TheRawPointerType.getPointer());
      } else
        return element_info->payload_type;
    }
  } break;

  case swift::TypeKind::Tuple: {
    auto tuple_type = cast<swift::TupleType>(swift_can_type);
    if (idx >= tuple_type->getNumElements())
      break;

    const auto &child = tuple_type->getElement(idx);

    // Format the integer.
    llvm::SmallString<16> printed_idx;
    llvm::raw_svector_ostream(printed_idx) << idx;
    child_name = GetTupleElementName(tuple_type, idx, printed_idx);

    CompilerType child_type = ToCompilerType(child.getType().getPointer());
    if (!get_type_size(child_byte_size, child_type))
      return {};
    child_is_base_class = false;
    child_is_deref_of_parent = false;

    CompilerType compiler_type = ToCompilerType(GetSwiftType(type));
    llvm::Optional<uint64_t> offset = GetInstanceVariableOffset(
        valobj, exe_ctx, compiler_type, printed_idx.c_str(), child_type);
    if (!offset)
      return {};

    child_byte_offset = *offset;
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
        if (!get_type_size(child_byte_size, superclass_type))
          return {};
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
                ToCompilerType(VD->getType().getPointer());
            child_name = VD->getNameStr().str();
            if (!get_type_size(child_byte_size, child_type))
              return {};
            child_is_base_class = false;
            child_is_deref_of_parent = false;
            child_byte_offset = 0;
            child_bitfield_bit_size = 0;
            child_bitfield_bit_offset = 0;
            return child_type;
          }
          return {};
        }

    auto stored_properties = GetStoredProperties(nominal);
    if (idx >= stored_properties.size())
      break;

    // Find the stored property with this index.
    auto property = stored_properties[idx];
    swift::Type child_swift_type = swift_can_type->getTypeOfMember(
        nominal->getModuleContext(), property);

    CompilerType child_type = ToCompilerType(child_swift_type.getPointer());
    child_name = property->getBaseName().userFacingName().str();
    if (!get_type_size(child_byte_size, child_type))
      return {};
    child_is_base_class = false;
    child_is_deref_of_parent = false;

    CompilerType compiler_type = ToCompilerType(GetSwiftType(type));
    llvm::Optional<uint64_t> offset = GetInstanceVariableOffset(
        valobj, exe_ctx, compiler_type, child_name.c_str(), child_type);
    if (!offset)
      return {};

    child_byte_offset = *offset;
    child_bitfield_bit_size = 0;
    child_bitfield_bit_offset = 0;
    return child_type;
  }

  case swift::TypeKind::Protocol:
  case swift::TypeKind::ProtocolComposition: {
    ProtocolInfo protocol_info;
    if (!GetProtocolTypeInfo(ToCompilerType(GetSwiftType(type)), protocol_info))
      break;

    if (idx >= protocol_info.m_num_storage_words)
      break;

    CompilerType compiler_type = ToCompilerType(GetSwiftType(type));
    CompilerType child_type;
    std::tie(child_type, child_name) = GetExistentialTypeChild(
        GetASTContext(), compiler_type, protocol_info, idx);
    if (!get_type_size(child_byte_size, child_type))
      return {};
    child_byte_offset = idx * child_byte_size;
    child_bitfield_bit_size = 0;
    child_bitfield_bit_offset = 0;
    child_is_base_class = false;
    child_is_deref_of_parent = false;

    return child_type;
  }

  case swift::TypeKind::ExistentialMetatype:
  case swift::TypeKind::Metatype:
    break;

  case swift::TypeKind::PrimaryArchetype:
  case swift::TypeKind::OpenedArchetype:
  case swift::TypeKind::OpaqueTypeArchetype:
  case swift::TypeKind::NestedArchetype:
  case swift::TypeKind::Function:
  case swift::TypeKind::GenericFunction:
    break;

  case swift::TypeKind::LValue:
    if (idx < GetNumChildren(type, omit_empty_base_classes, exe_ctx)) {
      CompilerType pointee_clang_type(GetNonReferenceType(type));
      Flags pointee_clang_type_flags(pointee_clang_type.GetTypeInfo());
      const char *parent_name = valobj ? valobj->GetName().GetCString() : NULL;
      if (parent_name) {
        child_name.assign(1, '&');
        child_name += parent_name;
      }

      // We have a pointer to a simple type
      if (idx == 0) {
        if (!get_type_size(child_byte_size, pointee_clang_type))
          return {};
        child_byte_offset = 0;
        return pointee_clang_type;
      }
    }
    break;
  case swift::TypeKind::UnboundGeneric:
  case swift::TypeKind::DynamicSelf:
    break;

  case swift::TypeKind::Optional:
  case swift::TypeKind::TypeAlias:
  case swift::TypeKind::Paren:
  case swift::TypeKind::Dictionary:
  case swift::TypeKind::ArraySlice:
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
    opaque_compiler_type_t type, const char *name, ExecutionContext *exe_ctx,
    bool omit_empty_base_classes, std::vector<uint32_t> &child_indexes) {
  VALID_OR_RETURN_CHECK_TYPE(type, 0);
  LLDB_SCOPED_TIMER();

  if (name && name[0]) {
    swift::CanType swift_can_type(GetCanonicalSwiftType(type));

    const swift::TypeKind type_kind = swift_can_type->getKind();
    switch (type_kind) {
    case swift::TypeKind::BuiltinDefaultActorStorage:
    case swift::TypeKind::BuiltinExecutor:
    case swift::TypeKind::BuiltinJob:
    case swift::TypeKind::BuiltinRawUnsafeContinuation:
    case swift::TypeKind::Error:
    case swift::TypeKind::Module:
    case swift::TypeKind::InOut:
    case swift::TypeKind::VariadicSequence:
    case swift::TypeKind::Placeholder:
    case swift::TypeKind::SILBlockStorage:
    case swift::TypeKind::SILBox:
    case swift::TypeKind::SILFunction:
    case swift::TypeKind::SILToken:
    case swift::TypeKind::TypeVariable:
    case swift::TypeKind::Unresolved:
      break;
    case swift::TypeKind::BuiltinInteger:
    case swift::TypeKind::BuiltinIntegerLiteral:
    case swift::TypeKind::BuiltinFloat:
    case swift::TypeKind::BuiltinRawPointer:
    case swift::TypeKind::BuiltinNativeObject:
    case swift::TypeKind::BuiltinUnsafeValueBuffer:
    case swift::TypeKind::BuiltinBridgeObject:
    case swift::TypeKind::BuiltinVector:
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
      uint32_t tuple_idx = StringConvert::ToUInt32(name, UINT32_MAX);
      if (tuple_idx != UINT32_MAX) {
        if (tuple_idx < tuple_type->getNumElements()) {
          child_indexes.push_back(tuple_idx);
          return child_indexes.size();
        } else
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
    case swift::TypeKind::ProtocolComposition: {
      ProtocolInfo protocol_info;
      if (!GetProtocolTypeInfo(ToCompilerType(GetSwiftType(type)),
                               protocol_info))
        break;

      CompilerType compiler_type = ToCompilerType(GetSwiftType(type));
      for (unsigned idx : swift::range(protocol_info.m_num_storage_words)) {
        CompilerType child_type;
        std::string child_name;
        std::tie(child_type, child_name) = GetExistentialTypeChild(
            GetASTContext(), compiler_type, protocol_info, idx);
        if (name == child_name) {
          child_indexes.push_back(idx);
          return child_indexes.size();
        }
      }
    } break;

    case swift::TypeKind::ExistentialMetatype:
    case swift::TypeKind::Metatype:
      break;

    case swift::TypeKind::PrimaryArchetype:
    case swift::TypeKind::OpenedArchetype:
    case swift::TypeKind::OpaqueTypeArchetype:
    case swift::TypeKind::NestedArchetype:
    case swift::TypeKind::Function:
    case swift::TypeKind::GenericFunction:
      break;
    case swift::TypeKind::LValue: {
      CompilerType pointee_clang_type(GetNonReferenceType(type));

      if (pointee_clang_type.IsAggregateType()) {
        return pointee_clang_type.GetIndexOfChildMemberWithName(
            name, exe_ctx, omit_empty_base_classes, child_indexes);
      }
    } break;
    case swift::TypeKind::UnboundGeneric:
    case swift::TypeKind::DynamicSelf:
      break;

    case swift::TypeKind::Optional:
    case swift::TypeKind::TypeAlias:
    case swift::TypeKind::Paren:
    case swift::TypeKind::Dictionary:
    case swift::TypeKind::ArraySlice:
      assert(false && "Not a canonical type");
      break;
    }
  }
  return 0;
}

size_t SwiftASTContext::GetNumTemplateArguments(opaque_compiler_type_t type) {
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
  swift::CanType swift_can_type = ::GetCanonicalSwiftType(type);
  if (!swift_can_type)
    return false;
  auto *ast = GetSwiftASTContext(&swift_can_type->getASTContext());
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
  if (auto *unbound_generic_type =
          swift_can_type->getAs<swift::UnboundGenericType>())
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
  assert(&swift_can_type->getASTContext() == GetASTContext());
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
  assert(&swift_can_type->getASTContext() == GetASTContext());
  if (auto *unbound_generic_type =
          swift_can_type->getAs<swift::UnboundGenericType>()) {
    auto *nominal_type_decl = unbound_generic_type->getDecl();
    swift::GenericSignature generic_sig =
        nominal_type_decl->getGenericSignature();
    swift::TypeArrayView<swift::GenericTypeParamType> depView 
        = generic_sig.getGenericParams();
    swift::Type depTy = depView[idx];
    return ToCompilerType({nominal_type_decl->mapTypeIntoContext(depTy)
                               ->castTo<swift::ArchetypeType>()});
  }

  return {};
}

CompilerType SwiftASTContext::GetGenericArgumentType(CompilerType ct,
                                                     size_t idx) {
  if (swift::Type swift_type = ::GetSwiftType(ct)) {
    auto *ast = GetSwiftASTContext(&swift_type->getASTContext());
    return ast->GetGenericArgumentType(swift_type.getPointer(), idx);
  }
  return {};
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

  return {this, type};
}

LazyBool SwiftASTContext::ShouldPrintAsOneLiner(opaque_compiler_type_t type,
                                                ValueObject *valobj) {
  if (type) {
    CompilerType can_compiler_type(GetCanonicalType(type));
    auto *ts = llvm::cast<TypeSystemSwift>(can_compiler_type.GetTypeSystem());
    if (ts->IsImportedType(can_compiler_type.GetOpaqueQualType(), nullptr))
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
#define DEPTH_INCREMENT 2

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
    opaque_compiler_type_t type, Stream *s, lldb::Format format,
    const lldb_private::DataExtractor &data, lldb::offset_t byte_offset,
    size_t byte_size, uint32_t bitfield_bit_size, uint32_t bitfield_bit_offset,
    ExecutionContextScope *exe_scope, bool is_base_class) {
  VALID_OR_RETURN_CHECK_TYPE(type, false);
  LLDB_SCOPED_TIMER();

  swift::CanType swift_can_type(GetCanonicalSwiftType(type));

  const swift::TypeKind type_kind = swift_can_type->getKind();
  switch (type_kind) {
  case swift::TypeKind::BuiltinDefaultActorStorage:
  case swift::TypeKind::BuiltinExecutor:
  case swift::TypeKind::BuiltinJob:
  case swift::TypeKind::BuiltinRawUnsafeContinuation:
  case swift::TypeKind::Error:
  case swift::TypeKind::Module:
  case swift::TypeKind::InOut:
  case swift::TypeKind::VariadicSequence:
  case swift::TypeKind::Placeholder:
  case swift::TypeKind::SILBlockStorage:
  case swift::TypeKind::SILBox:
  case swift::TypeKind::SILFunction:
  case swift::TypeKind::SILToken:
  case swift::TypeKind::TypeVariable:
  case swift::TypeKind::Unresolved:
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
  case swift::TypeKind::PrimaryArchetype:
  case swift::TypeKind::OpaqueTypeArchetype:
  case swift::TypeKind::OpenedArchetype:
  case swift::TypeKind::NestedArchetype:
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
    return DumpDataExtractor(data, s, byte_offset, format, byte_size,
                             item_count, UINT32_MAX, LLDB_INVALID_ADDRESS,
                             bitfield_bit_size, bitfield_bit_offset, exe_scope);
  } break;
  case swift::TypeKind::BuiltinVector:
    break;

  case swift::TypeKind::Tuple:
    break;

  case swift::TypeKind::UnmanagedStorage:
  case swift::TypeKind::UnownedStorage:
  case swift::TypeKind::WeakStorage:
    return ToCompilerType(swift_can_type->getReferenceStorageReferent())
        .DumpTypeValue(s, format, data, byte_offset, byte_size,
                       bitfield_bit_size, bitfield_bit_offset, exe_scope,
                       is_base_class);
  case swift::TypeKind::Enum:
  case swift::TypeKind::BoundGenericEnum: {
    SwiftEnumDescriptor *cached_enum_info = GetCachedEnumInfo(type);
    if (cached_enum_info) {
      auto enum_elem_info = cached_enum_info->GetElementFromData(data, true);
      if (enum_elem_info)
        s->Printf("%s", enum_elem_info->name.GetCString());
      else {
        lldb::offset_t ptr = 0;
        if (data.GetByteSize())
          s->Printf("<invalid> (0x%" PRIx8 ")", data.GetU8(&ptr));
        else
          s->Printf("<empty>");
      }
      return true;
    } else
      s->Printf("<unknown type>");
  } break;

  case swift::TypeKind::Struct:
  case swift::TypeKind::Protocol:
    return false;

  case swift::TypeKind::ExistentialMetatype:
  case swift::TypeKind::Metatype: {
    return DumpDataExtractor(data, s, byte_offset, eFormatPointer, byte_size, 1,
                             UINT32_MAX, LLDB_INVALID_ADDRESS,
                             bitfield_bit_size, bitfield_bit_offset, exe_scope);
  } break;

  case swift::TypeKind::ProtocolComposition:
  case swift::TypeKind::UnboundGeneric:
  case swift::TypeKind::BoundGenericStruct:
  case swift::TypeKind::DynamicSelf:
    break;

  case swift::TypeKind::Optional:
  case swift::TypeKind::TypeAlias:
  case swift::TypeKind::Paren:
  case swift::TypeKind::Dictionary:
  case swift::TypeKind::ArraySlice:
    assert(false && "Not a canonical type");
    break;
  }

  return 0;
}

bool SwiftASTContext::IsImportedType(opaque_compiler_type_t type,
                                     CompilerType *original_type) {
  VALID_OR_RETURN_CHECK_TYPE(type, false);
  LLDB_SCOPED_TIMER();
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
                               &objc_interface_decl->getASTContext()),
                           clang::QualType::getFromOpaquePtr(
                               objc_interface_decl->getTypeForDecl())
                               .getAsOpaquePtr());
        } else if (const clang::TypeDecl *type_decl =
                       llvm::dyn_cast<clang::TypeDecl>(clang_decl)) {
          *original_type = CompilerType(
              TypeSystemClang::GetASTContext(&type_decl->getASTContext()),
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

std::string SwiftASTContext::ImportName(const clang::NamedDecl *clang_decl) {
  if (auto clang_importer = GetClangImporter()) {
    swift::DeclName imported_name = clang_importer->importName(clang_decl, {});
    return imported_name.getBaseName().userFacingName().str();
  }
  return clang_decl->getName().str();
}

void SwiftASTContext::DumpSummary(opaque_compiler_type_t type,
                                  ExecutionContext *exe_ctx, Stream *s,
                                  const lldb_private::DataExtractor &data,
                                  lldb::offset_t data_byte_offset,
                                  size_t data_byte_size) {}

void SwiftASTContext::DumpTypeDescription(opaque_compiler_type_t type,
                                          lldb::DescriptionLevel level) {
  StreamFile s(stdout, false);
  DumpTypeDescription(type, &s, level);
}

void SwiftASTContext::DumpTypeDescription(opaque_compiler_type_t type,
                                          Stream *s,
                                          lldb::DescriptionLevel level) {
  DumpTypeDescription(type, s, false, true, level);
}

void SwiftASTContext::DumpTypeDescription(opaque_compiler_type_t type,
                                          bool print_help_if_available,
                                          bool print_extensions_if_available,
                                          lldb::DescriptionLevel level) {
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
                                          lldb::DescriptionLevel level) {
  LLDB_SCOPED_TIMER();
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

TypeSP SwiftASTContext::GetCachedType(ConstString mangled) {
  TypeSP type_sp;
  if (m_swift_type_map.Lookup(mangled.GetCString(), type_sp))
    return type_sp;
  else
    return TypeSP();
}

void SwiftASTContext::SetCachedType(ConstString mangled,
                                    const TypeSP &type_sp) {
  m_swift_type_map.Insert(mangled.GetCString(), type_sp);
}

DWARFASTParser *SwiftASTContext::GetDWARFParser() {
  if (!m_dwarf_ast_parser_ap)
    m_dwarf_ast_parser_ap.reset(new DWARFASTParserSwift(*this));
  return m_dwarf_ast_parser_ap.get();
}

std::vector<lldb::DataBufferSP> &
SwiftASTContext::GetASTVectorForModule(const Module *module) {
  return m_ast_file_data_map[const_cast<Module *>(module)];
}

SwiftASTContextForExpressions::SwiftASTContextForExpressions(
    std::string description, Target &target)
    : SwiftASTContext(std::move(description),
                      target.GetArchitecture().GetTriple(), &target),
      m_persistent_state_up(new SwiftPersistentExpressionState) {}

UserExpression *SwiftASTContextForExpressions::GetUserExpression(
    llvm::StringRef expr, llvm::StringRef prefix, lldb::LanguageType language,
    Expression::ResultType desired_type,
    const EvaluateExpressionOptions &options, ValueObject *ctx_obj) {
  TargetSP target_sp = m_target_wp.lock();
  if (!target_sp)
    return nullptr;
  if (ctx_obj != nullptr) {
    lldb_assert(0,
                "Swift doesn't support 'evaluate in the context"
                " of an object'.",
                __FUNCTION__, __FILE__, __LINE__);
    return nullptr;
  }

  return new SwiftUserExpression(*target_sp.get(), expr, prefix, language,
                                 desired_type, options);
}

PersistentExpressionState *
SwiftASTContextForExpressions::GetPersistentExpressionState() {
  return m_persistent_state_up.get();
}

static void DescribeFileUnit(Stream &s, swift::FileUnit *file_unit) {
  s.PutCString("kind = ");

  switch (file_unit->getKind()) {
  case swift::FileUnitKind::Source: {
    s.PutCString("Source, ");
    if (swift::SourceFile *source_file =
            llvm::dyn_cast<swift::SourceFile>(file_unit)) {
      s.Printf("filename = \"%s\", ", source_file->getFilename().str().c_str());
      s.PutCString("source file kind = ");
      switch (source_file->Kind) {
      case swift::SourceFileKind::Library:
        s.PutCString("Library");
        break;
      case swift::SourceFileKind::Main:
        s.PutCString("Main");
        break;
      case swift::SourceFileKind::SIL:
        s.PutCString("SIL");
        break;
      case swift::SourceFileKind::Interface:
        s.PutCString("Interface");
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
    swift::LoadedFile *loaded_file = llvm::cast<swift::LoadedFile>(file_unit);
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
                                        lldb::StackFrameWP &stack_frame_wp,
                                        Status &error) {
  LLDB_SCOPED_TIMER();
  if (!module.path.size())
    return nullptr;

  error.Clear();
  ConstString toplevel = module.path.front();
  const std::string &m_description = swift_ast_context.GetDescription();
  LOG_PRINTF(LIBLLDB_LOG_EXPRESSIONS, "Importing module %s",
             toplevel.AsCString());
  swift::ModuleDecl *swift_module = nullptr;
  lldb::StackFrameSP this_frame_sp(stack_frame_wp.lock());

  swift::ModuleDecl *imported_header_module =
      swift_ast_context.GetClangImporter()->getImportedHeaderModule();
  if (toplevel.GetStringRef() == imported_header_module->getName().str())
    swift_module = imported_header_module;
  else if (this_frame_sp) {
    lldb::ProcessSP process_sp(this_frame_sp->CalculateProcess());
    if (process_sp)
      swift_module =
          swift_ast_context.FindAndLoadModule(module, *process_sp.get(), error);
    else
      swift_module = swift_ast_context.GetModule(module, error);
  } else
    swift_module = swift_ast_context.GetModule(module, error);

  if (swift_module && IsDWARFImported(*swift_module)) {
    // This module was "imported" from DWARF. This basically means the
    // import as a Swift or Clang module failed. We have not yet
    // checked that DWARF debug info for this module actually exists
    // and there is no good mechanism to do so ahead of time.
    // We do know that we never load the stdlib from DWARF though.
    LOG_PRINTF(LIBLLDB_LOG_EXPRESSIONS,
               "\"Imported\" module %s via SwiftDWARFImporterDelegate "
               "(no Swift AST or Clang module found)",
               toplevel.AsCString());

    if (toplevel.GetStringRef() == swift::STDLIB_NAME)
      swift_module = nullptr;
  }

  if (!swift_module || !error.Success() || swift_ast_context.HasFatalErrors()) {
    LOG_PRINTF(LIBLLDB_LOG_EXPRESSIONS, "Couldn't import module %s: %s",
               toplevel.AsCString(), error.AsCString());

    if (!swift_module || swift_ast_context.HasFatalErrors()) {
      return nullptr;
    }
  }

  if (lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS)) {
    StreamString ss;
    for (swift::FileUnit *file_unit : swift_module->getFiles())
      DescribeFileUnit(ss, file_unit);
    LOG_PRINTF(LIBLLDB_LOG_EXPRESSIONS, "Imported module %s from {%s}",
               module.path.front().AsCString(), ss.GetData());
  }
  return swift_module;
}

bool SwiftASTContext::GetImplicitImports(
    SwiftASTContext &swift_ast_context, SymbolContext &sc,
    ExecutionContextScope &exe_scope, lldb::StackFrameWP &stack_frame_wp,
    llvm::SmallVectorImpl<swift::AttributedImport<swift::ImportedModule>>
        &modules,
    Status &error) {
  LLDB_SCOPED_TIMER();
  if (!swift_ast_context.GetCompileUnitImports(sc, stack_frame_wp, modules,
                                               error)) {
    return false;
  }

  auto *persistent_expression_state =
      sc.target_sp->GetSwiftPersistentExpressionState(exe_scope);

  // Get the hand-loaded modules from the SwiftPersistentExpressionState.
  for (auto &module_pair :
       persistent_expression_state->GetHandLoadedModules()) {

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
    auto *module =
        LoadOneModule(module_info, swift_ast_context, stack_frame_wp, error);
    if (!module)
      return false;

    attributed_import.module = swift::ImportedModule(module);
    modules.emplace_back(attributed_import);
  }
  return true;
}

bool SwiftASTContext::CacheUserImports(SwiftASTContext &swift_ast_context,
                                       SymbolContext &sc,
                                       ExecutionContextScope &exe_scope,
                                       lldb::StackFrameWP &stack_frame_wp,
                                       swift::SourceFile &source_file,
                                       Status &error) {
  llvm::SmallString<1> m_description;

  auto *persistent_expression_state =
      sc.target_sp->GetSwiftPersistentExpressionState(exe_scope);

  for (const auto &attributed_import : source_file.getImports()) {
    swift::ModuleDecl *module = attributed_import.module.importedModule;

    if (module) {
      std::string module_name;
      GetNameFromModule(module, module_name);
      if (!module_name.empty()) {
        SourceModule module_info;
        ConstString module_const_str(module_name);
        module_info.path.push_back(module_const_str);
        LOG_PRINTF(LIBLLDB_LOG_EXPRESSIONS,
                   "Performing auto import on found module: %s.\n",
                   module_name.c_str());
        if (!LoadOneModule(module_info, swift_ast_context, stack_frame_wp,
                           error))
          return false;

        // How do we tell we are in REPL or playground mode?
        persistent_expression_state->AddHandLoadedModule(module_const_str,
                                                         attributed_import);
      }
    }
  }
  return true;
}

bool SwiftASTContext::GetCompileUnitImports(
    SymbolContext &sc, lldb::StackFrameWP &stack_frame_wp,
    llvm::SmallVectorImpl<swift::AttributedImport<swift::ImportedModule>>
        &modules,
    Status &error) {
  return GetCompileUnitImportsImpl(sc, stack_frame_wp, &modules, error);
}

void SwiftASTContext::PerformCompileUnitImports(
    SymbolContext &sc, lldb::StackFrameWP &stack_frame_wp, Status &error) {
  GetCompileUnitImportsImpl(sc, stack_frame_wp, nullptr, error);
}

static std::pair<Module *, lldb::user_id_t>
GetCUSignature(CompileUnit &compile_unit) {
  return {compile_unit.GetModule().get(), compile_unit.GetID()};
}

bool SwiftASTContext::GetCompileUnitImportsImpl(
    SymbolContext &sc, lldb::StackFrameWP &stack_frame_wp,
    llvm::SmallVectorImpl<swift::AttributedImport<swift::ImportedModule>>
        *modules,
    Status &error) {
  LLDB_SCOPED_TIMER();
  CompileUnit *compile_unit = sc.comp_unit;
  if (compile_unit)
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
  swift_module.path.emplace_back("Swift");
  auto *stdlib =
      LoadOneModule(swift_module, *this, stack_frame_wp, error);
  if (!stdlib)
    return false;

  if (modules)
    modules->emplace_back(swift::ImportedModule(stdlib));

  if (!compile_unit || compile_unit->GetLanguage() != lldb::eLanguageTypeSwift)
    return true;

  for (const SourceModule &module : compile_unit->GetImportedModules()) {
    // When building the Swift stdlib with debug info these will
    // show up in "Swift.o", but we already imported them and
    // manually importing them will fail.
    if (module.path.size() &&
        llvm::StringSwitch<bool>(module.path.front().GetStringRef())
            .Cases("Swift", "SwiftShims", "Builtin", true)
            .Default(false))
      continue;

    auto *loaded_module =
        LoadOneModule(module, *this, stack_frame_wp, error);
    if (!loaded_module)
      return false;

    if (modules)
      modules->emplace_back(swift::ImportedModule(loaded_module));
  }
  return true;
}
