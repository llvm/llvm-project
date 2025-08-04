//===-- SwiftASTContext.h ---------------------------------------*- C++ -*-===//
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

#ifndef liblldb_SwiftASTContext_h_
#define liblldb_SwiftASTContext_h_

#include "Plugins/LanguageRuntime/Swift/LockGuarded.h"
#include "Plugins/TypeSystem/Swift/TypeSystemSwift.h"
#include "Plugins/TypeSystem/Swift/TypeSystemSwiftTypeRef.h"

#include "lldb/Core/Progress.h"
#include "lldb/Core/SwiftForward.h"
#include "lldb/Core/ThreadSafeDenseSet.h"
#include "lldb/Expression/DiagnosticManager.h"
#include "lldb/Utility/Either.h"

#include "swift/AST/Import.h"
#include "swift/AST/Module.h"
#include "swift/Demangling/ManglingFlavor.h"
#include "swift/Parse/ParseVersion.h"
#include "swift/Serialization/SerializationOptions.h"
#include "swift/SymbolGraphGen/SymbolGraphOptions.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Target/TargetOptions.h"

#include <memory>

namespace swift {
enum class IRGenDebugInfoLevel : unsigned;
class CanType;
class DependencyTracker;
struct ImplicitImportInfo;
class IRGenOptions;
class NominalTypeDecl;
class SearchPathOptions;
class SILModule;
struct TBDGenOptions;
class VarDecl;
class ModuleDecl;
class SourceFile;
class CASOptions;
struct PrintOptions;
class MemoryBufferSerializedModuleLoader;
namespace Demangle {
class Demangler;
class Node;
using NodePointer = Node *;
} // namespace Demangle
namespace irgen {
class FixedTypeInfo;
class TypeInfo;
} // namespace irgen
namespace serialization {
struct ValidationInfo;
class ExtendedValidationInfo;
} // namespace serialization
namespace Lowering {
class TypeConverter;
}
} // namespace swift

namespace clang {
namespace api_notes {
class APINotesManager;
}
class NamedDecl;
} // namespace clang

namespace llvm {
class LLVMContext;
}

class SwiftEnumDescriptor;

namespace lldb_private {

namespace plugin {
namespace dwarf {
class DWARFASTParser;
} // namespace dwarf
} // namespace plugin

struct SourceModule;
class SwiftASTContext;
class ClangExternalASTSourceCallbacks;
CompilerType ToCompilerType(swift::Type qual_type);

namespace detail {
/// Serves as the key for caching calls to LoadLibraryUsingPaths.
struct SwiftLibraryLookupRequest {
  std::string library_name;
  std::vector<std::string> search_paths;
  bool check_rpath = false;
  uint32_t process_uid = 0;

  bool operator==(const SwiftLibraryLookupRequest &o) const {
    return std::tie(library_name, search_paths, check_rpath, process_uid) ==
        std::tie(o.library_name, o.search_paths, o.check_rpath, o.process_uid);
  }
};
} // namespace detail
} // namespace lldb_private

namespace std {
template <> struct hash<lldb_private::detail::SwiftLibraryLookupRequest> {
  using argument_type = lldb_private::detail::SwiftLibraryLookupRequest;
  using result_type = std::size_t;

  result_type operator()(const argument_type &Arg) const {
    result_type result = std::hash<decltype(Arg.library_name)>()(Arg.library_name);
    result ^= std::hash<decltype(Arg.check_rpath)>()(Arg.check_rpath);
    result ^= std::hash<decltype(Arg.process_uid)>()(Arg.process_uid);
    for (const std::string &search_path : Arg.search_paths)
      result ^= std::hash<std::string>()(search_path);
    return result;
  }
};
} // end namespace std

namespace lldb_private {

using ThreadSafeASTContext = LockGuarded<swift::ASTContext>;

/// This "middle" class between TypeSystemSwiftTypeRef and
/// SwiftASTContextForExpressions will eventually go away, as more and
/// more functionality becomes available in TypeSystemSwiftTypeRef.
class SwiftASTContext : public TypeSystemSwift {
  /// LLVM RTTI support.
  static char ID;

public:
  typedef lldb_utility::Either<CompilerType, swift::Decl *> TypeOrDecl;

  /// LLVM RTTI support.
  /// \{
  bool isA(const void *ClassID) const override {
    return ClassID == &ID || TypeSystemSwift::isA(ClassID);
  }
  static bool classof(const TypeSystem *ts) { return ts->isA(&ID); }
  /// \}

private:
  struct EitherComparator {
    bool operator()(const TypeOrDecl &r1, const TypeOrDecl &r2) const {
      auto r1_as1 = r1.GetAs<CompilerType>();
      auto r1_as2 = r1.GetAs<swift::Decl *>();

      auto r2_as1 = r2.GetAs<CompilerType>();
      auto r2_as2 = r2.GetAs<swift::Decl *>();

      if (r1_as1.has_value() && r2_as1.has_value())
        return r1_as1.value() < r2_as1.value();

      if (r1_as2.has_value() && r2_as2.has_value())
        return r1_as2.value() < r2_as2.value();

      if (r1_as1.has_value() && r2_as2.has_value())
        return (void *)r1_as1->GetOpaqueQualType() < (void *)r2_as2.value();

      if (r1_as2.has_value() && r2_as1.has_value())
        return (void *)r1_as2.value() < (void *)r2_as1->GetOpaqueQualType();

      return false;
    }
  };

public:
  typedef std::set<TypeOrDecl, EitherComparator> TypesOrDecls;

  /// Provide the global LLVMContext.
  static llvm::LLVMContext &GetGlobalLLVMContext();

protected:
  // Constructors and destructors
  SwiftASTContext(std::string description, lldb::ModuleSP module_sp,
                  TypeSystemSwiftTypeRefSP typeref_typesystem);

public:

  SwiftASTContext(const SwiftASTContext &rhs) = delete;

  virtual ~SwiftASTContext();

#ifndef NDEBUG
  /// Provided only for unit tests.
  SwiftASTContext();
#endif

  /// Create a SwiftASTContext from a Module.  This context is used
  /// for frame variable and uses ClangImporter options specific to
  /// this lldb::Module.  The optional target is necessary when
  /// creating a module-specific scratch context.
  static lldb::TypeSystemSP
  CreateInstance(lldb::LanguageType language, Module &module,
                 TypeSystemSwiftTypeRef &typeref_typesystem);
  /// Create a SwiftASTContextForExpressions taylored to a specific symbol
  /// context.
  static lldb::TypeSystemSP
  CreateInstance(const SymbolContext &sc,
                 TypeSystemSwiftTypeRef &typeref_typesystem, bool repl = false,
                 bool playground = false, const char *extra_options = nullptr);

  static void EnumerateSupportedLanguages(
      std::set<lldb::LanguageType> &languages_for_types,
      std::set<lldb::LanguageType> &languages_for_expressions);

  /// Set LangOpt overrides LLDB needs.
  void SetCompilerInvocationLLDBOverrides();
  
  bool SupportsLanguage(lldb::LanguageType language) override;

  SwiftASTContextSP GetSwiftASTContext(const SymbolContext &sc) const override {
    if (auto ts = GetTypeSystemSwiftTypeRef())
      return ts->GetSwiftASTContext(sc);
    return {};
  }

  TypeSystemSwiftTypeRefSP GetTypeSystemSwiftTypeRef() override {
    return m_typeref_typesystem.lock();
  }

  std::shared_ptr<const TypeSystemSwiftTypeRef>
  GetTypeSystemSwiftTypeRef() const override {
    return m_typeref_typesystem.lock();
  }

  Status IsCompatible() override;

  swift::SourceManager &GetSourceManager();

  swift::LangOptions &GetLanguageOptions();

  swift::symbolgraphgen::SymbolGraphOptions &GetSymbolGraphOptions();

  swift::CASOptions &GetCASOptions();

  swift::TypeCheckerOptions &GetTypeCheckerOptions();

  swift::DiagnosticEngine &GetDiagnosticEngine();

  swift::SearchPathOptions &GetSearchPathOptions();

  swift::SerializationOptions &GetSerializationOptions();

  void InitializeSearchPathOptions(
      llvm::ArrayRef<std::pair<std::string, bool>> module_search_paths,
      llvm::ArrayRef<std::pair<std::string, bool>> framework_search_paths);

  swift::ClangImporterOptions &GetClangImporterOptions();

  swift::CompilerInvocation &GetCompilerInvocation();

  swift::SILOptions &GetSILOptions();

  ThreadSafeASTContext GetASTContext();

  ThreadSafeASTContext GetASTContext() const;

  swift::IRGenDebugInfoLevel GetGenerateDebugInfo();

  static swift::PrintOptions
  GetUserVisibleTypePrintingOptions(bool print_help_if_available);

  void SetGenerateDebugInfo(swift::IRGenDebugInfoLevel b);

  bool AddModuleSearchPath(llvm::StringRef path);

  void ConfigureModuleValidation(std::vector<std::string> &extra_args);

  /// Add a list of Clang arguments to the ClangImporter options and
  /// apply the working directory to any relative paths.
  void AddExtraClangArgs(
      const std::vector<std::string> &ExtraArgs,
      const std::vector<std::pair<std::string, bool>> module_search_paths,
      const std::vector<std::pair<std::string, bool>> framework_search_paths,
      llvm::StringRef overrideOpts = "");

  void AddExtraClangCC1Args(
      const std::vector<std::string> &source,
      const std::vector<std::pair<std::string, bool>> module_search_paths,
      const std::vector<std::pair<std::string, bool>> framework_search_paths,
      std::vector<std::string> &dest);
  static void AddExtraClangArgs(const std::vector<std::string>& source,
                                std::vector<std::string>& dest);
  static std::string GetPluginServer(llvm::StringRef plugin_library_path);
  /// Removes nonexisting VFS overlay options.
  static void FilterClangImporterOptions(std::vector<std::string> &extra_args,
                                         SwiftASTContext *ctx = nullptr);

  /// Add the target's swift-extra-clang-flags to the ClangImporter options.
  void AddUserClangArgs(TargetProperties &props);

  llvm::StringRef GetPlatformSDKPath() const { return m_platform_sdk_path; }

  void SetPlatformSDKPath(std::string &&sdk_path) {
    m_platform_sdk_path = sdk_path;
  }

  void SetPlatformSDKPath(llvm::StringRef path) {
    m_platform_sdk_path = path.str();
  }

  /// \return the ExtraArgs of the ClangImporterOptions.
  const std::vector<std::string> &GetClangArguments();

  /// Attempt to create an empty Swift module.
  llvm::Expected<swift::ModuleDecl &>
  CreateEmptyModule(std::string module_name);

  /// Attempt to create a Swift module.
  ///
  /// \param importInfo Information about which modules should be implicitly
  /// imported by each file of the module.
  /// \param populateFiles A function which populates the files for the module.
  /// Once called, the module's list of files may not change.
  llvm::Expected<swift::ModuleDecl &>
  CreateModule(std::string module_name, swift::ImplicitImportInfo importInfo,
               swift::ModuleDecl::PopulateFilesFn populateFiles);

  /// An RAII object to install a progress report callback.
  class ModuleImportProgressRAII {
    lldb::TypeSystemSP m_ts;
    Progress m_progress;

  public:
    ModuleImportProgressRAII(SwiftASTContext &ctx, std::string category);
    ~ModuleImportProgressRAII();
  };

  /// Install and return a module import RAII object.
  std::unique_ptr<ModuleImportProgressRAII>
  GetModuleImportProgressRAII(std::string category);

  // This function should only be called when all search paths
  // for all items in a swift::ASTContext have been setup to
  // allow for imports to happen correctly. Use with caution,
  // or use the GetModule() call that takes a FileSpec.
  llvm::Expected<swift::ModuleDecl &> GetModule(const SourceModule &module,
                                                bool *cached = nullptr);
  llvm::Expected<swift::ModuleDecl &> GetModule(const FileSpec &module_spec);
  llvm::Expected<swift::ModuleDecl &> ImportStdlib();

  void CacheModule(std::string module_name, swift::ModuleDecl *module);

  /// Call this after the search paths are set up, it will find the module given
  /// by module, load the module into the AST context, and (if import_dylib is
  /// set) also load any "LinkLibraries" that the module requires.
  template <typename ModuleT>
  swift::ModuleDecl *FindAndLoadModule(const ModuleT &module, Process &process,
                                       bool import_dylib, Status &error);

  void LoadModule(swift::ModuleDecl *swift_module, Process &process,
                  Status &error);

  /// Collect Swift modules in the .swift_ast section of \p module.
  void RegisterSectionModules(Module &module,
                              std::vector<std::string> &module_names);
  /// Import Swift modules in the .swift_ast section of \p module.
  void ImportSectionModules(Module &module,
                            const std::vector<std::string> &module_names);

  // Swift modules that are backed by dylibs (libFoo.dylib) rather than
  // frameworks don't actually record the library dependencies in the module.
  // This will hand load any libraries that are on the IRGen LinkLibraries list
  // using the compiler's search paths.
  // It doesn't do frameworks since frameworks don't need it and this is kind of
  // a hack anyway.

  void LoadExtraDylibs(Process &process, Status &error);

  swift::Identifier GetIdentifier(const llvm::StringRef &name);

  CompilerType FindType(const char *name, swift::ModuleDecl *swift_module);

  std::optional<SwiftASTContext::TypeOrDecl>
  FindTypeOrDecl(const char *name, swift::ModuleDecl *swift_module);

  size_t FindTypes(const char *name, swift::ModuleDecl *swift_module,
                   std::set<CompilerType> &results, bool append = true);

  size_t FindTypesOrDecls(const char *name, swift::ModuleDecl *swift_module,
                          TypesOrDecls &results, bool append = true);

  size_t FindContainedTypeOrDecl(llvm::StringRef name,
                                 TypeOrDecl container_type_or_decl,
                                 TypesOrDecls &results, bool append = true);

  size_t FindType(const char *name, std::set<CompilerType> &results,
                  bool append = true);

  /// Reconstruct a Swift AST type from a mangled name by looking its
  /// components up in Swift modules. Diagnose a warning on error.
  swift::TypeBase *ReconstructTypeOrWarn(ConstString mangled_typename);
  /// Reconstruct a Swift AST type from a mangled name by looking its
  /// components up in Swift modules.
  llvm::Expected<swift::TypeBase *>
  ReconstructType(ConstString mangled_typename);
  /// Reconstruct a Swift AST type from a mangled name by looking its
  /// components up in Swift modules.
  CompilerType
  GetTypeFromMangledTypename(ConstString mangled_typename) override;

  // Retrieve the Swift.AnyObject type.
  CompilerType GetAnyObjectType();

  /// Import and Swiftify a Clang type.
  /// \return Returns an invalid type if unsuccessful.
  CompilerType ImportClangType(CompilerType clang_type);

  /// Use ClangImporter to determine the swiftified name of \p
  /// clang_decl.
  std::string GetSwiftName(const clang::Decl *clang_decl,
                           TypeSystemClang &clang_typesystem) override;

  CompilerType GetBuiltinIntType();

  /// Attempts to convert a Clang type into a Swift type.
  /// For example, int is converted to Int32.
  CompilerType ConvertClangTypeToSwiftType(CompilerType clang_type) override;

  bool TypeHasArchetype(CompilerType type);

  /// Use \p ClangImporter to swiftify the decl's name.
  std::string ImportName(const clang::NamedDecl *clang_decl);

  static SwiftASTContext *GetSwiftASTContext(swift::ASTContext *ast);

  swift::irgen::IRGenerator &GetIRGenerator(swift::IRGenOptions &opts,
                                            swift::SILModule &module);

  swift::irgen::IRGenModule &GetIRGenModule();

  lldb::TargetWP GetTargetWP() const override;
  llvm::Triple GetTriple() const;

  bool SetTriple(const llvm::Triple triple, lldb_private::Module *module);
  void SetTriple(const SymbolContext &sc, const llvm::Triple triple) override;

  /// Condition a triple to be safe for use with Swift.  Swift is
  /// really peculiar about what CPU types it thinks it has standard
  /// libraries for.
  static llvm::Triple GetSwiftFriendlyTriple(llvm::Triple triple);

  CompilerType GetCompilerType(swift::TypeBase *swift_type);
  CompilerType GetCompilerType(ConstString mangled_name);
  /// Import compiler_type into this context and return the swift::Type.
  llvm::Expected<swift::Type> GetSwiftType(CompilerType compiler_type);
  /// Import compiler_type into this context and return the swift::CanType.
  swift::CanType GetCanonicalSwiftType(CompilerType compiler_type);
private:

protected:
  swift::Type GetSwiftType(lldb::opaque_compiler_type_t opaque_type);
  swift::Type GetSwiftTypeIgnoringErrors(CompilerType compiler_type);
  swift::CanType
  GetCanonicalSwiftType(lldb::opaque_compiler_type_t opaque_type);

public:

  /// Imports the type from the passed in type into this SwiftASTContext. The
  /// type must be a Swift type. If the type can be imported, returns the
  /// CompilerType for the imported type.
  /// If it cannot be, returns an invalid CompilerType, and sets the error to
  /// indicate what went wrong.
  CompilerType ImportType(CompilerType &type, Status &error);

  swift::ClangImporter *GetClangImporter();

  CompilerType
  CreateTupleType(const std::vector<TupleElement> &elements) override;
  bool IsTupleType(lldb::opaque_compiler_type_t type) override;
  std::optional<NonTriviallyManagedReferenceKind>
  GetNonTriviallyManagedReferenceKind(
      lldb::opaque_compiler_type_t type) override;

  /// Creates a GenericTypeParamType with the desired depth and index.
  CompilerType
  CreateGenericTypeParamType(unsigned int depth, unsigned int index,
                             swift::Mangle::ManglingFlavor flavor) override;

  CompilerType GetErrorType() override;

  /// Error handling
  /// \{
  bool HasDiagnostics() const;
  bool HasClangImporterErrors() const;

  void AddDiagnostic(lldb::Severity severity, llvm::StringRef message);
  void RaiseFatalError(std::string msg) const { m_fatal_errors = Status(msg); }
  static bool HasFatalErrors(swift::ASTContext *ast_context);
  bool HasFatalErrors() const {
    return m_logged_fatal_error || m_fatal_errors.Fail() ||
           HasFatalErrors(m_ast_context_ap.get());
  }

  /// Return only fatal errors.
  Status GetFatalErrors() const;
  void PrintDiagnostics(DiagnosticManager &diagnostic_manager,
                        uint32_t bufferID = UINT32_MAX, uint32_t first_line = 0,
                        uint32_t last_line = UINT32_MAX) const;

  /// A set of indices into the diagnostic vectors to mark the start
  /// of a transaction.
  struct DiagnosticCursor {
    size_t swift = 0;
    size_t clang = 0;
    size_t lldb = 0;
    size_t m_num_swift_errors = 0;
  };

  /// A lightweight RAII abstraction that sits on top of a diagnostic
  /// consumer that can be used capture diagnostics for one
  /// transaction and restore (most of) the state of the consumer
  /// after its destruction.  Clang errors and LLDB errors are
  /// persistent and intentionally not reset by this.
  class ScopedDiagnostics {
    swift::DiagnosticConsumer &m_consumer;
    const DiagnosticCursor m_cursor;

  public:
    enum class ErrorKind { swift, clang };
    ScopedDiagnostics(swift::DiagnosticConsumer &consumer);
    ~ScopedDiagnostics();
    /// Print all diagnostics that happened during the lifetime of
    /// this object to diagnostic_manager. If none is found, print the
    /// persistent diagnostics form the parent consumer.
    void PrintDiagnostics(DiagnosticManager &diagnostic_manager,
                          uint32_t bufferID = UINT32_MAX,
                          uint32_t first_line = 0,
                          uint32_t last_line = UINT32_MAX) const;
    std::optional<ErrorKind> GetOptionalErrorKind() const;
    bool HasErrors() const;
    /// Return all errors and warnings that happened during the lifetime of this
    /// object as a StringError.
    llvm::Error GetAllErrors() const;
    /// Return all errors and warnings that happened during the lifetime of this
    /// object an ExpressionError.
    llvm::Error GetAsExpressionError(lldb::ExpressionResults result) const;
  };
  std::unique_ptr<ScopedDiagnostics> getScopedDiagnosticConsumer();
  /// \}

  ConstString GetMangledTypeName(swift::TypeBase *);

  swift::IRGenOptions &GetIRGenOptions();
  swift::TBDGenOptions &GetTBDGenOptions();

  void ClearModuleDependentCaches() override;
  void LogConfiguration(bool repl = false, bool playground = false);
  bool HasTarget();
  bool HasExplicitModules() const { return m_has_explicit_modules; }
  bool CheckProcessChanged();

  // FIXME: this should be removed once we figure out who should really own the
  // DebuggerClient's that we are sticking into the Swift Modules.
  void AddDebuggerClient(swift::DebuggerClient *debugger_client);

  typedef llvm::StringMap<const swift::ModuleDecl &> SwiftModuleMap;

  const SwiftModuleMap &GetModuleCache() { return m_swift_module_cache; }

  const swift::irgen::TypeInfo *
  GetSwiftTypeInfo(lldb::opaque_compiler_type_t type);

  const swift::irgen::FixedTypeInfo *
  GetSwiftFixedTypeInfo(lldb::opaque_compiler_type_t type);

  bool IsFixedSize(CompilerType compiler_type);

  plugin::dwarf::DWARFASTParser *GetDWARFParser() override;

  // CompilerDecl functions
  ConstString DeclGetName(void *opaque_decl) override {
    return ConstString("");
  }

  // CompilerDeclContext functions

  std::vector<CompilerDecl>
  DeclContextFindDeclByName(void *opaque_decl_ctx, ConstString name,
                            const bool ignore_imported_decls) override {
    return {};
  }

  bool DeclContextIsContainedInLookup(void *opaque_decl_ctx,
                                      void *other_opaque_decl_ctx) override {
    if (opaque_decl_ctx == other_opaque_decl_ctx)
      return true;
    return false;
  }

  // Tests

#ifndef NDEBUG
  bool Verify(lldb::opaque_compiler_type_t type) override;
#endif

  bool IsArrayType(lldb::opaque_compiler_type_t type,
                   CompilerType *element_type, uint64_t *size,
                   bool *is_incomplete) override;

  bool IsAggregateType(lldb::opaque_compiler_type_t type) override;

  bool IsDefined(lldb::opaque_compiler_type_t type) override;

  bool IsFunctionType(lldb::opaque_compiler_type_t type) override;

  size_t
  GetNumberOfFunctionArguments(lldb::opaque_compiler_type_t type) override;

  CompilerType GetFunctionArgumentAtIndex(lldb::opaque_compiler_type_t type,
                                          const size_t index) override;

  bool IsFunctionPointerType(lldb::opaque_compiler_type_t type) override;

  bool IsPossibleDynamicType(lldb::opaque_compiler_type_t type,
                             CompilerType *target_type, // Can pass NULL
                             bool check_cplusplus, bool check_objc) override;

  bool IsPointerType(lldb::opaque_compiler_type_t type,
                     CompilerType *pointee_type) override;

  bool IsVoidType(lldb::opaque_compiler_type_t type) override;

  static bool IsGenericType(const CompilerType &compiler_type);

  /// Whether this is the Swift error type.
  bool IsErrorType(lldb::opaque_compiler_type_t type) override;

  struct ProtocolInfo {
    uint32_t m_num_protocols;
    uint32_t m_num_payload_words;
    uint32_t m_num_storage_words;
    bool m_is_class_only;
    bool m_is_objc;
    bool m_is_anyobject;
    bool m_is_errortype;

    /// The superclass bound, which can only be non-null when this is
    /// a class-bound existential.
    CompilerType m_superclass;

    /// The member index for the error value within an error
    /// existential.
    static constexpr uint32_t error_instance_index = 0;

    /// Retrieve the index at which the instance type occurs.
    uint32_t GetInstanceTypeIndex() const { return m_num_payload_words; }
  };

  bool GetProtocolTypeInfo(const CompilerType &type,
                           ProtocolInfo &protocol_info);

  static void ApplyWorkingDir(llvm::SmallVectorImpl<char> &clang_argument,
                              llvm::StringRef cur_working_dir);

  // AST related queries

  uint32_t GetPointerByteSize() override;

  // Accessors

  ConstString GetTypeName(lldb::opaque_compiler_type_t type,
                          bool BaseOnly) override;

  ConstString GetDisplayTypeName(lldb::opaque_compiler_type_t type,
                                 const SymbolContext *sc) override;

  ConstString GetMangledTypeName(lldb::opaque_compiler_type_t type) override;

  uint32_t GetTypeInfo(lldb::opaque_compiler_type_t type,
                       CompilerType *pointee_or_element_clang_type) override;

  lldb::TypeClass GetTypeClass(lldb::opaque_compiler_type_t type) override;

  // Creating related types

  /// Return the TypeSystemSwiftTypeRef version of this type.
  CompilerType GetTypeRefType(lldb::opaque_compiler_type_t type);

  CompilerType GetArrayElementType(lldb::opaque_compiler_type_t type,
                                   ExecutionContextScope *exe_scope) override;

  CompilerType GetCanonicalType(lldb::opaque_compiler_type_t type) override;

  CompilerType GetInstanceType(lldb::opaque_compiler_type_t type,
                               ExecutionContextScope *exe_scope) override;

  // Returns -1 if this isn't a function of if the function doesn't have a
  // prototype. Returns a value >override if there is a prototype.
  int GetFunctionArgumentCount(lldb::opaque_compiler_type_t type) override;

  CompilerType GetFunctionArgumentTypeAtIndex(lldb::opaque_compiler_type_t type,
                                              size_t idx) override;

  CompilerType
  GetFunctionReturnType(lldb::opaque_compiler_type_t type) override;

  size_t GetNumMemberFunctions(lldb::opaque_compiler_type_t type) override;

  TypeMemberFunctionImpl
  GetMemberFunctionAtIndex(lldb::opaque_compiler_type_t type,
                           size_t idx) override;

  CompilerType GetPointeeType(lldb::opaque_compiler_type_t type) override;

  CompilerType GetPointerType(lldb::opaque_compiler_type_t type) override;

  // Exploring the type

  llvm::Expected<uint64_t>
  GetBitSize(lldb::opaque_compiler_type_t type,
             ExecutionContextScope *exe_scope) override;

  std::optional<uint64_t>
  GetByteStride(lldb::opaque_compiler_type_t type,
                ExecutionContextScope *exe_scope) override;

  lldb::Encoding GetEncoding(lldb::opaque_compiler_type_t type,
                             uint64_t &count) override;

  llvm::Expected<uint32_t>
  GetNumChildren(lldb::opaque_compiler_type_t type,
                 bool omit_empty_base_classes,
                 const ExecutionContext *exe_ctx) override;

  uint32_t GetNumFields(lldb::opaque_compiler_type_t type,
                        ExecutionContext *exe_ctx = nullptr) override;

  CompilerType GetFieldAtIndex(lldb::opaque_compiler_type_t type, size_t idx,
                               std::string &name, uint64_t *bit_offset_ptr,
                               uint32_t *bitfield_bit_size_ptr,
                               bool *is_bitfield_ptr) override;

  llvm::Expected<CompilerType> GetChildCompilerTypeAtIndex(
      lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx, size_t idx,
      bool transparent_pointers, bool omit_empty_base_classes,
      bool ignore_array_bounds, std::string &child_name,
      uint32_t &child_byte_size, int32_t &child_byte_offset,
      uint32_t &child_bitfield_bit_size, uint32_t &child_bitfield_bit_offset,
      bool &child_is_base_class, bool &child_is_deref_of_parent,
      ValueObject *valobj, uint64_t &language_flags) override;

  // Lookup a child member given a name. This function will match member names
  // only and will descend into "clang_type" children in search for the first
  // member in this class, or any base class that matches "name".
  // TODO: Return all matches for a given name by returning a
  // vector<vector<uint32_t>> so we catch all names that match a given child
  // name, not just the first.
  size_t
  GetIndexOfChildMemberWithName(lldb::opaque_compiler_type_t type,
                                llvm::StringRef name, ExecutionContext *exe_ctx,
                                bool omit_empty_base_classes,
                                std::vector<uint32_t> &child_indexes) override;

  size_t GetNumTemplateArguments(lldb::opaque_compiler_type_t type,
                                 bool expand_pack) override;

  lldb::GenericKind GetGenericArgumentKind(lldb::opaque_compiler_type_t type,
                                           size_t idx);
  CompilerType GetUnboundGenericType(lldb::opaque_compiler_type_t type,
                                     size_t idx);
  CompilerType GetBoundGenericType(lldb::opaque_compiler_type_t type,
                                   size_t idx);
  CompilerType GetGenericArgumentType(CompilerType ct, size_t idx);
  CompilerType GetGenericArgumentType(lldb::opaque_compiler_type_t type,
                                      size_t idx) override;

  CompilerType GetTypeForFormatters(lldb::opaque_compiler_type_t type) override;

  LazyBool ShouldPrintAsOneLiner(lldb::opaque_compiler_type_t type,
                                 ValueObject *valobj) override;

  bool IsMeaninglessWithoutDynamicResolution(
      lldb::opaque_compiler_type_t type) override;

  bool GetSelectedEnumCase(const CompilerType &type, const DataExtractor &data,
                           ConstString *name, bool *has_payload,
                           CompilerType *payload, bool *is_indirect);

  // Dumping types
#ifndef NDEBUG
  /// Convenience LLVM-style dump method for use in the debugger only.
  LLVM_DUMP_METHOD virtual void
  dump(lldb::opaque_compiler_type_t type) const override;
#endif

  bool DumpTypeValue(lldb::opaque_compiler_type_t type, Stream &s,
                     lldb::Format format, const DataExtractor &data,
                     lldb::offset_t data_offset, size_t data_byte_size,
                     uint32_t bitfield_bit_size, uint32_t bitfield_bit_offset,
                     ExecutionContextScope *exe_scope,
                     bool is_base_class) override;

  void DumpTypeDescription(
      lldb::opaque_compiler_type_t type,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull,
      ExecutionContextScope *exe_scope = nullptr) override; // Dump to stdout

  void DumpTypeDescription(
      lldb::opaque_compiler_type_t type, Stream &s,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull,
      ExecutionContextScope *exe_scope = nullptr) override;

  void DumpTypeDescription(
      lldb::opaque_compiler_type_t type, bool print_help_if_available,
      bool print_extensions_if_available,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull,
      ExecutionContextScope *exe_scope = nullptr) override;

  void DumpTypeDescription(
      lldb::opaque_compiler_type_t type, Stream *s,
      bool print_help_if_available, bool print_extensions_if_available,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull,
      ExecutionContextScope *exe_scope = nullptr) override;

  // TODO: Determine if these methods should move to TypeSystemClang.

  bool IsPointerOrReferenceType(lldb::opaque_compiler_type_t type,
                                CompilerType *pointee_type) override;

  std::optional<size_t>
  GetTypeBitAlign(lldb::opaque_compiler_type_t type,
                  ExecutionContextScope *exe_scope) override;

  CompilerType GetBuiltinTypeForEncodingAndBitSize(lldb::Encoding encoding,
                                                   size_t bit_size) override {
    return CompilerType();
  }

  bool IsTypedefType(lldb::opaque_compiler_type_t type) override;

  // If the current object represents a typedef type, get the underlying type
  CompilerType GetTypedefedType(lldb::opaque_compiler_type_t type) override;
  CompilerType GetUnboundType(lldb::opaque_compiler_type_t type);
  std::string GetSuperclassName(const CompilerType &superclass_type);
  CompilerType
  GetFullyUnqualifiedType(lldb::opaque_compiler_type_t type) override;
  
  uint32_t GetNumDirectBaseClasses(lldb::opaque_compiler_type_t type) override;
  CompilerType GetDirectBaseClassAtIndex(lldb::opaque_compiler_type_t type,
                                         size_t idx,
                                         uint32_t *bit_offset_ptr) override;

  bool IsReferenceType(lldb::opaque_compiler_type_t type,
                       CompilerType *pointee_type, bool *is_rvalue) override;

  uint32_t GetNumPointeeChildren(lldb::opaque_compiler_type_t type);

  bool IsImportedType(lldb::opaque_compiler_type_t type,
                      CompilerType *original_type = nullptr) override;

  CompilerType GetReferentType(lldb::opaque_compiler_type_t type) override;
  CompilerType GetStaticSelfType(lldb::opaque_compiler_type_t type) override;

  /// Retrieve/import the modules imported by the compilation
  /// unit. Early-exists with false if there was an import failure.
  bool GetCompileUnitImports(
      const SymbolContext &sc, lldb::ProcessSP process_sp,
      llvm::SmallVectorImpl<swift::AttributedImport<swift::ImportedModule>>
          &modules,
      Status &error);

  /// Perform all the implicit imports for the current frame.
  void PerformCompileUnitImports(const SymbolContext &sc, lldb::ProcessSP process_sp,
                                 Status &error);

  /// Returns the mangling flavor associated with this ASTContext.
  swift::Mangle::ManglingFlavor GetManglingFlavor();

protected:
  bool GetCompileUnitImportsImpl(
      const SymbolContext &sc, lldb::ProcessSP process_sp,
      llvm::SmallVectorImpl<swift::AttributedImport<swift::ImportedModule>>
          *modules,
      Status &error);

  /// This map uses the string value of ConstStrings as the key, and the
  /// TypeBase
  /// * as the value. Since the ConstString strings are uniqued, we can use
  /// pointer equality for string value equality.
  typedef llvm::DenseMap<const char *, swift::TypeBase *>
      SwiftTypeFromMangledNameMap;
  /// Similar logic applies to this "reverse" map
  typedef llvm::DenseMap<swift::TypeBase *, const char *>
      SwiftMangledNameFromTypeMap;

  /// Called by the VALID_OR_RETURN macro to log all errors.
  void LogFatalErrors() const;
  Status GetAllDiagnostics() const;

  llvm::TargetOptions *getTargetOptions();

  swift::ModuleDecl *GetScratchModule();

  swift::Lowering::TypeConverter *GetSILTypes();

  swift::SILModule *GetSILModule();

  swift::MemoryBufferSerializedModuleLoader *GetMemoryBufferModuleLoader();

  swift::ModuleDecl *GetCachedModule(std::string module_name);

  void CacheDemangledType(ConstString mangled_name,
                          swift::TypeBase *found_type);

  void CacheDemangledTypeFailure(ConstString mangled_name);

  bool LoadOneImage(Process &process, FileSpec &link_lib_spec, Status &error);

  bool LoadLibraryUsingPaths(Process &process, llvm::StringRef library_name,
                             std::vector<std::string> &search_paths,
                             bool check_rpath, StreamString &all_dlopen_errors);

  bool TargetHasNoSDK();

  std::vector<lldb::DataBufferSP> &GetASTVectorForModule(const Module *module);

  CompilerType GetAsClangType(ConstString mangled_name);

  /// Retrieve the stored properties for the given nominal type declaration.
  llvm::ArrayRef<swift::VarDecl *>
  GetStoredProperties(swift::NominalTypeDecl *nominal);

  SwiftEnumDescriptor *GetCachedEnumInfo(lldb::opaque_compiler_type_t type);

  friend class CompilerType;

  void ApplyDiagnosticOptions();

  /// Apply a PathMappingList dictionary on all search paths in the
  /// ClangImporterOptions.
  void RemapClangImporterOptions(const PathMappingList &path_map);

  /// Data members.
  /// @{
  std::weak_ptr<TypeSystemSwiftTypeRef> m_typeref_typesystem;
  std::unique_ptr<swift::CompilerInvocation> m_compiler_invocation_ap;
  std::unique_ptr<swift::SourceManager> m_source_manager_up;
  std::unique_ptr<swift::DiagnosticEngine> m_diagnostic_engine_ap;
  // CompilerInvocation, SourceMgr, and DiagEngine must come before
  // the ASTContext, so they get deallocated *after* the ASTContext.
  std::unique_ptr<swift::ASTContext> m_ast_context_ap;
  std::recursive_mutex m_ast_context_mutex;
  std::unique_ptr<llvm::TargetOptions> m_target_options_ap;
  std::unique_ptr<swift::irgen::IRGenerator> m_ir_generator_ap;
  std::unique_ptr<swift::irgen::IRGenModule> m_ir_gen_module_ap;
  llvm::once_flag m_ir_gen_module_once;
  mutable std::once_flag m_swift_import_warning;
  mutable std::once_flag m_swift_warning_streamed;
  std::unique_ptr<swift::DiagnosticConsumer> m_diagnostic_consumer_ap;
  std::unique_ptr<swift::DependencyTracker> m_dependency_tracker;
  swift::ModuleDecl *m_scratch_module = nullptr;
  std::unique_ptr<swift::Lowering::TypeConverter> m_sil_types_ap;
  std::unique_ptr<swift::SILModule> m_sil_module_ap;
  /// Owned by the AST.
  swift::MemoryBufferSerializedModuleLoader *m_memory_buffer_module_loader =
      nullptr;
  swift::ClangImporter *m_clangimporter = nullptr;
  /// Wraps the clang::ASTContext owned by ClangImporter.
  std::shared_ptr<TypeSystemClang> m_clangimporter_typesystem;
  std::unique_ptr<swift::DWARFImporterDelegate> m_dwarfimporter_delegate_up;
  SwiftModuleMap m_swift_module_cache;
  SwiftTypeFromMangledNameMap m_mangled_name_to_type_map;
  SwiftMangledNameFromTypeMap m_type_to_mangled_name_map;
  uint32_t m_pointer_byte_size = 0;
  uint32_t m_pointer_bit_align = 0;
  CompilerType m_void_function_type;
  /// Only if this AST belongs to a target, and an expression has been
  /// evaluated will the target's process pointer be filled in
  lldb_private::Process *m_process = nullptr;
  std::string m_platform_sdk_path;
  /// All previously library loads in LoadLibraryUsingPaths with their
  /// respective result (true = loaded, false = failed to load).
  std::unordered_map<detail::SwiftLibraryLookupRequest, bool>
      library_load_cache;
  /// A cache for GetCompileUnitImports();
  llvm::DenseSet<std::pair<Module *, lldb::user_id_t>> m_cu_imports;

  typedef std::map<Module *, std::vector<lldb::DataBufferSP>> ASTFileDataMap;
  ASTFileDataMap m_ast_file_data_map;
  // FIXME: this vector is needed because the LLDBNameLookup debugger clients
  // are being put into the Module for the SourceFile that we compile the
  // expression into, and so have to live as long as the Module. But it's too
  // late to change swift to get it to take ownership of these DebuggerClients.
  // Since we use the same Target SwiftASTContext for all our compilations,
  // holding them here will keep them alive as long as we need.
  std::vector<std::unique_ptr<swift::DebuggerClient>> m_debugger_clients;
  bool m_initialized_language_options = false;
  bool m_initialized_search_path_options = false;
  bool m_initialized_clang_importer_options = false;
  bool m_has_explicit_modules = false;
  mutable bool m_reported_fatal_error = false;
  mutable bool m_logged_fatal_error = false;

  /// Whether this is a scratch or a module AST context.
  bool m_is_scratch_context = false;

  mutable Status m_fatal_errors;

  typedef ThreadSafeDenseSet<const char *> SwiftMangledNameSet;
  SwiftMangledNameSet m_negative_type_cache;

  /// Record the set of stored properties for each nominal type declaration
  /// for which we've asked this question.
  ///
  /// All of the information in this DenseMap is easily re-constructed
  /// with NominalTypeDecl::getStoredProperties(), but we cache the
  /// result to provide constant-time indexed access.
  llvm::DenseMap<swift::NominalTypeDecl *, std::vector<swift::VarDecl *>>
      m_stored_properties;

  /// @}
};

/// Deprecated.
class SwiftASTContextForModule : public SwiftASTContext {
  // LLVM RTTI support
  static char ID;

public:
  /// LLVM RTTI support
  /// \{
  bool isA(const void *ClassID) const override {
    return ClassID == &ID || SwiftASTContext::isA(ClassID);
  }
  static bool classof(const TypeSystem *ts) { return ts->isA(&ID); }
  /// \}

  SwiftASTContextForModule(std::string description, lldb::ModuleSP module_sp,
                           TypeSystemSwiftTypeRefSP typeref_typesystem)
      : SwiftASTContext(description, module_sp, typeref_typesystem) {}
  virtual ~SwiftASTContextForModule();
};

class SwiftASTContextForExpressions : public SwiftASTContext {
  // LLVM RTTI support
  static char ID;

public:
  /// LLVM RTTI support
  /// \{
  bool isA(const void *ClassID) const override {
    return ClassID == &ID || SwiftASTContext::isA(ClassID);
  }
  static bool classof(const TypeSystem *ts) { return ts->isA(&ID); }
  /// \}

  SwiftASTContextForExpressions(std::string description,
                                lldb::ModuleSP module_sp,
                                TypeSystemSwiftTypeRefSP typeref_typesystem);
  virtual ~SwiftASTContextForExpressions();

  UserExpression *GetUserExpression(llvm::StringRef expr,
                                    llvm::StringRef prefix,
                                    SourceLanguage language,
                                    Expression::ResultType desired_type,
                                    const EvaluateExpressionOptions &options,
                                    ValueObject *ctx_obj) override {
    if (auto ts = m_typeref_typesystem.lock())
      return ts->GetUserExpression(expr, prefix, language, desired_type,
                                   options, ctx_obj);
    return nullptr;
  }

  PersistentExpressionState *GetPersistentExpressionState() override;

  void ModulesDidLoad(ModuleList &module_list);

  typedef llvm::StringMap<swift::AttributedImport<swift::ImportedModule>>
      HandLoadedModuleSet;

  // Insert to the list of hand-loaded modules, (no actual loading occurs).
  void AddHandLoadedModule(
      ConstString module_name,
      swift::AttributedImport<swift::ImportedModule> attributed_import) {
    m_hand_loaded_modules.insert_or_assign(module_name.GetStringRef(),
                                           attributed_import);
  }

  /// Retrieves the modules that need to be implicitly imported in a given
  /// execution scope. This includes the modules imported by both the compile
  /// unit as well as any imports from previous expression evaluations.
  bool GetImplicitImports(
      SymbolContext &sc, lldb::ProcessSP process_sp,
      llvm::SmallVectorImpl<swift::AttributedImport<swift::ImportedModule>>
          &modules,
      Status &error);

  // FIXME: the correct thing to do would be to get the modules by calling
  // CompilerInstance::getImplicitImportInfo, instead of loading these
  // modules manually. However, we currently don't have  access to a
  // CompilerInstance, which is why this function is needed.
  void LoadImplicitModules(lldb::TargetSP target, lldb::ProcessSP process,
                           ExecutionContextScope &exe_scope);
  /// Cache the user's imports from a SourceFile in a given execution scope such
  /// that they are carried over into future expression evaluations.
  bool CacheUserImports(lldb::ProcessSP process_sp,
                        swift::SourceFile &source_file, Status &error);

protected:
  /// These are the names of modules that we have loaded by hand into
  /// the Contexts we make for parsing.
  HandLoadedModuleSet m_hand_loaded_modules;
};

} // namespace lldb_private

#endif // #ifndef liblldb_SwiftASTContext_h_
