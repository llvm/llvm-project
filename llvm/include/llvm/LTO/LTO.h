//===-LTO.h - LLVM Link Time Optimizer ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares functions and classes used to support LTO. It is intended
// to be used both by LTO classes as well as by clients (gold-plugin) that
// don't utilize the LTO code generator interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LTO_LTO_H
#define LLVM_LTO_LTO_H

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/LTO/Config.h"
#include "llvm/Linker/IRMover.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Support/thread.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/IPO/FunctionImport.h"

namespace llvm {

class Error;
class LLVMContext;
class MemoryBufferRef;
class Module;
class Target;
class raw_pwrite_stream;

/// Helper to load a module from bitcode.
std::unique_ptr<Module> loadModuleFromBuffer(const MemoryBufferRef &Buffer,
                                             LLVMContext &Context, bool Lazy);

/// Provide a "loader" for the FunctionImporter to access function from other
/// modules.
class ModuleLoader {
  /// The context that will be used for importing.
  LLVMContext &Context;

  /// Map from Module identifier to MemoryBuffer. Used by clients like the
  /// FunctionImported to request loading a Module.
  StringMap<MemoryBufferRef> &ModuleMap;

public:
  ModuleLoader(LLVMContext &Context, StringMap<MemoryBufferRef> &ModuleMap)
      : Context(Context), ModuleMap(ModuleMap) {}

  /// Load a module on demand.
  std::unique_ptr<Module> operator()(StringRef Identifier) {
    return loadModuleFromBuffer(ModuleMap[Identifier], Context, /*Lazy*/ true);
  }
};


/// Resolve Weak and LinkOnce values in the \p Index. Linkage changes recorded
/// in the index and the ThinLTO backends must apply the changes to the Module
/// via thinLTOResolveWeakForLinkerModule.
///
/// This is done for correctness (if value exported, ensure we always
/// emit a copy), and compile-time optimization (allow drop of duplicates).
void thinLTOResolveWeakForLinkerInIndex(
    ModuleSummaryIndex &Index,
    function_ref<bool(GlobalValue::GUID, const GlobalValueSummary *)>
        isPrevailing,
    function_ref<void(StringRef, GlobalValue::GUID, GlobalValue::LinkageTypes)>
        recordNewLinkage);

/// Update the linkages in the given \p Index to mark exported values
/// as external and non-exported values as internal. The ThinLTO backends
/// must apply the changes to the Module via thinLTOInternalizeModule.
void thinLTOInternalizeAndPromoteInIndex(
    ModuleSummaryIndex &Index,
    function_ref<bool(StringRef, GlobalValue::GUID)> isExported);

namespace lto {

class LTO;
struct SymbolResolution;
class ThinBackendProc;

/// An input file. This is a wrapper for IRObjectFile that exposes only the
/// information that an LTO client should need in order to do symbol resolution.
class InputFile {
  // FIXME: Remove LTO class friendship once we have bitcode symbol tables.
  friend LTO;
  InputFile() = default;

  // FIXME: Remove the LLVMContext once we have bitcode symbol tables.
  LLVMContext Ctx;
  std::unique_ptr<object::IRObjectFile> Obj;

public:
  /// Create an InputFile.
  static Expected<std::unique_ptr<InputFile>> create(MemoryBufferRef Object);

  class symbol_iterator;

  /// This is a wrapper for object::basic_symbol_iterator that exposes only the
  /// information that an LTO client should need in order to do symbol
  /// resolution.
  ///
  /// This object is ephemeral; it is only valid as long as an iterator obtained
  /// from symbols() refers to it.
  class Symbol {
    friend symbol_iterator;
    friend LTO;

    object::basic_symbol_iterator I;
    const GlobalValue *GV;
    uint32_t Flags;
    SmallString<64> Name;

    bool shouldSkip() {
      return !(Flags & object::BasicSymbolRef::SF_Global) ||
             (Flags & object::BasicSymbolRef::SF_FormatSpecific);
    }

    void skip() {
      const object::SymbolicFile *Obj = I->getObject();
      auto E = Obj->symbol_end();
      while (I != E) {
        Flags = I->getFlags();
        if (!shouldSkip())
          break;
        ++I;
      }
      if (I == E)
        return;

      Name.clear();
      {
        raw_svector_ostream OS(Name);
        I->printName(OS);
      }
      GV = cast<object::IRObjectFile>(Obj)->getSymbolGV(I->getRawDataRefImpl());
    }

  public:
    Symbol(object::basic_symbol_iterator I) : I(I) { skip(); }

    StringRef getName() const { return Name; }
    StringRef getIRName() const {
      if (GV)
        return GV->getName();
      return StringRef();
    }
    uint32_t getFlags() const { return Flags; }
    GlobalValue::VisibilityTypes getVisibility() const {
      if (GV)
        return GV->getVisibility();
      return GlobalValue::DefaultVisibility;
    }
    bool canBeOmittedFromSymbolTable() const {
      return GV && llvm::canBeOmittedFromSymbolTable(GV);
    }
    Expected<const Comdat *> getComdat() const {
      const GlobalObject *GO;
      if (auto *GA = dyn_cast<GlobalAlias>(GV)) {
        GO = GA->getBaseObject();
        if (!GO)
          return make_error<StringError>("Unable to determine comdat of alias!",
                                         inconvertibleErrorCode());
      } else {
        GO = cast<GlobalObject>(GV);
      }
      if (GV)
        return GV->getComdat();
      return nullptr;
    }
    uint64_t getCommonSize() const {
      assert(Flags & object::BasicSymbolRef::SF_Common);
      if (!GV)
        return 0;
      return GV->getParent()->getDataLayout().getTypeAllocSize(
          GV->getType()->getElementType());
    }
    unsigned getCommonAlignment() const {
      assert(Flags & object::BasicSymbolRef::SF_Common);
      if (!GV)
        return 0;
      return GV->getAlignment();
    }
  };

  class symbol_iterator {
    Symbol Sym;

  public:
    symbol_iterator(object::basic_symbol_iterator I) : Sym(I) {}

    symbol_iterator &operator++() {
      ++Sym.I;
      Sym.skip();
      return *this;
    }

    symbol_iterator operator++(int) {
      symbol_iterator I = *this;
      ++*this;
      return I;
    }

    const Symbol &operator*() const { return Sym; }
    const Symbol *operator->() const { return &Sym; }

    bool operator!=(const symbol_iterator &Other) const {
      return Sym.I != Other.Sym.I;
    }
  };

  /// A range over the symbols in this InputFile.
  iterator_range<symbol_iterator> symbols() {
    return llvm::make_range(symbol_iterator(Obj->symbol_begin()),
                            symbol_iterator(Obj->symbol_end()));
  }

  StringRef getSourceFileName() const {
    return Obj->getModule().getSourceFileName();
  }
};

/// A ThinBackend defines what happens after the thin-link phase during ThinLTO.
/// The details of this type definition aren't important; clients can only
/// create a ThinBackend using one of the create*ThinBackend() functions below.
typedef std::function<std::unique_ptr<ThinBackendProc>(
    Config &C, ModuleSummaryIndex &CombinedIndex,
    StringMap<GVSummaryMapTy> &ModuleToDefinedGVSummaries,
    AddStreamFn AddStream)>
    ThinBackend;

/// This ThinBackend runs the individual backend jobs in-process.
ThinBackend createInProcessThinBackend(unsigned ParallelismLevel);

/// This ThinBackend writes individual module indexes to files, instead of
/// running the individual backend jobs. This backend is for distributed builds
/// where separate processes will invoke the real backends.
///
/// To find the path to write the index to, the backend checks if the path has a
/// prefix of OldPrefix; if so, it replaces that prefix with NewPrefix. It then
/// appends ".thinlto.bc" and writes the index to that path. If
/// ShouldEmitImportsFiles is true it also writes a list of imported files to a
/// similar path with ".imports" appended instead.
ThinBackend createWriteIndexesThinBackend(std::string OldPrefix,
                                          std::string NewPrefix,
                                          bool ShouldEmitImportsFiles,
                                          std::string LinkedObjectsFile);

/// This class implements a resolution-based interface to LLVM's LTO
/// functionality. It supports regular LTO, parallel LTO code generation and
/// ThinLTO. You can use it from a linker in the following way:
/// - Set hooks and code generation options (see lto::Config struct defined in
///   Config.h), and use the lto::Config object to create an lto::LTO object.
/// - Create lto::InputFile objects using lto::InputFile::create(), then use
///   the symbols() function to enumerate its symbols and compute a resolution
///   for each symbol (see SymbolResolution below).
/// - After the linker has visited each input file (and each regular object
///   file) and computed a resolution for each symbol, take each lto::InputFile
///   and pass it and an array of symbol resolutions to the add() function.
/// - Call the getMaxTasks() function to get an upper bound on the number of
///   native object files that LTO may add to the link.
/// - Call the run() function. This function will use the supplied AddStream
///   function to add up to getMaxTasks() native object files to the link.
class LTO {
  friend InputFile;

public:
  /// Create an LTO object. A default constructed LTO object has a reasonable
  /// production configuration, but you can customize it by passing arguments to
  /// this constructor.
  /// FIXME: We do currently require the DiagHandler field to be set in Conf.
  /// Until that is fixed, a Config argument is required.
  LTO(Config Conf, ThinBackend Backend = nullptr,
      unsigned ParallelCodeGenParallelismLevel = 1);

  /// Add an input file to the LTO link, using the provided symbol resolutions.
  /// The symbol resolutions must appear in the enumeration order given by
  /// InputFile::symbols().
  Error add(std::unique_ptr<InputFile> Obj, ArrayRef<SymbolResolution> Res);

  /// Returns an upper bound on the number of tasks that the client may expect.
  /// This may only be called after all IR object files have been added. For a
  /// full description of tasks see LTOBackend.h.
  unsigned getMaxTasks() const;

  /// Runs the LTO pipeline. This function calls the supplied AddStream function
  /// to add native object files to the link.
  Error run(AddStreamFn AddStream);

private:
  Config Conf;

  struct RegularLTOState {
    RegularLTOState(unsigned ParallelCodeGenParallelismLevel, Config &Conf);

    unsigned ParallelCodeGenParallelismLevel;
    LTOLLVMContext Ctx;
    bool HasModule = false;
    std::unique_ptr<Module> CombinedModule;
    IRMover Mover;
  } RegularLTO;

  struct ThinLTOState {
    ThinLTOState(ThinBackend Backend);

    ThinBackend Backend;
    ModuleSummaryIndex CombinedIndex;
    MapVector<StringRef, MemoryBufferRef> ModuleMap;
    DenseMap<GlobalValue::GUID, StringRef> PrevailingModuleForGUID;
  } ThinLTO;

  // The global resolution for a particular (mangled) symbol name. This is in
  // particular necessary to track whether each symbol can be internalized.
  // Because any input file may introduce a new cross-partition reference, we
  // cannot make any final internalization decisions until all input files have
  // been added and the client has called run(). During run() we apply
  // internalization decisions either directly to the module (for regular LTO)
  // or to the combined index (for ThinLTO).
  struct GlobalResolution {
    /// The unmangled name of the global.
    std::string IRName;

    bool UnnamedAddr = true;

    /// This field keeps track of the partition number of this global. The
    /// regular LTO object is partition 0, while each ThinLTO object has its own
    /// partition number from 1 onwards.
    ///
    /// Any global that is defined or used by more than one partition, or that
    /// is referenced externally, may not be internalized.
    ///
    /// Partitions generally have a one-to-one correspondence with tasks, except
    /// that we use partition 0 for all parallel LTO code generation partitions.
    /// Any partitioning of the combined LTO object is done internally by the
    /// LTO backend.
    unsigned Partition = Unknown;

    /// Special partition numbers.
    enum : unsigned {
      /// A partition number has not yet been assigned to this global.
      Unknown = -1u,

      /// This global is either used by more than one partition or has an
      /// external reference, and therefore cannot be internalized.
      External = -2u,
    };
  };

  // Global mapping from mangled symbol names to resolutions.
  StringMap<GlobalResolution> GlobalResolutions;

  void writeToResolutionFile(InputFile *Input, ArrayRef<SymbolResolution> Res);

  void addSymbolToGlobalRes(object::IRObjectFile *Obj,
                            SmallPtrSet<GlobalValue *, 8> &Used,
                            const InputFile::Symbol &Sym, SymbolResolution Res,
                            unsigned Partition);

  Error addRegularLTO(std::unique_ptr<InputFile> Input,
                      ArrayRef<SymbolResolution> Res);
  Error addThinLTO(std::unique_ptr<InputFile> Input,
                   ArrayRef<SymbolResolution> Res);

  Error runRegularLTO(AddStreamFn AddStream);
  Error runThinLTO(AddStreamFn AddStream);

  mutable bool CalledGetMaxTasks = false;
};

/// The resolution for a symbol. The linker must provide a SymbolResolution for
/// each global symbol based on its internal resolution of that symbol.
struct SymbolResolution {
  SymbolResolution()
      : Prevailing(0), FinalDefinitionInLinkageUnit(0), VisibleToRegularObj(0) {
  }
  /// The linker has chosen this definition of the symbol.
  unsigned Prevailing : 1;

  /// The definition of this symbol is unpreemptable at runtime and is known to
  /// be in this linkage unit.
  unsigned FinalDefinitionInLinkageUnit : 1;

  /// The definition of this symbol is visible outside of the LTO unit.
  unsigned VisibleToRegularObj : 1;
};

} // namespace lto
} // namespace llvm

#endif
