//===-- gold-plugin.cpp - Plugin to gold for Link Time Optimization  ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a gold plugin for LLVM. It provides an LLVM implementation of the
// interface described in http://gcc.gnu.org/wiki/whopr/driver .
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/ParallelCG.h"
#include "llvm/Config/config.h" // plugin-api.h requires HAVE_STDINT_H
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Linker/IRMover.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Object/ModuleSummaryIndexObjectFile.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/thread.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/FunctionImport.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/FunctionImportUtils.h"
#include "llvm/Transforms/Utils/GlobalStatus.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <list>
#include <plugin-api.h>
#include <system_error>
#include <utility>
#include <vector>

// FIXME: remove this declaration when we stop maintaining Ubuntu Quantal and
// Precise and Debian Wheezy (binutils 2.23 is required)
#define LDPO_PIE 3

#define LDPT_GET_SYMBOLS_V3 28

using namespace llvm;

static ld_plugin_status discard_message(int level, const char *format, ...) {
  // Die loudly. Recent versions of Gold pass ld_plugin_message as the first
  // callback in the transfer vector. This should never be called.
  abort();
}

static ld_plugin_release_input_file release_input_file = nullptr;
static ld_plugin_get_input_file get_input_file = nullptr;
static ld_plugin_message message = discard_message;

namespace {
struct claimed_file {
  void *handle;
  void *leader_handle;
  std::vector<ld_plugin_symbol> syms;
  off_t filesize;
  std::string name;
};

/// RAII wrapper to manage opening and releasing of a ld_plugin_input_file.
struct PluginInputFile {
  void *Handle;
  std::unique_ptr<ld_plugin_input_file> File;

  PluginInputFile(void *Handle) : Handle(Handle) {
    File = llvm::make_unique<ld_plugin_input_file>();
    if (get_input_file(Handle, File.get()) != LDPS_OK)
      message(LDPL_FATAL, "Failed to get file information");
  }
  ~PluginInputFile() {
    // File would have been reset to nullptr if we moved this object
    // to a new owner.
    if (File)
      if (release_input_file(Handle) != LDPS_OK)
        message(LDPL_FATAL, "Failed to release file information");
  }

  ld_plugin_input_file &file() { return *File; }

  PluginInputFile(PluginInputFile &&RHS) = default;
  PluginInputFile &operator=(PluginInputFile &&RHS) = default;
};

struct ResolutionInfo {
  uint64_t CommonSize = 0;
  unsigned CommonAlign = 0;
  bool IsLinkonceOdr = true;
  GlobalValue::UnnamedAddr UnnamedAddr = GlobalValue::UnnamedAddr::Global;
  GlobalValue::VisibilityTypes Visibility = GlobalValue::DefaultVisibility;
  bool CommonInternal = false;
  bool UseCommon = false;
};

/// Class to own information used by a task or during its cleanup for a
/// ThinLTO backend instantiation.
class ThinLTOTaskInfo {
  /// The output stream the task will codegen into.
  std::unique_ptr<raw_fd_ostream> OS;

  /// The file name corresponding to the output stream, used during cleanup.
  std::string Filename;

  /// Flag indicating whether the output file is a temp file that must be
  /// added to the cleanup list during cleanup.
  bool TempOutFile;

public:
  ThinLTOTaskInfo(std::unique_ptr<raw_fd_ostream> OS, std::string Filename,
                  bool TempOutFile)
      : OS(std::move(OS)), Filename(std::move(Filename)),
        TempOutFile(TempOutFile) {}

  /// Performs task related cleanup activities that must be done
  /// single-threaded (i.e. call backs to gold).
  void cleanup();
};
}

static ld_plugin_add_symbols add_symbols = nullptr;
static ld_plugin_get_symbols get_symbols = nullptr;
static ld_plugin_add_input_file add_input_file = nullptr;
static ld_plugin_set_extra_library_path set_extra_library_path = nullptr;
static ld_plugin_get_view get_view = nullptr;
static Optional<Reloc::Model> RelocationModel;
static std::string output_name = "";
static std::list<claimed_file> Modules;
static DenseMap<int, void *> FDToLeaderHandle;
static StringMap<ResolutionInfo> ResInfo;
static std::vector<std::string> Cleanup;
static llvm::TargetOptions TargetOpts;
static std::string DefaultTriple = sys::getDefaultTargetTriple();

namespace options {
  enum OutputType {
    OT_NORMAL,
    OT_DISABLE,
    OT_BC_ONLY,
    OT_SAVE_TEMPS
  };
  static bool generate_api_file = false;
  static OutputType TheOutputType = OT_NORMAL;
  static unsigned OptLevel = 2;
  // Default parallelism of 0 used to indicate that user did not specify.
  // Actual parallelism default value depends on implementation.
  // Currently, code generation defaults to no parallelism, whereas
  // ThinLTO uses the hardware_concurrency as the default.
  static unsigned Parallelism = 0;
#ifdef NDEBUG
  static bool DisableVerify = true;
#else
  static bool DisableVerify = false;
#endif
  static std::string obj_path;
  static std::string extra_library_path;
  static std::string triple;
  static std::string mcpu;
  // When the thinlto plugin option is specified, only read the function
  // the information from intermediate files and write a combined
  // global index for the ThinLTO backends.
  static bool thinlto = false;
  // If false, all ThinLTO backend compilations through code gen are performed
  // using multiple threads in the gold-plugin, before handing control back to
  // gold. If true, write individual backend index files which reflect
  // the import decisions, and exit afterwards. The assumption is
  // that the build system will launch the backend processes.
  static bool thinlto_index_only = false;
  // If true, when generating individual index files for distributed backends,
  // also generate a "${bitcodefile}.imports" file at the same location for each
  // bitcode file, listing the files it imports from in plain text. This is to
  // support distributed build file staging.
  static bool thinlto_emit_imports_files = false;
  // Option to control where files for a distributed backend (the individual
  // index files and optional imports files) are created.
  // If specified, expects a string of the form "oldprefix:newprefix", and
  // instead of generating these files in the same directory path as the
  // corresponding bitcode file, will use a path formed by replacing the
  // bitcode file's path prefix matching oldprefix with newprefix.
  static std::string thinlto_prefix_replace;
  // Additional options to pass into the code generator.
  // Note: This array will contain all plugin options which are not claimed
  // as plugin exclusive to pass to the code generator.
  // For example, "generate-api-file" and "as"options are for the plugin
  // use only and will not be passed.
  static std::vector<const char *> extra;

  static void process_plugin_option(const char *opt_)
  {
    if (opt_ == nullptr)
      return;
    llvm::StringRef opt = opt_;

    if (opt == "generate-api-file") {
      generate_api_file = true;
    } else if (opt.startswith("mcpu=")) {
      mcpu = opt.substr(strlen("mcpu="));
    } else if (opt.startswith("extra-library-path=")) {
      extra_library_path = opt.substr(strlen("extra_library_path="));
    } else if (opt.startswith("mtriple=")) {
      triple = opt.substr(strlen("mtriple="));
    } else if (opt.startswith("obj-path=")) {
      obj_path = opt.substr(strlen("obj-path="));
    } else if (opt == "emit-llvm") {
      TheOutputType = OT_BC_ONLY;
    } else if (opt == "save-temps") {
      TheOutputType = OT_SAVE_TEMPS;
    } else if (opt == "disable-output") {
      TheOutputType = OT_DISABLE;
    } else if (opt == "thinlto") {
      thinlto = true;
    } else if (opt == "thinlto-index-only") {
      thinlto_index_only = true;
    } else if (opt == "thinlto-emit-imports-files") {
      thinlto_emit_imports_files = true;
    } else if (opt.startswith("thinlto-prefix-replace=")) {
      thinlto_prefix_replace = opt.substr(strlen("thinlto-prefix-replace="));
      if (thinlto_prefix_replace.find(";") == std::string::npos)
        message(LDPL_FATAL, "thinlto-prefix-replace expects 'old;new' format");
    } else if (opt.size() == 2 && opt[0] == 'O') {
      if (opt[1] < '0' || opt[1] > '3')
        message(LDPL_FATAL, "Optimization level must be between 0 and 3");
      OptLevel = opt[1] - '0';
    } else if (opt.startswith("jobs=")) {
      if (StringRef(opt_ + 5).getAsInteger(10, Parallelism))
        message(LDPL_FATAL, "Invalid parallelism level: %s", opt_ + 5);
    } else if (opt == "disable-verify") {
      DisableVerify = true;
    } else {
      // Save this option to pass to the code generator.
      // ParseCommandLineOptions() expects argv[0] to be program name. Lazily
      // add that.
      if (extra.empty())
        extra.push_back("LLVMgold");

      extra.push_back(opt_);
    }
  }
}

static ld_plugin_status claim_file_hook(const ld_plugin_input_file *file,
                                        int *claimed);
static ld_plugin_status all_symbols_read_hook(void);
static ld_plugin_status cleanup_hook(void);

extern "C" ld_plugin_status onload(ld_plugin_tv *tv);
ld_plugin_status onload(ld_plugin_tv *tv) {
  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();
  InitializeAllAsmPrinters();

  // We're given a pointer to the first transfer vector. We read through them
  // until we find one where tv_tag == LDPT_NULL. The REGISTER_* tagged values
  // contain pointers to functions that we need to call to register our own
  // hooks. The others are addresses of functions we can use to call into gold
  // for services.

  bool registeredClaimFile = false;
  bool RegisteredAllSymbolsRead = false;

  for (; tv->tv_tag != LDPT_NULL; ++tv) {
    // Cast tv_tag to int to allow values not in "enum ld_plugin_tag", like, for
    // example, LDPT_GET_SYMBOLS_V3 when building against an older plugin-api.h
    // header.
    switch (static_cast<int>(tv->tv_tag)) {
    case LDPT_OUTPUT_NAME:
      output_name = tv->tv_u.tv_string;
      break;
    case LDPT_LINKER_OUTPUT:
      switch (tv->tv_u.tv_val) {
      case LDPO_REL: // .o
      case LDPO_DYN: // .so
      case LDPO_PIE: // position independent executable
        RelocationModel = Reloc::PIC_;
        break;
      case LDPO_EXEC: // .exe
        RelocationModel = Reloc::Static;
        break;
      default:
        message(LDPL_ERROR, "Unknown output file type %d", tv->tv_u.tv_val);
        return LDPS_ERR;
      }
      break;
    case LDPT_OPTION:
      options::process_plugin_option(tv->tv_u.tv_string);
      break;
    case LDPT_REGISTER_CLAIM_FILE_HOOK: {
      ld_plugin_register_claim_file callback;
      callback = tv->tv_u.tv_register_claim_file;

      if (callback(claim_file_hook) != LDPS_OK)
        return LDPS_ERR;

      registeredClaimFile = true;
    } break;
    case LDPT_REGISTER_ALL_SYMBOLS_READ_HOOK: {
      ld_plugin_register_all_symbols_read callback;
      callback = tv->tv_u.tv_register_all_symbols_read;

      if (callback(all_symbols_read_hook) != LDPS_OK)
        return LDPS_ERR;

      RegisteredAllSymbolsRead = true;
    } break;
    case LDPT_REGISTER_CLEANUP_HOOK: {
      ld_plugin_register_cleanup callback;
      callback = tv->tv_u.tv_register_cleanup;

      if (callback(cleanup_hook) != LDPS_OK)
        return LDPS_ERR;
    } break;
    case LDPT_GET_INPUT_FILE:
      get_input_file = tv->tv_u.tv_get_input_file;
      break;
    case LDPT_RELEASE_INPUT_FILE:
      release_input_file = tv->tv_u.tv_release_input_file;
      break;
    case LDPT_ADD_SYMBOLS:
      add_symbols = tv->tv_u.tv_add_symbols;
      break;
    case LDPT_GET_SYMBOLS_V2:
      // Do not override get_symbols_v3 with get_symbols_v2.
      if (!get_symbols)
        get_symbols = tv->tv_u.tv_get_symbols;
      break;
    case LDPT_GET_SYMBOLS_V3:
      get_symbols = tv->tv_u.tv_get_symbols;
      break;
    case LDPT_ADD_INPUT_FILE:
      add_input_file = tv->tv_u.tv_add_input_file;
      break;
    case LDPT_SET_EXTRA_LIBRARY_PATH:
      set_extra_library_path = tv->tv_u.tv_set_extra_library_path;
      break;
    case LDPT_GET_VIEW:
      get_view = tv->tv_u.tv_get_view;
      break;
    case LDPT_MESSAGE:
      message = tv->tv_u.tv_message;
      break;
    default:
      break;
    }
  }

  if (!registeredClaimFile) {
    message(LDPL_ERROR, "register_claim_file not passed to LLVMgold.");
    return LDPS_ERR;
  }
  if (!add_symbols) {
    message(LDPL_ERROR, "add_symbols not passed to LLVMgold.");
    return LDPS_ERR;
  }

  if (!RegisteredAllSymbolsRead)
    return LDPS_OK;

  if (!get_input_file) {
    message(LDPL_ERROR, "get_input_file not passed to LLVMgold.");
    return LDPS_ERR;
  }
  if (!release_input_file) {
    message(LDPL_ERROR, "release_input_file not passed to LLVMgold.");
    return LDPS_ERR;
  }

  return LDPS_OK;
}

static const GlobalObject *getBaseObject(const GlobalValue &GV) {
  if (auto *GA = dyn_cast<GlobalAlias>(&GV))
    return GA->getBaseObject();
  return cast<GlobalObject>(&GV);
}

static bool shouldSkip(uint32_t Symflags) {
  if (!(Symflags & object::BasicSymbolRef::SF_Global))
    return true;
  if (Symflags & object::BasicSymbolRef::SF_FormatSpecific)
    return true;
  return false;
}

static void diagnosticHandler(const DiagnosticInfo &DI) {
  if (const auto *BDI = dyn_cast<BitcodeDiagnosticInfo>(&DI)) {
    std::error_code EC = BDI->getError();
    if (EC == BitcodeError::InvalidBitcodeSignature)
      return;
  }

  std::string ErrStorage;
  {
    raw_string_ostream OS(ErrStorage);
    DiagnosticPrinterRawOStream DP(OS);
    DI.print(DP);
  }
  ld_plugin_level Level;
  switch (DI.getSeverity()) {
  case DS_Error:
    message(LDPL_FATAL, "LLVM gold plugin has failed to create LTO module: %s",
            ErrStorage.c_str());
  case DS_Warning:
    Level = LDPL_WARNING;
    break;
  case DS_Note:
  case DS_Remark:
    Level = LDPL_INFO;
    break;
  }
  message(Level, "LLVM gold plugin: %s",  ErrStorage.c_str());
}

static void diagnosticHandlerForContext(const DiagnosticInfo &DI,
                                        void *Context) {
  diagnosticHandler(DI);
}

static GlobalValue::VisibilityTypes
getMinVisibility(GlobalValue::VisibilityTypes A,
                 GlobalValue::VisibilityTypes B) {
  if (A == GlobalValue::HiddenVisibility)
    return A;
  if (B == GlobalValue::HiddenVisibility)
    return B;
  if (A == GlobalValue::ProtectedVisibility)
    return A;
  return B;
}

/// Called by gold to see whether this file is one that our plugin can handle.
/// We'll try to open it and register all the symbols with add_symbol if
/// possible.
static ld_plugin_status claim_file_hook(const ld_plugin_input_file *file,
                                        int *claimed) {
  LLVMContext Context;
  MemoryBufferRef BufferRef;
  std::unique_ptr<MemoryBuffer> Buffer;
  if (get_view) {
    const void *view;
    if (get_view(file->handle, &view) != LDPS_OK) {
      message(LDPL_ERROR, "Failed to get a view of %s", file->name);
      return LDPS_ERR;
    }
    BufferRef =
        MemoryBufferRef(StringRef((const char *)view, file->filesize), "");
  } else {
    int64_t offset = 0;
    // Gold has found what might be IR part-way inside of a file, such as
    // an .a archive.
    if (file->offset) {
      offset = file->offset;
    }
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
        MemoryBuffer::getOpenFileSlice(file->fd, file->name, file->filesize,
                                       offset);
    if (std::error_code EC = BufferOrErr.getError()) {
      message(LDPL_ERROR, EC.message().c_str());
      return LDPS_ERR;
    }
    Buffer = std::move(BufferOrErr.get());
    BufferRef = Buffer->getMemBufferRef();
  }

  Context.setDiagnosticHandler(diagnosticHandlerForContext);
  ErrorOr<std::unique_ptr<object::IRObjectFile>> ObjOrErr =
      object::IRObjectFile::create(BufferRef, Context);
  std::error_code EC = ObjOrErr.getError();
  if (EC == object::object_error::invalid_file_type ||
      EC == object::object_error::bitcode_section_not_found)
    return LDPS_OK;

  *claimed = 1;

  if (EC) {
    message(LDPL_ERROR, "LLVM gold plugin has failed to create LTO module: %s",
            EC.message().c_str());
    return LDPS_ERR;
  }
  std::unique_ptr<object::IRObjectFile> Obj = std::move(*ObjOrErr);

  Modules.resize(Modules.size() + 1);
  claimed_file &cf = Modules.back();

  cf.handle = file->handle;
  // Keep track of the first handle for each file descriptor, since there are
  // multiple in the case of an archive. This is used later in the case of
  // ThinLTO parallel backends to ensure that each file is only opened and
  // released once.
  auto LeaderHandle =
      FDToLeaderHandle.insert(std::make_pair(file->fd, file->handle)).first;
  cf.leader_handle = LeaderHandle->second;
  // Save the filesize since for parallel ThinLTO backends we can only
  // invoke get_input_file once per archive (only for the leader handle).
  cf.filesize = file->filesize;
  // In the case of an archive library, all but the first member must have a
  // non-zero offset, which we can append to the file name to obtain a
  // unique name.
  cf.name = file->name;
  if (file->offset)
    cf.name += ".llvm." + std::to_string(file->offset) + "." +
               sys::path::filename(Obj->getModule().getSourceFileName()).str();

  // If we are doing ThinLTO compilation, don't need to process the symbols.
  // Later we simply build a combined index file after all files are claimed.
  if (options::thinlto && options::thinlto_index_only)
    return LDPS_OK;

  for (auto &Sym : Obj->symbols()) {
    uint32_t Symflags = Sym.getFlags();
    if (shouldSkip(Symflags))
      continue;

    cf.syms.push_back(ld_plugin_symbol());
    ld_plugin_symbol &sym = cf.syms.back();
    sym.version = nullptr;

    SmallString<64> Name;
    {
      raw_svector_ostream OS(Name);
      Sym.printName(OS);
    }
    sym.name = strdup(Name.c_str());

    const GlobalValue *GV = Obj->getSymbolGV(Sym.getRawDataRefImpl());

    ResolutionInfo &Res = ResInfo[sym.name];

    sym.visibility = LDPV_DEFAULT;
    if (GV) {
      Res.UnnamedAddr =
          GlobalValue::getMinUnnamedAddr(Res.UnnamedAddr, GV->getUnnamedAddr());
      Res.IsLinkonceOdr &= GV->hasLinkOnceLinkage();
      Res.Visibility = getMinVisibility(Res.Visibility, GV->getVisibility());
      switch (GV->getVisibility()) {
      case GlobalValue::DefaultVisibility:
        break;
      case GlobalValue::HiddenVisibility:
        sym.visibility = LDPV_HIDDEN;
        break;
      case GlobalValue::ProtectedVisibility:
        sym.visibility = LDPV_PROTECTED;
        break;
      }
    }

    if (Symflags & object::BasicSymbolRef::SF_Undefined) {
      sym.def = LDPK_UNDEF;
      if (GV && GV->hasExternalWeakLinkage())
        sym.def = LDPK_WEAKUNDEF;
    } else {
      sym.def = LDPK_DEF;
      if (GV) {
        assert(!GV->hasExternalWeakLinkage() &&
               !GV->hasAvailableExternallyLinkage() && "Not a declaration!");
        if (GV->hasCommonLinkage())
          sym.def = LDPK_COMMON;
        else if (GV->isWeakForLinker())
          sym.def = LDPK_WEAKDEF;
      }
    }

    sym.size = 0;
    sym.comdat_key = nullptr;
    if (GV) {
      const GlobalObject *Base = getBaseObject(*GV);
      if (!Base)
        message(LDPL_FATAL, "Unable to determine comdat of alias!");
      const Comdat *C = Base->getComdat();
      if (C)
        sym.comdat_key = strdup(C->getName().str().c_str());
    }

    sym.resolution = LDPR_UNKNOWN;
  }

  if (!cf.syms.empty()) {
    if (add_symbols(cf.handle, cf.syms.size(), cf.syms.data()) != LDPS_OK) {
      message(LDPL_ERROR, "Unable to add symbols!");
      return LDPS_ERR;
    }
  }

  return LDPS_OK;
}

static void internalize(GlobalValue &GV) {
  if (GV.isDeclarationForLinker())
    return; // We get here if there is a matching asm definition.
  if (!GV.hasLocalLinkage())
    GV.setLinkage(GlobalValue::InternalLinkage);
}

static const char *getResolutionName(ld_plugin_symbol_resolution R) {
  switch (R) {
  case LDPR_UNKNOWN:
    return "UNKNOWN";
  case LDPR_UNDEF:
    return "UNDEF";
  case LDPR_PREVAILING_DEF:
    return "PREVAILING_DEF";
  case LDPR_PREVAILING_DEF_IRONLY:
    return "PREVAILING_DEF_IRONLY";
  case LDPR_PREEMPTED_REG:
    return "PREEMPTED_REG";
  case LDPR_PREEMPTED_IR:
    return "PREEMPTED_IR";
  case LDPR_RESOLVED_IR:
    return "RESOLVED_IR";
  case LDPR_RESOLVED_EXEC:
    return "RESOLVED_EXEC";
  case LDPR_RESOLVED_DYN:
    return "RESOLVED_DYN";
  case LDPR_PREVAILING_DEF_IRONLY_EXP:
    return "PREVAILING_DEF_IRONLY_EXP";
  }
  llvm_unreachable("Unknown resolution");
}

static void freeSymName(ld_plugin_symbol &Sym) {
  free(Sym.name);
  free(Sym.comdat_key);
  Sym.name = nullptr;
  Sym.comdat_key = nullptr;
}

/// Helper to get a file's symbols and a view into it via gold callbacks.
static const void *getSymbolsAndView(claimed_file &F) {
  ld_plugin_status status = get_symbols(F.handle, F.syms.size(), F.syms.data());
  if (status == LDPS_NO_SYMS)
    return nullptr;

  if (status != LDPS_OK)
    message(LDPL_FATAL, "Failed to get symbol information");

  const void *View;
  if (get_view(F.handle, &View) != LDPS_OK)
    message(LDPL_FATAL, "Failed to get a view of file");

  return View;
}

static std::unique_ptr<ModuleSummaryIndex>
getModuleSummaryIndexForFile(claimed_file &F) {
  const void *View = getSymbolsAndView(F);
  if (!View)
    return nullptr;

  MemoryBufferRef BufferRef(StringRef((const char *)View, F.filesize), F.name);

  // Don't bother trying to build an index if there is no summary information
  // in this bitcode file.
  if (!object::ModuleSummaryIndexObjectFile::hasGlobalValueSummaryInMemBuffer(
          BufferRef, diagnosticHandler))
    return std::unique_ptr<ModuleSummaryIndex>(nullptr);

  ErrorOr<std::unique_ptr<object::ModuleSummaryIndexObjectFile>> ObjOrErr =
      object::ModuleSummaryIndexObjectFile::create(BufferRef,
                                                   diagnosticHandler);

  if (std::error_code EC = ObjOrErr.getError())
    message(LDPL_FATAL,
            "Could not read module summary index bitcode from file : %s",
            EC.message().c_str());

  object::ModuleSummaryIndexObjectFile &Obj = **ObjOrErr;

  return Obj.takeIndex();
}

static std::unique_ptr<Module>
getModuleForFile(LLVMContext &Context, claimed_file &F, const void *View,
                 StringRef Name, raw_fd_ostream *ApiFile,
                 StringSet<> &Internalize, std::vector<GlobalValue *> &Keep,
                 StringMap<unsigned> &Realign) {
  MemoryBufferRef BufferRef(StringRef((const char *)View, F.filesize), Name);
  ErrorOr<std::unique_ptr<object::IRObjectFile>> ObjOrErr =
      object::IRObjectFile::create(BufferRef, Context);

  if (std::error_code EC = ObjOrErr.getError())
    message(LDPL_FATAL, "Could not read bitcode from file : %s",
            EC.message().c_str());

  object::IRObjectFile &Obj = **ObjOrErr;

  Module &M = Obj.getModule();

  M.materializeMetadata();
  UpgradeDebugInfo(M);

  SmallPtrSet<GlobalValue *, 8> Used;
  collectUsedGlobalVariables(M, Used, /*CompilerUsed*/ false);

  unsigned SymNum = 0;
  for (auto &ObjSym : Obj.symbols()) {
    GlobalValue *GV = Obj.getSymbolGV(ObjSym.getRawDataRefImpl());
    if (GV && GV->hasAppendingLinkage())
      Keep.push_back(GV);

    if (shouldSkip(ObjSym.getFlags()))
      continue;
    ld_plugin_symbol &Sym = F.syms[SymNum];
    ++SymNum;

    ld_plugin_symbol_resolution Resolution =
        (ld_plugin_symbol_resolution)Sym.resolution;

    if (options::generate_api_file)
      *ApiFile << Sym.name << ' ' << getResolutionName(Resolution) << '\n';

    if (!GV) {
      freeSymName(Sym);
      continue; // Asm symbol.
    }

    ResolutionInfo &Res = ResInfo[Sym.name];
    if (Resolution == LDPR_PREVAILING_DEF_IRONLY_EXP && !Res.IsLinkonceOdr)
      Resolution = LDPR_PREVAILING_DEF;

    // In ThinLTO mode change all prevailing resolutions to LDPR_PREVAILING_DEF.
    // For ThinLTO the IR files are compiled through the backend independently,
    // so we need to ensure that any prevailing linkonce copy will be emitted
    // into the object file by making it weak. Additionally, we can skip the
    // IRONLY handling for internalization, which isn't performed in ThinLTO
    // mode currently anyway.
    if (options::thinlto && (Resolution == LDPR_PREVAILING_DEF_IRONLY_EXP ||
                             Resolution == LDPR_PREVAILING_DEF_IRONLY))
      Resolution = LDPR_PREVAILING_DEF;

    GV->setUnnamedAddr(Res.UnnamedAddr);
    GV->setVisibility(Res.Visibility);

    // Override gold's resolution for common symbols. We want the largest
    // one to win.
    if (GV->hasCommonLinkage()) {
      if (Resolution == LDPR_PREVAILING_DEF_IRONLY)
        Res.CommonInternal = true;

      if (Resolution == LDPR_PREVAILING_DEF_IRONLY ||
          Resolution == LDPR_PREVAILING_DEF)
        Res.UseCommon = true;

      const DataLayout &DL = GV->getParent()->getDataLayout();
      uint64_t Size = DL.getTypeAllocSize(GV->getType()->getElementType());
      unsigned Align = GV->getAlignment();

      if (Res.UseCommon && Size >= Res.CommonSize) {
        // Take GV.
        if (Res.CommonInternal)
          Resolution = LDPR_PREVAILING_DEF_IRONLY;
        else
          Resolution = LDPR_PREVAILING_DEF;
        cast<GlobalVariable>(GV)->setAlignment(
            std::max(Res.CommonAlign, Align));
      } else {
        // Do not take GV, it's smaller than what we already have in the
        // combined module.
        Resolution = LDPR_PREEMPTED_IR;
        if (Align > Res.CommonAlign)
          // Need to raise the alignment though.
          Realign[Sym.name] = Align;
      }

      Res.CommonSize = std::max(Res.CommonSize, Size);
      Res.CommonAlign = std::max(Res.CommonAlign, Align);
    }

    switch (Resolution) {
    case LDPR_UNKNOWN:
      llvm_unreachable("Unexpected resolution");

    case LDPR_RESOLVED_IR:
    case LDPR_RESOLVED_EXEC:
    case LDPR_RESOLVED_DYN:
    case LDPR_PREEMPTED_IR:
    case LDPR_PREEMPTED_REG:
      break;

    case LDPR_UNDEF:
      if (!GV->isDeclarationForLinker())
        assert(GV->hasComdat());
      break;

    case LDPR_PREVAILING_DEF_IRONLY: {
      Keep.push_back(GV);
      // The IR linker has to be able to map this value to a declaration,
      // so we can only internalize after linking.
      if (!Used.count(GV))
        Internalize.insert(GV->getName());
      break;
    }

    case LDPR_PREVAILING_DEF:
      Keep.push_back(GV);
      // There is a non IR use, so we have to force optimizations to keep this.
      switch (GV->getLinkage()) {
      default:
        break;
      case GlobalValue::LinkOnceAnyLinkage:
        GV->setLinkage(GlobalValue::WeakAnyLinkage);
        break;
      case GlobalValue::LinkOnceODRLinkage:
        GV->setLinkage(GlobalValue::WeakODRLinkage);
        break;
      }
      break;

    case LDPR_PREVAILING_DEF_IRONLY_EXP: {
      Keep.push_back(GV);
      if (canBeOmittedFromSymbolTable(GV))
        Internalize.insert(GV->getName());
      break;
    }
    }

    freeSymName(Sym);
  }

  return Obj.takeModule();
}

static void saveBCFile(StringRef Path, Module &M) {
  std::error_code EC;
  raw_fd_ostream OS(Path, EC, sys::fs::OpenFlags::F_None);
  if (EC)
    message(LDPL_FATAL, "Failed to write the output file.");
  WriteBitcodeToFile(&M, OS, /* ShouldPreserveUseListOrder */ false);
}

static void recordFile(std::string Filename, bool TempOutFile) {
  if (add_input_file(Filename.c_str()) != LDPS_OK)
    message(LDPL_FATAL,
            "Unable to add .o file to the link. File left behind in: %s",
            Filename.c_str());
  if (TempOutFile)
    Cleanup.push_back(Filename.c_str());
}

void ThinLTOTaskInfo::cleanup() {
  // Close the output file descriptor before we pass it to gold.
  OS->close();

  recordFile(Filename, TempOutFile);
}

namespace {
/// Class to manage optimization and code generation for a module, possibly
/// in a thread (ThinLTO).
class CodeGen {
  /// The module for which this will generate code.
  std::unique_ptr<llvm::Module> M;

  /// The output stream to generate code into.
  raw_fd_ostream *OS;

  /// The task ID when this was invoked in a thread (ThinLTO).
  int TaskID;

  /// The module summary index for ThinLTO tasks.
  const ModuleSummaryIndex *CombinedIndex;

  /// The target machine for generating code for this module.
  std::unique_ptr<TargetMachine> TM;

  /// Filename to use as base when save-temps is enabled, used to get
  /// a unique and identifiable save-temps output file for each ThinLTO backend.
  std::string SaveTempsFilename;

  /// Map from a module name to the corresponding buffer holding a view of the
  /// bitcode provided via the get_view gold callback.
  StringMap<MemoryBufferRef> *ModuleMap;

  // Functions to import into this module.
  FunctionImporter::ImportMapTy *ImportList;

  // Map of globals defined in this module to their summary.
  std::map<GlobalValue::GUID, GlobalValueSummary *> *DefinedGlobals;

public:
  /// Constructor used by full LTO.
  CodeGen(std::unique_ptr<llvm::Module> M)
      : M(std::move(M)), OS(nullptr), TaskID(-1), CombinedIndex(nullptr),
        ModuleMap(nullptr) {
    initTargetMachine();
  }
  /// Constructor used by ThinLTO.
  CodeGen(std::unique_ptr<llvm::Module> M, raw_fd_ostream *OS, int TaskID,
          const ModuleSummaryIndex *CombinedIndex, std::string Filename,
          StringMap<MemoryBufferRef> *ModuleMap,
          FunctionImporter::ImportMapTy *ImportList,
          std::map<GlobalValue::GUID, GlobalValueSummary *> *DefinedGlobals)
      : M(std::move(M)), OS(OS), TaskID(TaskID), CombinedIndex(CombinedIndex),
        SaveTempsFilename(std::move(Filename)), ModuleMap(ModuleMap),
        ImportList(ImportList), DefinedGlobals(DefinedGlobals) {
    assert(options::thinlto == !!CombinedIndex &&
           "Expected module summary index iff performing ThinLTO");
    initTargetMachine();
  }

  /// Invoke LTO passes and the code generator for the module.
  void runAll();

  /// Invoke the actual code generation to emit Module's object to file.
  void runCodegenPasses();

private:
  const Target *TheTarget;
  std::string TripleStr;
  std::string FeaturesString;
  TargetOptions Options;

  /// Create a target machine for the module. Must be unique for each
  /// module/task.
  void initTargetMachine();

  std::unique_ptr<TargetMachine> createTargetMachine();

  /// Run all LTO passes on the module.
  void runLTOPasses();

  /// Sets up output files necessary to perform optional multi-threaded
  /// split code generation, and invokes the code generation implementation.
  /// If BCFileName is not empty, saves bitcode for module partitions into
  /// {BCFileName}0 .. {BCFileName}N.
  void runSplitCodeGen(const SmallString<128> &BCFilename);
};
}

static SubtargetFeatures getFeatures(Triple &TheTriple) {
  SubtargetFeatures Features;
  Features.getDefaultSubtargetFeatures(TheTriple);
  for (const std::string &A : MAttrs)
    Features.AddFeature(A);
  return Features;
}

static CodeGenOpt::Level getCGOptLevel() {
  switch (options::OptLevel) {
  case 0:
    return CodeGenOpt::None;
  case 1:
    return CodeGenOpt::Less;
  case 2:
    return CodeGenOpt::Default;
  case 3:
    return CodeGenOpt::Aggressive;
  }
  llvm_unreachable("Invalid optimization level");
}

void CodeGen::initTargetMachine() {
  TripleStr = M->getTargetTriple();
  Triple TheTriple(TripleStr);

  std::string ErrMsg;
  TheTarget = TargetRegistry::lookupTarget(TripleStr, ErrMsg);
  if (!TheTarget)
    message(LDPL_FATAL, "Target not found: %s", ErrMsg.c_str());

  SubtargetFeatures Features = getFeatures(TheTriple);
  FeaturesString = Features.getString();
  Options = InitTargetOptionsFromCodeGenFlags();

  // Disable the new X86 relax relocations since gold might not support them.
  // FIXME: Check the gold version or add a new option to enable them.
  Options.RelaxELFRelocations = false;

  TM = createTargetMachine();
}

std::unique_ptr<TargetMachine> CodeGen::createTargetMachine() {
  CodeGenOpt::Level CGOptLevel = getCGOptLevel();

  return std::unique_ptr<TargetMachine>(TheTarget->createTargetMachine(
      TripleStr, options::mcpu, FeaturesString, Options, RelocationModel,
      CodeModel::Default, CGOptLevel));
}

void CodeGen::runLTOPasses() {
  M->setDataLayout(TM->createDataLayout());

  if (CombinedIndex) {
    // Apply summary-based internalization decisions. Skip if there are no
    // defined globals from the summary since not only is it unnecessary, but
    // if this module did not have a summary section the internalizer will
    // assert if it finds any definitions in this module that aren't in the
    // DefinedGlobals set.
    if (!DefinedGlobals->empty())
      thinLTOInternalizeModule(*M, *DefinedGlobals);

    // Create a loader that will parse the bitcode from the buffers
    // in the ModuleMap.
    ModuleLoader Loader(M->getContext(), *ModuleMap);

    // Perform function importing.
    FunctionImporter Importer(*CombinedIndex, Loader);
    Importer.importFunctions(*M, *ImportList);
  }

  legacy::PassManager passes;
  passes.add(createTargetTransformInfoWrapperPass(TM->getTargetIRAnalysis()));

  PassManagerBuilder PMB;
  PMB.LibraryInfo = new TargetLibraryInfoImpl(Triple(TM->getTargetTriple()));
  PMB.Inliner = createFunctionInliningPass();
  // Unconditionally verify input since it is not verified before this
  // point and has unknown origin.
  PMB.VerifyInput = true;
  PMB.VerifyOutput = !options::DisableVerify;
  PMB.LoopVectorize = true;
  PMB.SLPVectorize = true;
  PMB.OptLevel = options::OptLevel;
  if (options::thinlto)
    PMB.populateThinLTOPassManager(passes);
  else
    PMB.populateLTOPassManager(passes);
  passes.run(*M);
}

/// Open a file and return the new file descriptor given a base input
/// file name, a flag indicating whether a temp file should be generated,
/// and an optional task id. The new filename generated is
/// returned in \p NewFilename.
static int openOutputFile(SmallString<128> InFilename, bool TempOutFile,
                          SmallString<128> &NewFilename, int TaskID = -1) {
  int FD;
  if (TempOutFile) {
    std::error_code EC =
        sys::fs::createTemporaryFile("lto-llvm", "o", FD, NewFilename);
    if (EC)
      message(LDPL_FATAL, "Could not create temporary file: %s",
              EC.message().c_str());
  } else {
    NewFilename = InFilename;
    if (TaskID >= 0)
      NewFilename += utostr(TaskID);
    std::error_code EC =
        sys::fs::openFileForWrite(NewFilename, FD, sys::fs::F_None);
    if (EC)
      message(LDPL_FATAL, "Could not open file: %s", EC.message().c_str());
  }
  return FD;
}

void CodeGen::runCodegenPasses() {
  assert(OS && "Output stream must be set before emitting to file");
  legacy::PassManager CodeGenPasses;
  if (TM->addPassesToEmitFile(CodeGenPasses, *OS,
                              TargetMachine::CGFT_ObjectFile))
    report_fatal_error("Failed to setup codegen");
  CodeGenPasses.run(*M);
}

void CodeGen::runSplitCodeGen(const SmallString<128> &BCFilename) {
  SmallString<128> Filename;
  // Note that openOutputFile will append a unique ID for each task
  if (!options::obj_path.empty())
    Filename = options::obj_path;
  else if (options::TheOutputType == options::OT_SAVE_TEMPS)
    Filename = output_name + ".o";

  // Note that the default parallelism is 1 instead of the
  // hardware_concurrency, as there are behavioral differences between
  // parallelism levels (e.g. symbol ordering will be different, and some uses
  // of inline asm currently have issues with parallelism >1).
  unsigned int MaxThreads = options::Parallelism ? options::Parallelism : 1;

  std::vector<SmallString<128>> Filenames(MaxThreads);
  std::vector<SmallString<128>> BCFilenames(MaxThreads);
  bool TempOutFile = Filename.empty();
  {
    // Open a file descriptor for each backend task. This is done in a block
    // so that the output file descriptors are closed before gold opens them.
    std::list<llvm::raw_fd_ostream> OSs;
    std::vector<llvm::raw_pwrite_stream *> OSPtrs(MaxThreads);
    for (unsigned I = 0; I != MaxThreads; ++I) {
      int FD = openOutputFile(Filename, TempOutFile, Filenames[I],
                              // Only append ID if there are multiple tasks.
                              MaxThreads > 1 ? I : -1);
      OSs.emplace_back(FD, true);
      OSPtrs[I] = &OSs.back();
    }

    std::list<llvm::raw_fd_ostream> BCOSs;
    std::vector<llvm::raw_pwrite_stream *> BCOSPtrs;
    if (!BCFilename.empty() && MaxThreads > 1) {
      for (unsigned I = 0; I != MaxThreads; ++I) {
        int FD = openOutputFile(BCFilename, false, BCFilenames[I], I);
        BCOSs.emplace_back(FD, true);
        BCOSPtrs.push_back(&BCOSs.back());
      }
    }

    // Run backend tasks.
    splitCodeGen(std::move(M), OSPtrs, BCOSPtrs,
                 [&]() { return createTargetMachine(); });
  }

  for (auto &Filename : Filenames)
    recordFile(Filename.c_str(), TempOutFile);
}

void CodeGen::runAll() {
  runLTOPasses();

  SmallString<128> OptFilename;
  if (options::TheOutputType == options::OT_SAVE_TEMPS) {
    OptFilename = output_name;
    // If the CodeGen client provided a filename, use it. Always expect
    // a provided filename if we are in a task (i.e. ThinLTO backend).
    assert(!SaveTempsFilename.empty() || TaskID == -1);
    if (!SaveTempsFilename.empty())
      OptFilename = SaveTempsFilename;
    OptFilename += ".opt.bc";
    saveBCFile(OptFilename, *M);
  }

  // If we are already in a thread (i.e. ThinLTO), just perform
  // codegen passes directly.
  if (TaskID >= 0)
    runCodegenPasses();
  // Otherwise attempt split code gen.
  else
    runSplitCodeGen(OptFilename);
}

/// Links the module in \p View from file \p F into the combined module
/// saved in the IRMover \p L.
static void linkInModule(LLVMContext &Context, IRMover &L, claimed_file &F,
                         const void *View, StringRef Name,
                         raw_fd_ostream *ApiFile, StringSet<> &Internalize,
                         bool SetName = false) {
  std::vector<GlobalValue *> Keep;
  StringMap<unsigned> Realign;
  std::unique_ptr<Module> M = getModuleForFile(Context, F, View, Name, ApiFile,
                                               Internalize, Keep, Realign);
  if (!M.get())
    return;
  if (!options::triple.empty())
    M->setTargetTriple(options::triple.c_str());
  else if (M->getTargetTriple().empty()) {
    M->setTargetTriple(DefaultTriple);
  }

  // For ThinLTO we want to propagate the source file name to ensure
  // we can create the correct global identifiers matching those in the
  // original module.
  if (SetName)
    L.getModule().setSourceFileName(M->getSourceFileName());

  if (Error E = L.move(std::move(M), Keep,
                       [](GlobalValue &, IRMover::ValueAdder) {})) {
    handleAllErrors(std::move(E), [&](const llvm::ErrorInfoBase &EIB) {
      message(LDPL_FATAL, "Failed to link module %s: %s", Name.str().c_str(),
              EIB.message().c_str());
    });
  }

  for (const auto &I : Realign) {
    GlobalValue *Dst = L.getModule().getNamedValue(I.first());
    if (!Dst)
      continue;
    cast<GlobalVariable>(Dst)->setAlignment(I.second);
  }
}

/// Perform the ThinLTO backend on a single module, invoking the LTO and codegen
/// pipelines.
static void thinLTOBackendTask(claimed_file &F, const void *View,
                               StringRef Name, raw_fd_ostream *ApiFile,
                               const ModuleSummaryIndex &CombinedIndex,
                               raw_fd_ostream *OS, unsigned TaskID,
                               StringMap<MemoryBufferRef> &ModuleMap,
                               FunctionImporter::ImportMapTy &ImportList,
                               std::map<GlobalValue::GUID, GlobalValueSummary *> &DefinedGlobals) {
  // Need to use a separate context for each task
  LLVMContext Context;
  Context.setDiscardValueNames(options::TheOutputType !=
                               options::OT_SAVE_TEMPS);
  Context.enableDebugTypeODRUniquing(); // Merge debug info types.
  Context.setDiagnosticHandler(diagnosticHandlerForContext, nullptr, true);

  std::unique_ptr<llvm::Module> NewModule(new llvm::Module(Name, Context));
  IRMover L(*NewModule.get());

  StringSet<> Dummy;
  linkInModule(Context, L, F, View, Name, ApiFile, Dummy, true);
  if (renameModuleForThinLTO(*NewModule, CombinedIndex))
    message(LDPL_FATAL, "Failed to rename module for ThinLTO");

  CodeGen codeGen(std::move(NewModule), OS, TaskID, &CombinedIndex, Name,
                  &ModuleMap, &ImportList, &DefinedGlobals);
  codeGen.runAll();
}

/// Launch each module's backend pipeline in a separate task in a thread pool.
static void
thinLTOBackends(raw_fd_ostream *ApiFile,
                const ModuleSummaryIndex &CombinedIndex,
                StringMap<MemoryBufferRef> &ModuleMap,
                StringMap<FunctionImporter::ImportMapTy> &ImportLists,
  StringMap<std::map<GlobalValue::GUID, GlobalValueSummary *>>
      &ModuleToDefinedGVSummaries) {
  unsigned TaskCount = 0;
  std::vector<ThinLTOTaskInfo> Tasks;
  Tasks.reserve(Modules.size());
  unsigned int MaxThreads = options::Parallelism
                                ? options::Parallelism
                                : thread::hardware_concurrency();

  // Create ThreadPool in nested scope so that threads will be joined
  // on destruction.
  {
    ThreadPool ThinLTOThreadPool(MaxThreads);
    for (claimed_file &F : Modules) {
      // Do all the gold callbacks in the main thread, since gold is not thread
      // safe by default.
      const void *View = getSymbolsAndView(F);
      if (!View)
        continue;

      SmallString<128> Filename;
      if (!options::obj_path.empty())
        // Note that openOutputFile will append a unique ID for each task
        Filename = options::obj_path;
      else if (options::TheOutputType == options::OT_SAVE_TEMPS) {
        // Use the input file name so that we get a unique and identifiable
        // output file for each ThinLTO backend task.
        Filename = F.name;
        Filename += ".thinlto.o";
      }
      bool TempOutFile = Filename.empty();

      SmallString<128> NewFilename;
      int FD = openOutputFile(Filename, TempOutFile, NewFilename,
                              // Only append the TaskID if we will use the
                              // non-unique obj_path.
                              !options::obj_path.empty() ? TaskCount : -1);
      TaskCount++;
      std::unique_ptr<raw_fd_ostream> OS =
          llvm::make_unique<raw_fd_ostream>(FD, true);

      // Enqueue the task
      ThinLTOThreadPool.async(thinLTOBackendTask, std::ref(F), View, F.name,
                              ApiFile, std::ref(CombinedIndex), OS.get(),
                              TaskCount, std::ref(ModuleMap),
                              std::ref(ImportLists[F.name]),
                              std::ref(ModuleToDefinedGVSummaries[F.name]));

      // Record the information needed by the task or during its cleanup
      // to a ThinLTOTaskInfo instance. For information needed by the task
      // the unique_ptr ownership is transferred to the ThinLTOTaskInfo.
      Tasks.emplace_back(std::move(OS), NewFilename.c_str(), TempOutFile);
    }
  }

  for (auto &Task : Tasks)
    Task.cleanup();
}

/// Parse the thinlto_prefix_replace option into the \p OldPrefix and
/// \p NewPrefix strings, if it was specified.
static void getThinLTOOldAndNewPrefix(std::string &OldPrefix,
                                      std::string &NewPrefix) {
  StringRef PrefixReplace = options::thinlto_prefix_replace;
  assert(PrefixReplace.empty() || PrefixReplace.find(";") != StringRef::npos);
  std::pair<StringRef, StringRef> Split = PrefixReplace.split(";");
  OldPrefix = Split.first.str();
  NewPrefix = Split.second.str();
}

/// Given the original \p Path to an output file, replace any path
/// prefix matching \p OldPrefix with \p NewPrefix. Also, create the
/// resulting directory if it does not yet exist.
static std::string getThinLTOOutputFile(const std::string &Path,
                                        const std::string &OldPrefix,
                                        const std::string &NewPrefix) {
  if (OldPrefix.empty() && NewPrefix.empty())
    return Path;
  SmallString<128> NewPath(Path);
  llvm::sys::path::replace_path_prefix(NewPath, OldPrefix, NewPrefix);
  StringRef ParentPath = llvm::sys::path::parent_path(NewPath.str());
  if (!ParentPath.empty()) {
    // Make sure the new directory exists, creating it if necessary.
    if (std::error_code EC = llvm::sys::fs::create_directories(ParentPath))
      llvm::errs() << "warning: could not create directory '" << ParentPath
                   << "': " << EC.message() << '\n';
  }
  return NewPath.str();
}

/// Perform ThinLTO link, which creates the combined index file.
/// Also, either launch backend threads or (under thinlto-index-only)
/// emit individual index files for distributed backends and exit.
static ld_plugin_status thinLTOLink(raw_fd_ostream *ApiFile) {
  // Map from a module name to the corresponding buffer holding a view of the
  // bitcode provided via the get_view gold callback.
  StringMap<MemoryBufferRef> ModuleMap;
  // Map to own RAII objects that manage the file opening and releasing
  // interfaces with gold.
  DenseMap<void *, std::unique_ptr<PluginInputFile>> HandleToInputFile;

  // Keep track of internalization candidates as well as those that may not
  // be internalized because they are refereneced from other IR modules.
  DenseSet<GlobalValue::GUID> Internalize;
  DenseSet<GlobalValue::GUID> CrossReferenced;

  ModuleSummaryIndex CombinedIndex;
  uint64_t NextModuleId = 0;
  for (claimed_file &F : Modules) {
    if (!HandleToInputFile.count(F.leader_handle))
      HandleToInputFile.insert(std::make_pair(
          F.leader_handle, llvm::make_unique<PluginInputFile>(F.handle)));
    // Pass this into getModuleSummaryIndexForFile
    const void *View = getSymbolsAndView(F);
    if (!View)
      continue;

    MemoryBufferRef ModuleBuffer(StringRef((const char *)View, F.filesize),
                                 F.name);
    assert(ModuleMap.find(ModuleBuffer.getBufferIdentifier()) ==
               ModuleMap.end() &&
           "Expect unique Buffer Identifier");
    ModuleMap[ModuleBuffer.getBufferIdentifier()] = ModuleBuffer;

    std::unique_ptr<ModuleSummaryIndex> Index = getModuleSummaryIndexForFile(F);

    // Skip files without a module summary.
    if (Index)
      CombinedIndex.mergeFrom(std::move(Index), ++NextModuleId);

    // Look for internalization candidates based on gold's symbol resolution
    // information. Also track symbols referenced from other IR modules.
    for (auto &Sym : F.syms) {
      ld_plugin_symbol_resolution Resolution =
          (ld_plugin_symbol_resolution)Sym.resolution;
      if (Resolution == LDPR_PREVAILING_DEF_IRONLY)
        Internalize.insert(GlobalValue::getGUID(Sym.name));
      if (Resolution == LDPR_RESOLVED_IR || Resolution == LDPR_PREEMPTED_IR)
        CrossReferenced.insert(GlobalValue::getGUID(Sym.name));
    }
  }

  // Remove symbols referenced from other IR modules from the internalization
  // candidate set.
  for (auto &S : CrossReferenced)
    Internalize.erase(S);

  // Collect for each module the list of function it defines (GUID ->
  // Summary).
  StringMap<std::map<GlobalValue::GUID, GlobalValueSummary *>>
      ModuleToDefinedGVSummaries(NextModuleId);
  CombinedIndex.collectDefinedGVSummariesPerModule(ModuleToDefinedGVSummaries);

  StringMap<FunctionImporter::ImportMapTy> ImportLists(NextModuleId);
  StringMap<FunctionImporter::ExportSetTy> ExportLists(NextModuleId);
  ComputeCrossModuleImport(CombinedIndex, ModuleToDefinedGVSummaries,
                           ImportLists, ExportLists);

  // Callback for internalization, to prevent internalization of symbols
  // that were not candidates initially, and those that are being imported
  // (which introduces new cross references).
  auto isExported = [&](StringRef ModuleIdentifier, GlobalValue::GUID GUID) {
    const auto &ExportList = ExportLists.find(ModuleIdentifier);
    return (ExportList != ExportLists.end() &&
            ExportList->second.count(GUID)) ||
           !Internalize.count(GUID);
  };

  // Use global summary-based analysis to identify symbols that can be
  // internalized (because they aren't exported or preserved as per callback).
  // Changes are made in the index, consumed in the ThinLTO backends.
  thinLTOInternalizeAndPromoteInIndex(CombinedIndex, isExported);

  if (options::thinlto_emit_imports_files && !options::thinlto_index_only)
    message(LDPL_WARNING,
            "thinlto-emit-imports-files ignored unless thinlto-index-only");

  if (options::thinlto_index_only) {
    // If the thinlto-prefix-replace option was specified, parse it and
    // extract the old and new prefixes.
    std::string OldPrefix, NewPrefix;
    getThinLTOOldAndNewPrefix(OldPrefix, NewPrefix);

    // For each input bitcode file, generate an individual index that
    // contains summaries only for its own global values, and for any that
    // should be imported.
    for (claimed_file &F : Modules) {
      std::error_code EC;

      std::string NewModulePath =
          getThinLTOOutputFile(F.name, OldPrefix, NewPrefix);
      raw_fd_ostream OS((Twine(NewModulePath) + ".thinlto.bc").str(), EC,
                        sys::fs::OpenFlags::F_None);
      if (EC)
        message(LDPL_FATAL, "Unable to open %s.thinlto.bc for writing: %s",
                NewModulePath.c_str(), EC.message().c_str());
      // Build a map of module to the GUIDs and summary objects that should
      // be written to its index.
      std::map<std::string, GVSummaryMapTy> ModuleToSummariesForIndex;
      gatherImportedSummariesForModule(F.name, ModuleToDefinedGVSummaries,
                                       ImportLists, ModuleToSummariesForIndex);
      WriteIndexToFile(CombinedIndex, OS, &ModuleToSummariesForIndex);

      if (options::thinlto_emit_imports_files) {
        if ((EC = EmitImportsFiles(F.name,
                                   (Twine(NewModulePath) + ".imports").str(),
                                   ImportLists)))
          message(LDPL_FATAL, "Unable to open %s.imports",
                  NewModulePath.c_str(), EC.message().c_str());
      }
    }

    cleanup_hook();
    exit(0);
  }

  // Create OS in nested scope so that it will be closed on destruction.
  {
    std::error_code EC;
    raw_fd_ostream OS(output_name + ".thinlto.bc", EC,
                      sys::fs::OpenFlags::F_None);
    if (EC)
      message(LDPL_FATAL, "Unable to open %s.thinlto.bc for writing: %s",
              output_name.data(), EC.message().c_str());
    WriteIndexToFile(CombinedIndex, OS);
  }

  thinLTOBackends(ApiFile, CombinedIndex, ModuleMap, ImportLists,
                  ModuleToDefinedGVSummaries);
  return LDPS_OK;
}

/// gold informs us that all symbols have been read. At this point, we use
/// get_symbols to see if any of our definitions have been overridden by a
/// native object file. Then, perform optimization and codegen.
static ld_plugin_status allSymbolsReadHook(raw_fd_ostream *ApiFile) {
  if (Modules.empty())
    return LDPS_OK;

  if (unsigned NumOpts = options::extra.size())
    cl::ParseCommandLineOptions(NumOpts, &options::extra[0]);

  if (options::thinlto)
    return thinLTOLink(ApiFile);

  LLVMContext Context;
  Context.setDiscardValueNames(options::TheOutputType !=
                               options::OT_SAVE_TEMPS);
  Context.enableDebugTypeODRUniquing(); // Merge debug info types.
  Context.setDiagnosticHandler(diagnosticHandlerForContext, nullptr, true);

  std::unique_ptr<Module> Combined(new Module("ld-temp.o", Context));
  IRMover L(*Combined);

  StringSet<> Internalize;
  for (claimed_file &F : Modules) {
    // RAII object to manage the file opening and releasing interfaces with
    // gold.
    PluginInputFile InputFile(F.handle);
    const void *View = getSymbolsAndView(F);
    if (!View)
      continue;
    linkInModule(Context, L, F, View, F.name, ApiFile, Internalize);
  }

  for (const auto &Name : Internalize) {
    GlobalValue *GV = Combined->getNamedValue(Name.first());
    if (GV)
      internalize(*GV);
  }

  if (options::TheOutputType == options::OT_DISABLE)
    return LDPS_OK;

  if (options::TheOutputType != options::OT_NORMAL) {
    std::string path;
    if (options::TheOutputType == options::OT_BC_ONLY)
      path = output_name;
    else
      path = output_name + ".bc";
    saveBCFile(path, *Combined);
    if (options::TheOutputType == options::OT_BC_ONLY)
      return LDPS_OK;
  }

  CodeGen codeGen(std::move(Combined));
  codeGen.runAll();

  if (!options::extra_library_path.empty() &&
      set_extra_library_path(options::extra_library_path.c_str()) != LDPS_OK)
    message(LDPL_FATAL, "Unable to set the extra library path.");

  return LDPS_OK;
}

static ld_plugin_status all_symbols_read_hook(void) {
  ld_plugin_status Ret;
  if (!options::generate_api_file) {
    Ret = allSymbolsReadHook(nullptr);
  } else {
    std::error_code EC;
    raw_fd_ostream ApiFile("apifile.txt", EC, sys::fs::F_None);
    if (EC)
      message(LDPL_FATAL, "Unable to open apifile.txt for writing: %s",
              EC.message().c_str());
    Ret = allSymbolsReadHook(&ApiFile);
  }

  llvm_shutdown();

  if (options::TheOutputType == options::OT_BC_ONLY ||
      options::TheOutputType == options::OT_DISABLE) {
    if (options::TheOutputType == options::OT_DISABLE) {
      // Remove the output file here since ld.bfd creates the output file
      // early.
      std::error_code EC = sys::fs::remove(output_name);
      if (EC)
        message(LDPL_ERROR, "Failed to delete '%s': %s", output_name.c_str(),
                EC.message().c_str());
    }
    exit(0);
  }

  return Ret;
}

static ld_plugin_status cleanup_hook(void) {
  for (std::string &Name : Cleanup) {
    std::error_code EC = sys::fs::remove(Name);
    if (EC)
      message(LDPL_ERROR, "Failed to delete '%s': %s", Name.c_str(),
              EC.message().c_str());
  }

  return LDPS_OK;
}
