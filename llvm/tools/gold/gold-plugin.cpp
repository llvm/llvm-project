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

#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/Config/config.h" // plugin-api.h requires HAVE_STDINT_H
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <list>
#include <map>
#include <plugin-api.h>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

// FIXME: remove this declaration when we stop maintaining Ubuntu Quantal and
// Precise and Debian Wheezy (binutils 2.23 is required)
#define LDPO_PIE 3

#define LDPT_GET_SYMBOLS_V3 28

using namespace llvm;
using namespace lto;

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
  bool CanOmitFromDynSym = true;
  bool DefaultVisibility = true;
};

struct CommonResolution {
  bool Prevailing = false;
  bool VisibleToRegularObj = false;
  uint64_t Size = 0;
  unsigned Align = 0;
};

}

static ld_plugin_add_symbols add_symbols = nullptr;
static ld_plugin_get_symbols get_symbols = nullptr;
static ld_plugin_add_input_file add_input_file = nullptr;
static ld_plugin_set_extra_library_path set_extra_library_path = nullptr;
static ld_plugin_get_view get_view = nullptr;
static bool IsExecutable = false;
static Optional<Reloc::Model> RelocationModel;
static std::string output_name = "";
static std::list<claimed_file> Modules;
static DenseMap<int, void *> FDToLeaderHandle;
static StringMap<ResolutionInfo> ResInfo;
static std::map<std::string, CommonResolution> Commons;
static std::vector<std::string> Cleanup;
static llvm::TargetOptions TargetOpts;
static size_t MaxTasks;

namespace options {
  enum OutputType {
    OT_NORMAL,
    OT_DISABLE,
    OT_BC_ONLY,
    OT_SAVE_TEMPS
  };
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
  // If non-empty, holds the name of a file in which to write the list of
  // oject files gold selected for inclusion in the link after symbol
  // resolution (i.e. they had selected symbols). This will only be non-empty
  // in the thinlto_index_only case. It is used to identify files, which may
  // have originally been within archive libraries specified via
  // --start-lib/--end-lib pairs, that should be included in the final
  // native link process (since intervening function importing and inlining
  // may change the symbol resolution detected in the final link and which
  // files to include out of --start-lib/--end-lib libraries as a result).
  static std::string thinlto_linked_objects_file;
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
  static std::vector<const char *> extra;

  static void process_plugin_option(const char *opt_)
  {
    if (opt_ == nullptr)
      return;
    llvm::StringRef opt = opt_;

    if (opt.startswith("mcpu=")) {
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
    } else if (opt.startswith("thinlto-index-only=")) {
      thinlto_index_only = true;
      thinlto_linked_objects_file = opt.substr(strlen("thinlto-index-only="));
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
        IsExecutable = false;
        RelocationModel = Reloc::PIC_;
        break;
      case LDPO_PIE: // position independent executable
        IsExecutable = true;
        RelocationModel = Reloc::PIC_;
        break;
      case LDPO_EXEC: // .exe
        IsExecutable = true;
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

static void check(Error E, std::string Msg = "LLVM gold plugin") {
  handleAllErrors(std::move(E), [&](ErrorInfoBase &EIB) {
    message(LDPL_FATAL, "%s: %s", Msg.c_str(), EIB.message().c_str());
    return Error::success();
  });
}

template <typename T> static T check(Expected<T> E) {
  if (E)
    return std::move(*E);
  check(E.takeError());
  return T();
}

/// Called by gold to see whether this file is one that our plugin can handle.
/// We'll try to open it and register all the symbols with add_symbol if
/// possible.
static ld_plugin_status claim_file_hook(const ld_plugin_input_file *file,
                                        int *claimed) {
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

  *claimed = 1;

  Expected<std::unique_ptr<InputFile>> ObjOrErr = InputFile::create(BufferRef);
  if (!ObjOrErr) {
    handleAllErrors(ObjOrErr.takeError(), [&](const ErrorInfoBase &EI) {
      std::error_code EC = EI.convertToErrorCode();
      if (EC == object::object_error::invalid_file_type ||
          EC == object::object_error::bitcode_section_not_found)
        *claimed = 0;
      else
        message(LDPL_ERROR,
                "LLVM gold plugin has failed to create LTO module: %s",
                EI.message().c_str());
    });

    return *claimed ? LDPS_ERR : LDPS_OK;
  }

  std::unique_ptr<InputFile> Obj = std::move(*ObjOrErr);

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
               sys::path::filename(Obj->getSourceFileName()).str();

  for (auto &Sym : Obj->symbols()) {
    uint32_t Symflags = Sym.getFlags();

    cf.syms.push_back(ld_plugin_symbol());
    ld_plugin_symbol &sym = cf.syms.back();
    sym.version = nullptr;
    StringRef Name = Sym.getName();
    sym.name = strdup(Name.str().c_str());

    ResolutionInfo &Res = ResInfo[Name];

    Res.CanOmitFromDynSym &= Sym.canBeOmittedFromSymbolTable();

    sym.visibility = LDPV_DEFAULT;
    GlobalValue::VisibilityTypes Vis = Sym.getVisibility();
    if (Vis != GlobalValue::DefaultVisibility)
      Res.DefaultVisibility = false;
    switch (Vis) {
    case GlobalValue::DefaultVisibility:
      break;
    case GlobalValue::HiddenVisibility:
      sym.visibility = LDPV_HIDDEN;
      break;
    case GlobalValue::ProtectedVisibility:
      sym.visibility = LDPV_PROTECTED;
      break;
    }

    if (Symflags & object::BasicSymbolRef::SF_Undefined) {
      sym.def = LDPK_UNDEF;
      if (Symflags & object::BasicSymbolRef::SF_Weak)
        sym.def = LDPK_WEAKUNDEF;
    } else if (Symflags & object::BasicSymbolRef::SF_Common)
      sym.def = LDPK_COMMON;
    else if (Symflags & object::BasicSymbolRef::SF_Weak)
      sym.def = LDPK_WEAKDEF;
    else
      sym.def = LDPK_DEF;

    sym.size = 0;
    sym.comdat_key = nullptr;
    const Comdat *C = check(Sym.getComdat());
    if (C)
      sym.comdat_key = strdup(C->getName().str().c_str());

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

static void addModule(LTO &Lto, claimed_file &F, const void *View) {
  MemoryBufferRef BufferRef(StringRef((const char *)View, F.filesize), F.name);
  Expected<std::unique_ptr<InputFile>> ObjOrErr = InputFile::create(BufferRef);

  if (!ObjOrErr)
    message(LDPL_FATAL, "Could not read bitcode from file : %s",
            toString(ObjOrErr.takeError()).c_str());

  InputFile &Obj = **ObjOrErr;

  unsigned SymNum = 0;
  std::vector<SymbolResolution> Resols(F.syms.size());
  for (auto &ObjSym : Obj.symbols()) {
    ld_plugin_symbol &Sym = F.syms[SymNum];
    SymbolResolution &R = Resols[SymNum];
    ++SymNum;

    ld_plugin_symbol_resolution Resolution =
        (ld_plugin_symbol_resolution)Sym.resolution;

    ResolutionInfo &Res = ResInfo[Sym.name];

    switch (Resolution) {
    case LDPR_UNKNOWN:
      llvm_unreachable("Unexpected resolution");

    case LDPR_RESOLVED_IR:
    case LDPR_RESOLVED_EXEC:
    case LDPR_RESOLVED_DYN:
    case LDPR_PREEMPTED_IR:
    case LDPR_PREEMPTED_REG:
    case LDPR_UNDEF:
      break;

    case LDPR_PREVAILING_DEF_IRONLY:
      R.Prevailing = true;
      break;

    case LDPR_PREVAILING_DEF:
      R.Prevailing = true;
      R.VisibleToRegularObj = true;
      break;

    case LDPR_PREVAILING_DEF_IRONLY_EXP:
      R.Prevailing = true;
      if (!Res.CanOmitFromDynSym)
        R.VisibleToRegularObj = true;
      break;
    }

    if (Resolution != LDPR_RESOLVED_DYN && Resolution != LDPR_UNDEF &&
        (IsExecutable || !Res.DefaultVisibility))
      R.FinalDefinitionInLinkageUnit = true;

    if (ObjSym.getFlags() & object::BasicSymbolRef::SF_Common) {
      // We ignore gold's resolution for common symbols. A common symbol with
      // the correct size and alignment is added to the module by the pre-opt
      // module hook if any common symbol prevailed.
      CommonResolution &CommonRes = Commons[ObjSym.getIRName()];
      if (R.Prevailing) {
        CommonRes.Prevailing = true;
        CommonRes.VisibleToRegularObj = R.VisibleToRegularObj;
      }
      CommonRes.Size = std::max(CommonRes.Size,
                                static_cast<uint64_t>(ObjSym.getCommonSize()));
      CommonRes.Align = std::max(CommonRes.Align, ObjSym.getCommonAlignment());
      R.Prevailing = false;
    }

    freeSymName(Sym);
  }

  check(Lto.add(std::move(*ObjOrErr), Resols),
        std::string("Failed to link module ") + F.name);
}

static void recordFile(std::string Filename, bool TempOutFile) {
  if (add_input_file(Filename.c_str()) != LDPS_OK)
    message(LDPL_FATAL,
            "Unable to add .o file to the link. File left behind in: %s",
            Filename.c_str());
  if (TempOutFile)
    Cleanup.push_back(Filename.c_str());
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

/// Add all required common symbols to M, which is expected to be the first
/// combined module.
static void addCommons(Module &M) {
  for (auto &I : Commons) {
    if (!I.second.Prevailing)
      continue;
    ArrayType *Ty =
        ArrayType::get(Type::getInt8Ty(M.getContext()), I.second.Size);
    GlobalVariable *OldGV = M.getNamedGlobal(I.first);
    auto *GV = new GlobalVariable(M, Ty, false, GlobalValue::CommonLinkage,
                                  ConstantAggregateZero::get(Ty), "");
    GV->setAlignment(I.second.Align);
    if (OldGV) {
      OldGV->replaceAllUsesWith(ConstantExpr::getBitCast(GV, OldGV->getType()));
      GV->takeName(OldGV);
      OldGV->eraseFromParent();
    } else {
      GV->setName(I.first);
    }
    // We may only internalize commons if there is a single LTO task because
    // other native object files may require the common.
    if (MaxTasks == 1 && !I.second.VisibleToRegularObj)
      GV->setLinkage(GlobalValue::InternalLinkage);
  }
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

static std::unique_ptr<LTO> createLTO() {
  Config Conf;
  ThinBackend Backend;
  unsigned ParallelCodeGenParallelismLevel = 1;

  Conf.CPU = options::mcpu;
  Conf.Options = InitTargetOptionsFromCodeGenFlags();

  // Disable the new X86 relax relocations since gold might not support them.
  // FIXME: Check the gold version or add a new option to enable them.
  Conf.Options.RelaxELFRelocations = false;

  Conf.MAttrs = MAttrs;
  Conf.RelocModel = *RelocationModel;
  Conf.CGOptLevel = getCGOptLevel();
  Conf.DisableVerify = options::DisableVerify;
  Conf.OptLevel = options::OptLevel;
  if (options::Parallelism) {
    if (options::thinlto)
      Backend = createInProcessThinBackend(options::Parallelism);
    else
      ParallelCodeGenParallelismLevel = options::Parallelism;
  }
  if (options::thinlto_index_only) {
    std::string OldPrefix, NewPrefix;
    getThinLTOOldAndNewPrefix(OldPrefix, NewPrefix);
    Backend = createWriteIndexesThinBackend(
        OldPrefix, NewPrefix, options::thinlto_emit_imports_files,
        options::thinlto_linked_objects_file);
  }

  Conf.OverrideTriple = options::triple;
  Conf.DefaultTriple = sys::getDefaultTargetTriple();

  Conf.DiagHandler = diagnosticHandler;

  Conf.PreOptModuleHook = [](size_t Task, Module &M) {
    if (Task == 0)
      addCommons(M);
    return true;
  };

  switch (options::TheOutputType) {
  case options::OT_NORMAL:
    break;

  case options::OT_DISABLE:
    Conf.PreOptModuleHook = [](size_t Task, Module &M) { return false; };
    break;

  case options::OT_BC_ONLY:
    Conf.PostInternalizeModuleHook = [](size_t Task, Module &M) {
      std::error_code EC;
      raw_fd_ostream OS(output_name, EC, sys::fs::OpenFlags::F_None);
      if (EC)
        message(LDPL_FATAL, "Failed to write the output file.");
      WriteBitcodeToFile(&M, OS, /* ShouldPreserveUseListOrder */ false);
      return false;
    };
    break;

  case options::OT_SAVE_TEMPS:
    check(Conf.addSaveTemps(output_name, /* UseInputModulePath */ true));
    break;
  }

  return llvm::make_unique<LTO>(std::move(Conf), Backend,
                                ParallelCodeGenParallelismLevel);
}

/// gold informs us that all symbols have been read. At this point, we use
/// get_symbols to see if any of our definitions have been overridden by a
/// native object file. Then, perform optimization and codegen.
static ld_plugin_status allSymbolsReadHook() {
  if (Modules.empty())
    return LDPS_OK;

  if (unsigned NumOpts = options::extra.size())
    cl::ParseCommandLineOptions(NumOpts, &options::extra[0]);

  std::unique_ptr<LTO> Lto = createLTO();

  for (claimed_file &F : Modules) {
    PluginInputFile InputFile(F.handle);
    const void *View = getSymbolsAndView(F);
    if (!View)
      continue;
    addModule(*Lto, F, View);
  }

  SmallString<128> Filename;
  // Note that openOutputFile will append a unique ID for each task
  if (!options::obj_path.empty())
    Filename = options::obj_path;
  else if (options::TheOutputType == options::OT_SAVE_TEMPS)
    Filename = output_name + ".o";
  bool SaveTemps = !Filename.empty();

  MaxTasks = Lto->getMaxTasks();
  std::vector<uintptr_t> IsTemporary(MaxTasks);
  std::vector<SmallString<128>> Filenames(MaxTasks);

  auto AddStream = [&](size_t Task) {
    int FD = openOutputFile(Filename, /*TempOutFile=*/!SaveTemps,
                            Filenames[Task], MaxTasks > 1 ? Task : -1);
    IsTemporary[Task] = !SaveTemps;

    return llvm::make_unique<llvm::raw_fd_ostream>(FD, true);
  };

  check(Lto->run(AddStream));

  if (options::TheOutputType == options::OT_DISABLE ||
      options::TheOutputType == options::OT_BC_ONLY)
    return LDPS_OK;

  if (options::thinlto_index_only) {
    cleanup_hook();
    exit(0);
  }

  for (unsigned I = 0; I != MaxTasks; ++I)
    if (!Filenames[I].empty())
      recordFile(Filenames[I].str(), IsTemporary[I]);

  if (!options::extra_library_path.empty() &&
      set_extra_library_path(options::extra_library_path.c_str()) != LDPS_OK)
    message(LDPL_FATAL, "Unable to set the extra library path.");

  return LDPS_OK;
}

static ld_plugin_status all_symbols_read_hook(void) {
  ld_plugin_status Ret = allSymbolsReadHook();
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
