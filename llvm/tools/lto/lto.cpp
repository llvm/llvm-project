//===-lto.cpp - LLVM Link Time Optimizer ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Link Time Optimization library. This library is
// intended to be used by linker to optimize code at link time.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/lto.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/LTO/LTOCodeGenerator.h"
#include "llvm/LTO/LTOModule.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"

// extra command-line flags needed for LTOCodeGenerator
static cl::opt<char>
OptLevel("O",
         cl::desc("Optimization level. [-O0, -O1, -O2, or -O3] "
                  "(default = '-O2')"),
         cl::Prefix,
         cl::ZeroOrMore,
         cl::init('2'));

static cl::opt<bool>
DisableInline("disable-inlining", cl::init(false),
  cl::desc("Do not run the inliner pass"));

static cl::opt<bool>
DisableGVNLoadPRE("disable-gvn-loadpre", cl::init(false),
  cl::desc("Do not run the GVN load PRE pass"));

static cl::opt<bool>
DisableLTOVectorization("disable-lto-vectorization", cl::init(false),
  cl::desc("Do not run loop or slp vectorization during LTO"));

#ifdef NDEBUG
static bool VerifyByDefault = false;
#else
static bool VerifyByDefault = true;
#endif

static cl::opt<bool> DisableVerify(
    "disable-llvm-verifier", cl::init(!VerifyByDefault),
    cl::desc("Don't run the LLVM verifier during the optimization pipeline"));

// Holds most recent error string.
// *** Not thread safe ***
static std::string sLastErrorString;

// Holds the initialization state of the LTO module.
// *** Not thread safe ***
static bool initialized = false;

// Holds the command-line option parsing state of the LTO module.
static bool parsedOptions = false;

// Initialize the configured targets if they have not been initialized.
static void lto_initialize() {
  if (!initialized) {
#ifdef LLVM_ON_WIN32
    // Dialog box on crash disabling doesn't work across DLL boundaries, so do
    // it here.
    llvm::sys::DisableSystemDialogsOnCrash();
#endif

    InitializeAllTargetInfos();
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmParsers();
    InitializeAllAsmPrinters();
    InitializeAllDisassemblers();
    initialized = true;
  }
}

namespace {

static void handleLibLTODiagnostic(lto_codegen_diagnostic_severity_t Severity,
                                   const char *Msg, void *) {
  sLastErrorString = Msg;
  sLastErrorString += "\n";
}

// This derived class owns the native object file. This helps implement the
// libLTO API semantics, which require that the code generator owns the object
// file.
struct LibLTOCodeGenerator : LTOCodeGenerator {
  LibLTOCodeGenerator() : LTOCodeGenerator(getGlobalContext()) {
    setDiagnosticHandler(handleLibLTODiagnostic, nullptr); }
  LibLTOCodeGenerator(std::unique_ptr<LLVMContext> Context)
      : LTOCodeGenerator(*Context), OwnedContext(std::move(Context)) {
    setDiagnosticHandler(handleLibLTODiagnostic, nullptr); }

  std::unique_ptr<MemoryBuffer> NativeObjectFile;
  std::unique_ptr<LLVMContext> OwnedContext;
};

}

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(LibLTOCodeGenerator, lto_code_gen_t)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(LTOModule, lto_module_t)

// Convert the subtarget features into a string to pass to LTOCodeGenerator.
static void lto_add_attrs(lto_code_gen_t cg) {
  LTOCodeGenerator *CG = unwrap(cg);
  if (MAttrs.size()) {
    std::string attrs;
    for (unsigned i = 0; i < MAttrs.size(); ++i) {
      if (i > 0)
        attrs.append(",");
      attrs.append(MAttrs[i]);
    }

    CG->setAttr(attrs.c_str());
  }

  if (OptLevel < '0' || OptLevel > '3')
    report_fatal_error("Optimization level must be between 0 and 3");
  CG->setOptLevel(OptLevel - '0');
}

extern const char* lto_get_version() {
  return LTOCodeGenerator::getVersionString();
}

const char* lto_get_error_message() {
  return sLastErrorString.c_str();
}

bool lto_module_is_object_file(const char* path) {
  return LTOModule::isBitcodeFile(path);
}

bool lto_module_is_object_file_for_target(const char* path,
                                          const char* target_triplet_prefix) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer = MemoryBuffer::getFile(path);
  if (!Buffer)
    return false;
  return LTOModule::isBitcodeForTarget(Buffer->get(), target_triplet_prefix);
}

bool lto_module_is_object_file_in_memory(const void* mem, size_t length) {
  return LTOModule::isBitcodeFile(mem, length);
}

bool
lto_module_is_object_file_in_memory_for_target(const void* mem,
                                            size_t length,
                                            const char* target_triplet_prefix) {
  std::unique_ptr<MemoryBuffer> buffer(LTOModule::makeBuffer(mem, length));
  if (!buffer)
    return false;
  return LTOModule::isBitcodeForTarget(buffer.get(), target_triplet_prefix);
}

lto_module_t lto_module_create(const char* path) {
  lto_initialize();
  llvm::TargetOptions Options = InitTargetOptionsFromCodeGenFlags();
  return wrap(LTOModule::createFromFile(path, Options, sLastErrorString));
}

lto_module_t lto_module_create_from_fd(int fd, const char *path, size_t size) {
  lto_initialize();
  llvm::TargetOptions Options = InitTargetOptionsFromCodeGenFlags();
  return wrap(
      LTOModule::createFromOpenFile(fd, path, size, Options, sLastErrorString));
}

lto_module_t lto_module_create_from_fd_at_offset(int fd, const char *path,
                                                 size_t file_size,
                                                 size_t map_size,
                                                 off_t offset) {
  lto_initialize();
  llvm::TargetOptions Options = InitTargetOptionsFromCodeGenFlags();
  return wrap(LTOModule::createFromOpenFileSlice(fd, path, map_size, offset,
                                                 Options, sLastErrorString));
}

lto_module_t lto_module_create_from_memory(const void* mem, size_t length) {
  lto_initialize();
  llvm::TargetOptions Options = InitTargetOptionsFromCodeGenFlags();
  return wrap(LTOModule::createFromBuffer(mem, length, Options, sLastErrorString));
}

lto_module_t lto_module_create_from_memory_with_path(const void* mem,
                                                     size_t length,
                                                     const char *path) {
  lto_initialize();
  llvm::TargetOptions Options = InitTargetOptionsFromCodeGenFlags();
  return wrap(
      LTOModule::createFromBuffer(mem, length, Options, sLastErrorString, path));
}

lto_module_t lto_module_create_in_local_context(const void *mem, size_t length,
                                                const char *path) {
  lto_initialize();
  llvm::TargetOptions Options = InitTargetOptionsFromCodeGenFlags();
  return wrap(LTOModule::createInLocalContext(mem, length, Options,
                                              sLastErrorString, path));
}

lto_module_t lto_module_create_in_codegen_context(const void *mem,
                                                  size_t length,
                                                  const char *path,
                                                  lto_code_gen_t cg) {
  lto_initialize();
  llvm::TargetOptions Options = InitTargetOptionsFromCodeGenFlags();
  return wrap(LTOModule::createInContext(mem, length, Options, sLastErrorString,
                                         path, &unwrap(cg)->getContext()));
}

void lto_module_dispose(lto_module_t mod) { delete unwrap(mod); }

const char* lto_module_get_target_triple(lto_module_t mod) {
  return unwrap(mod)->getTargetTriple().c_str();
}

void lto_module_set_target_triple(lto_module_t mod, const char *triple) {
  return unwrap(mod)->setTargetTriple(triple);
}

unsigned int lto_module_get_num_symbols(lto_module_t mod) {
  return unwrap(mod)->getSymbolCount();
}

const char* lto_module_get_symbol_name(lto_module_t mod, unsigned int index) {
  return unwrap(mod)->getSymbolName(index);
}

lto_symbol_attributes lto_module_get_symbol_attribute(lto_module_t mod,
                                                      unsigned int index) {
  return unwrap(mod)->getSymbolAttributes(index);
}

const char* lto_module_get_linkeropts(lto_module_t mod) {
  return unwrap(mod)->getLinkerOpts();
}

void lto_codegen_set_diagnostic_handler(lto_code_gen_t cg,
                                        lto_diagnostic_handler_t diag_handler,
                                        void *ctxt) {
  unwrap(cg)->setDiagnosticHandler(diag_handler, ctxt);
}

static lto_code_gen_t createCodeGen(bool InLocalContext) {
  lto_initialize();

  TargetOptions Options = InitTargetOptionsFromCodeGenFlags();

  LibLTOCodeGenerator *CodeGen =
      InLocalContext ? new LibLTOCodeGenerator(make_unique<LLVMContext>())
                     : new LibLTOCodeGenerator();
  CodeGen->setTargetOptions(Options);
  return wrap(CodeGen);
}

lto_code_gen_t lto_codegen_create(void) {
  return createCodeGen(/* InLocalContext */ false);
}

lto_code_gen_t lto_codegen_create_in_local_context(void) {
  return createCodeGen(/* InLocalContext */ true);
}

void lto_codegen_dispose(lto_code_gen_t cg) { delete unwrap(cg); }

bool lto_codegen_add_module(lto_code_gen_t cg, lto_module_t mod) {
  return !unwrap(cg)->addModule(unwrap(mod));
}

void lto_codegen_set_module(lto_code_gen_t cg, lto_module_t mod) {
  unwrap(cg)->setModule(std::unique_ptr<LTOModule>(unwrap(mod)));
}

bool lto_codegen_set_debug_model(lto_code_gen_t cg, lto_debug_model debug) {
  unwrap(cg)->setDebugInfo(debug);
  return false;
}

bool lto_codegen_set_pic_model(lto_code_gen_t cg, lto_codegen_model model) {
  switch (model) {
  case LTO_CODEGEN_PIC_MODEL_STATIC:
    unwrap(cg)->setCodePICModel(Reloc::Static);
    return false;
  case LTO_CODEGEN_PIC_MODEL_DYNAMIC:
    unwrap(cg)->setCodePICModel(Reloc::PIC_);
    return false;
  case LTO_CODEGEN_PIC_MODEL_DYNAMIC_NO_PIC:
    unwrap(cg)->setCodePICModel(Reloc::DynamicNoPIC);
    return false;
  case LTO_CODEGEN_PIC_MODEL_DEFAULT:
    unwrap(cg)->setCodePICModel(Reloc::Default);
    return false;
  }
  sLastErrorString = "Unknown PIC model";
  return true;
}

void lto_codegen_set_cpu(lto_code_gen_t cg, const char *cpu) {
  return unwrap(cg)->setCpu(cpu);
}

void lto_codegen_set_assembler_path(lto_code_gen_t cg, const char *path) {
  // In here only for backwards compatibility. We use MC now.
}

void lto_codegen_set_assembler_args(lto_code_gen_t cg, const char **args,
                                    int nargs) {
  // In here only for backwards compatibility. We use MC now.
}

void lto_codegen_add_must_preserve_symbol(lto_code_gen_t cg,
                                          const char *symbol) {
  unwrap(cg)->addMustPreserveSymbol(symbol);
}

static void maybeParseOptions(lto_code_gen_t cg) {
  if (!parsedOptions) {
    unwrap(cg)->parseCodeGenDebugOptions();
    lto_add_attrs(cg);
    parsedOptions = true;
  }
}

bool lto_codegen_write_merged_modules(lto_code_gen_t cg, const char *path) {
  maybeParseOptions(cg);
  return !unwrap(cg)->writeMergedModules(path);
}

const void *lto_codegen_compile(lto_code_gen_t cg, size_t *length) {
  maybeParseOptions(cg);
  LibLTOCodeGenerator *CG = unwrap(cg);
  CG->NativeObjectFile =
      CG->compile(DisableVerify, DisableInline, DisableGVNLoadPRE,
                  DisableLTOVectorization);
  if (!CG->NativeObjectFile)
    return nullptr;
  *length = CG->NativeObjectFile->getBufferSize();
  return CG->NativeObjectFile->getBufferStart();
}

bool lto_codegen_optimize(lto_code_gen_t cg) {
  maybeParseOptions(cg);
  return !unwrap(cg)->optimize(DisableVerify, DisableInline, DisableGVNLoadPRE,
                               DisableLTOVectorization);
}

const void *lto_codegen_compile_optimized(lto_code_gen_t cg, size_t *length) {
  maybeParseOptions(cg);
  LibLTOCodeGenerator *CG = unwrap(cg);
  CG->NativeObjectFile = CG->compileOptimized();
  if (!CG->NativeObjectFile)
    return nullptr;
  *length = CG->NativeObjectFile->getBufferSize();
  return CG->NativeObjectFile->getBufferStart();
}

bool lto_codegen_compile_to_file(lto_code_gen_t cg, const char **name) {
  maybeParseOptions(cg);
  return !unwrap(cg)->compile_to_file(
      name, DisableVerify, DisableInline, DisableGVNLoadPRE,
      DisableLTOVectorization);
}

void lto_codegen_debug_options(lto_code_gen_t cg, const char *opt) {
  unwrap(cg)->setCodeGenDebugOptions(opt);
}

unsigned int lto_api_version() { return LTO_API_VERSION; }

void lto_codegen_set_should_internalize(lto_code_gen_t cg,
                                        bool ShouldInternalize) {
  unwrap(cg)->setShouldInternalize(ShouldInternalize);
}

void lto_codegen_set_should_embed_uselists(lto_code_gen_t cg,
                                           lto_bool_t ShouldEmbedUselists) {
  unwrap(cg)->setShouldEmbedUselists(ShouldEmbedUselists);
}
