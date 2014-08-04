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
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/LTO/LTOCodeGenerator.h"
#include "llvm/LTO/LTOModule.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"

// extra command-line flags needed for LTOCodeGenerator
static cl::opt<bool>
DisableOpt("disable-opt", cl::init(false),
  cl::desc("Do not run any optimization passes"));

static cl::opt<bool>
DisableInline("disable-inlining", cl::init(false),
  cl::desc("Do not run the inliner pass"));

static cl::opt<bool>
DisableGVNLoadPRE("disable-gvn-loadpre", cl::init(false),
  cl::desc("Do not run the GVN load PRE pass"));

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
    InitializeAllTargetInfos();
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmParsers();
    InitializeAllAsmPrinters();
    InitializeAllDisassemblers();
    initialized = true;
  }
}

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(LTOCodeGenerator, lto_code_gen_t)
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

unsigned int lto_module_get_num_deplibs(lto_module_t mod) {
  return unwrap(mod)->getDependentLibraryCount();
}

const char* lto_module_get_deplib(lto_module_t mod, unsigned int index) {
  return unwrap(mod)->getDependentLibrary(index);
}

unsigned int lto_module_get_num_linkeropts(lto_module_t mod) {
  return unwrap(mod)->getLinkerOptCount();
}

const char* lto_module_get_linkeropt(lto_module_t mod, unsigned int index) {
  return unwrap(mod)->getLinkerOpt(index);
}

void lto_codegen_set_diagnostic_handler(lto_code_gen_t cg,
                                        lto_diagnostic_handler_t diag_handler,
                                        void *ctxt) {
  unwrap(cg)->setDiagnosticHandler(diag_handler, ctxt);
}

lto_code_gen_t lto_codegen_create(void) {
  lto_initialize();

  TargetOptions Options = InitTargetOptionsFromCodeGenFlags();

  LTOCodeGenerator *CodeGen = new LTOCodeGenerator();
  if (CodeGen)
    CodeGen->setTargetOptions(Options);
  return wrap(CodeGen);
}

void lto_codegen_dispose(lto_code_gen_t cg) { delete unwrap(cg); }

bool lto_codegen_add_module(lto_code_gen_t cg, lto_module_t mod) {
  return !unwrap(cg)->addModule(unwrap(mod), sLastErrorString);
}

bool lto_codegen_set_debug_model(lto_code_gen_t cg, lto_debug_model debug) {
  unwrap(cg)->setDebugInfo(debug);
  return false;
}

bool lto_codegen_set_pic_model(lto_code_gen_t cg, lto_codegen_model model) {
  unwrap(cg)->setCodePICModel(model);
  return false;
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

bool lto_codegen_write_merged_modules(lto_code_gen_t cg, const char *path) {
  if (!parsedOptions) {
    unwrap(cg)->parseCodeGenDebugOptions();
    lto_add_attrs(cg);
    parsedOptions = true;
  }
  return !unwrap(cg)->writeMergedModules(path, sLastErrorString);
}

const void *lto_codegen_compile(lto_code_gen_t cg, size_t *length) {
  if (!parsedOptions) {
    unwrap(cg)->parseCodeGenDebugOptions();
    lto_add_attrs(cg);
    parsedOptions = true;
  }
  return unwrap(cg)->compile(length, DisableOpt, DisableInline,
                             DisableGVNLoadPRE, sLastErrorString);
}

bool lto_codegen_compile_to_file(lto_code_gen_t cg, const char **name) {
  if (!parsedOptions) {
    unwrap(cg)->parseCodeGenDebugOptions();
    lto_add_attrs(cg);
    parsedOptions = true;
  }
  return !unwrap(cg)->compile_to_file(name, DisableOpt, DisableInline,
                                      DisableGVNLoadPRE, sLastErrorString);
}

void lto_codegen_debug_options(lto_code_gen_t cg, const char *opt) {
  unwrap(cg)->setCodeGenDebugOptions(opt);
}
