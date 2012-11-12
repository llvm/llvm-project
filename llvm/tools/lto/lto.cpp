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
#include "llvm-c/Core.h"

#include "LTOModule.h"
#include "LTOCodeGenerator.h"


// Holds most recent error string.
// *** Not thread safe ***
static std::string sLastErrorString;

/// lto_get_version - Returns a printable string.
extern const char* lto_get_version() {
  return LTOCodeGenerator::getVersionString();
}

/// lto_get_error_message - Returns the last error string or NULL if last
/// operation was successful.
const char* lto_get_error_message() {
  return sLastErrorString.c_str();
}

/// lto_module_is_object_file - Validates if a file is a loadable object file.
bool lto_module_is_object_file(const char* path) {
  return LTOModule::isBitcodeFile(path);
}

/// lto_module_is_object_file_for_target - Validates if a file is a loadable
/// object file compilable for requested target.
bool lto_module_is_object_file_for_target(const char* path,
                                          const char* target_triplet_prefix) {
  return LTOModule::isBitcodeFileForTarget(path, target_triplet_prefix);
}

/// lto_module_is_object_file_in_memory - Validates if a buffer is a loadable
/// object file.
bool lto_module_is_object_file_in_memory(const void* mem, size_t length) {
  return LTOModule::isBitcodeFile(mem, length);
}

/// lto_module_is_object_file_in_memory_for_target - Validates if a buffer is a
/// loadable object file compilable for the target.
bool
lto_module_is_object_file_in_memory_for_target(const void* mem,
                                            size_t length,
                                            const char* target_triplet_prefix) {
  return LTOModule::isBitcodeFileForTarget(mem, length, target_triplet_prefix);
}

/// lto_module_create - Loads an object file from disk. Returns NULL on error
/// (check lto_get_error_message() for details).
lto_module_t lto_module_create(const char* path) {
  return LTOModule::makeLTOModule(path, sLastErrorString);
}

/// lto_module_create_from_fd - Loads an object file from disk. Returns NULL on
/// error (check lto_get_error_message() for details).
lto_module_t lto_module_create_from_fd(int fd, const char *path, size_t size) {
  return LTOModule::makeLTOModule(fd, path, size, sLastErrorString);
}

/// lto_module_create_from_fd_at_offset - Loads an object file from disk.
/// Returns NULL on error (check lto_get_error_message() for details).
lto_module_t lto_module_create_from_fd_at_offset(int fd, const char *path,
                                                 size_t file_size,
                                                 size_t map_size,
                                                 off_t offset) {
  return LTOModule::makeLTOModule(fd, path, file_size, map_size,
                                  offset, sLastErrorString);
}

/// lto_module_create_from_memory - Loads an object file from memory. Returns
/// NULL on error (check lto_get_error_message() for details).
lto_module_t lto_module_create_from_memory(const void* mem, size_t length) {
  return LTOModule::makeLTOModule(mem, length, sLastErrorString);
}

/// lto_module_dispose - Frees all memory for a module. Upon return the
/// lto_module_t is no longer valid.
void lto_module_dispose(lto_module_t mod) {
  delete mod;
}

/// lto_module_get_target_triple - Returns triplet string which the object
/// module was compiled under.
const char* lto_module_get_target_triple(lto_module_t mod) {
  return mod->getTargetTriple();
}

/// lto_module_set_target_triple - Sets triple string with which the object will
/// be codegened.
void lto_module_set_target_triple(lto_module_t mod, const char *triple) {
  return mod->setTargetTriple(triple);
}

/// lto_module_get_num_symbols - Returns the number of symbols in the object
/// module.
unsigned int lto_module_get_num_symbols(lto_module_t mod) {
  return mod->getSymbolCount();
}

/// lto_module_get_symbol_name - Returns the name of the ith symbol in the
/// object module.
const char* lto_module_get_symbol_name(lto_module_t mod, unsigned int index) {
  return mod->getSymbolName(index);
}

/// lto_module_get_symbol_attribute - Returns the attributes of the ith symbol
/// in the object module.
lto_symbol_attributes lto_module_get_symbol_attribute(lto_module_t mod,
                                                      unsigned int index) {
  return mod->getSymbolAttributes(index);
}

/// lto_codegen_create - Instantiates a code generator. Returns NULL if there
/// is an error.
lto_code_gen_t lto_codegen_create(void) {
  return new LTOCodeGenerator();
}

/// lto_codegen_dispose - Frees all memory for a code generator. Upon return the
/// lto_code_gen_t is no longer valid.
void lto_codegen_dispose(lto_code_gen_t cg) {
  delete cg;
}

/// lto_codegen_add_module - Add an object module to the set of modules for
/// which code will be generated. Returns true on error (check
/// lto_get_error_message() for details).
bool lto_codegen_add_module(lto_code_gen_t cg, lto_module_t mod) {
  return cg->addModule(mod, sLastErrorString);
}

/// lto_codegen_set_debug_model - Sets what if any format of debug info should
/// be generated. Returns true on error (check lto_get_error_message() for
/// details).
bool lto_codegen_set_debug_model(lto_code_gen_t cg, lto_debug_model debug) {
  return cg->setDebugInfo(debug, sLastErrorString);
}

/// lto_codegen_set_pic_model - Sets what code model to generated. Returns true
/// on error (check lto_get_error_message() for details).
bool lto_codegen_set_pic_model(lto_code_gen_t cg, lto_codegen_model model) {
  return cg->setCodePICModel(model, sLastErrorString);
}

/// lto_codegen_set_cpu - Sets the cpu to generate code for.
void lto_codegen_set_cpu(lto_code_gen_t cg, const char *cpu) {
  return cg->setCpu(cpu);
}

/// lto_codegen_set_assembler_path - Sets the path to the assembler tool.
void lto_codegen_set_assembler_path(lto_code_gen_t cg, const char *path) {
  // In here only for backwards compatibility. We use MC now.
}

/// lto_codegen_set_assembler_args - Sets extra arguments that libLTO should
/// pass to the assembler.
void lto_codegen_set_assembler_args(lto_code_gen_t cg, const char **args,
                                    int nargs) {
  // In here only for backwards compatibility. We use MC now.
}

/// lto_codegen_add_must_preserve_symbol - Adds to a list of all global symbols
/// that must exist in the final generated code. If a function is not listed
/// there, it might be inlined into every usage and optimized away.
void lto_codegen_add_must_preserve_symbol(lto_code_gen_t cg,
                                          const char *symbol) {
  cg->addMustPreserveSymbol(symbol);
}

/// lto_codegen_write_merged_modules - Writes a new file at the specified path
/// that contains the merged contents of all modules added so far. Returns true
/// on error (check lto_get_error_message() for details).
bool lto_codegen_write_merged_modules(lto_code_gen_t cg, const char *path) {
  return cg->writeMergedModules(path, sLastErrorString);
}

/// lto_codegen_compile - Generates code for all added modules into one native
/// object file. On success returns a pointer to a generated mach-o/ELF buffer
/// and length set to the buffer size. The buffer is owned by the lto_code_gen_t
/// object and will be freed when lto_codegen_dispose() is called, or
/// lto_codegen_compile() is called again. On failure, returns NULL (check
/// lto_get_error_message() for details).
const void *lto_codegen_compile(lto_code_gen_t cg, size_t *length) {
  return cg->compile(length, sLastErrorString);
}

/// lto_codegen_compile_to_file - Generates code for all added modules into one
/// native object file. The name of the file is written to name. Returns true on
/// error.
bool lto_codegen_compile_to_file(lto_code_gen_t cg, const char **name) {
  return cg->compile_to_file(name, sLastErrorString);
}

/// lto_codegen_debug_options - Used to pass extra options to the code
/// generator.
void lto_codegen_debug_options(lto_code_gen_t cg, const char *opt) {
  cg->setCodeGenDebugOptions(opt);
}
