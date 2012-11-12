/*===-- llvm-c/lto.h - LTO Public C Interface ---------------------*- C -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header provides public interface to an abstract link time optimization*|
|* library.  LLVM provides an implementation of this interface for use with   *|
|* llvm bitcode files.                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LTO_H
#define LTO_H  1

#include <stdbool.h>
#include <stddef.h>
#include <unistd.h>

/**
 * @defgroup LLVMCLTO LTO
 * @ingroup LLVMC
 *
 * @{
 */

#define LTO_API_VERSION 4

typedef enum {
    LTO_SYMBOL_ALIGNMENT_MASK              = 0x0000001F, /* log2 of alignment */
    LTO_SYMBOL_PERMISSIONS_MASK            = 0x000000E0,
    LTO_SYMBOL_PERMISSIONS_CODE            = 0x000000A0,
    LTO_SYMBOL_PERMISSIONS_DATA            = 0x000000C0,
    LTO_SYMBOL_PERMISSIONS_RODATA          = 0x00000080,
    LTO_SYMBOL_DEFINITION_MASK             = 0x00000700,
    LTO_SYMBOL_DEFINITION_REGULAR          = 0x00000100,
    LTO_SYMBOL_DEFINITION_TENTATIVE        = 0x00000200,
    LTO_SYMBOL_DEFINITION_WEAK             = 0x00000300,
    LTO_SYMBOL_DEFINITION_UNDEFINED        = 0x00000400,
    LTO_SYMBOL_DEFINITION_WEAKUNDEF        = 0x00000500,
    LTO_SYMBOL_SCOPE_MASK                  = 0x00003800,
    LTO_SYMBOL_SCOPE_INTERNAL              = 0x00000800,
    LTO_SYMBOL_SCOPE_HIDDEN                = 0x00001000,
    LTO_SYMBOL_SCOPE_PROTECTED             = 0x00002000,
    LTO_SYMBOL_SCOPE_DEFAULT               = 0x00001800,
    LTO_SYMBOL_SCOPE_DEFAULT_CAN_BE_HIDDEN = 0x00002800
} lto_symbol_attributes;

typedef enum {
    LTO_DEBUG_MODEL_NONE         = 0,
    LTO_DEBUG_MODEL_DWARF        = 1
} lto_debug_model;

typedef enum {
    LTO_CODEGEN_PIC_MODEL_STATIC         = 0,
    LTO_CODEGEN_PIC_MODEL_DYNAMIC        = 1,
    LTO_CODEGEN_PIC_MODEL_DYNAMIC_NO_PIC = 2
} lto_codegen_model;


/** opaque reference to a loaded object module */
typedef struct LTOModule*         lto_module_t;

/** opaque reference to a code generator */
typedef struct LTOCodeGenerator*  lto_code_gen_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Returns a printable string.
 */
extern const char*
lto_get_version(void);


/**
 * Returns the last error string or NULL if last operation was successful.
 */
extern const char*
lto_get_error_message(void);

/**
 * Checks if a file is a loadable object file.
 */
extern bool
lto_module_is_object_file(const char* path);


/**
 * Checks if a file is a loadable object compiled for requested target.
 */
extern bool
lto_module_is_object_file_for_target(const char* path,
                                     const char* target_triple_prefix);


/**
 * Checks if a buffer is a loadable object file.
 */
extern bool
lto_module_is_object_file_in_memory(const void* mem, size_t length);


/**
 * Checks if a buffer is a loadable object compiled for requested target.
 */
extern bool
lto_module_is_object_file_in_memory_for_target(const void* mem, size_t length,
                                              const char* target_triple_prefix);


/**
 * Loads an object file from disk.
 * Returns NULL on error (check lto_get_error_message() for details).
 */
extern lto_module_t
lto_module_create(const char* path);


/**
 * Loads an object file from memory.
 * Returns NULL on error (check lto_get_error_message() for details).
 */
extern lto_module_t
lto_module_create_from_memory(const void* mem, size_t length);

/**
 * Loads an object file from disk. The seek point of fd is not preserved.
 * Returns NULL on error (check lto_get_error_message() for details).
 */
extern lto_module_t
lto_module_create_from_fd(int fd, const char *path, size_t file_size);

/**
 * Loads an object file from disk. The seek point of fd is not preserved.
 * Returns NULL on error (check lto_get_error_message() for details).
 */
extern lto_module_t
lto_module_create_from_fd_at_offset(int fd, const char *path, size_t file_size,
                                    size_t map_size, off_t offset);


/**
 * Frees all memory internally allocated by the module.
 * Upon return the lto_module_t is no longer valid.
 */
extern void
lto_module_dispose(lto_module_t mod);


/**
 * Returns triple string which the object module was compiled under.
 */
extern const char*
lto_module_get_target_triple(lto_module_t mod);

/**
 * Sets triple string with which the object will be codegened.
 */
extern void
lto_module_set_target_triple(lto_module_t mod, const char *triple);


/**
 * Returns the number of symbols in the object module.
 */
extern unsigned int
lto_module_get_num_symbols(lto_module_t mod);


/**
 * Returns the name of the ith symbol in the object module.
 */
extern const char*
lto_module_get_symbol_name(lto_module_t mod, unsigned int index);


/**
 * Returns the attributes of the ith symbol in the object module.
 */
extern lto_symbol_attributes
lto_module_get_symbol_attribute(lto_module_t mod, unsigned int index);


/**
 * Instantiates a code generator.
 * Returns NULL on error (check lto_get_error_message() for details).
 */
extern lto_code_gen_t
lto_codegen_create(void);


/**
 * Frees all code generator and all memory it internally allocated.
 * Upon return the lto_code_gen_t is no longer valid.
 */
extern void
lto_codegen_dispose(lto_code_gen_t);



/**
 * Add an object module to the set of modules for which code will be generated.
 * Returns true on error (check lto_get_error_message() for details).
 */
extern bool
lto_codegen_add_module(lto_code_gen_t cg, lto_module_t mod);



/**
 * Sets if debug info should be generated.
 * Returns true on error (check lto_get_error_message() for details).
 */
extern bool
lto_codegen_set_debug_model(lto_code_gen_t cg, lto_debug_model);


/**
 * Sets which PIC code model to generated.
 * Returns true on error (check lto_get_error_message() for details).
 */
extern bool
lto_codegen_set_pic_model(lto_code_gen_t cg, lto_codegen_model);


/**
 * Sets the cpu to generate code for.
 */
extern void
lto_codegen_set_cpu(lto_code_gen_t cg, const char *cpu);


/**
 * Sets the location of the assembler tool to run. If not set, libLTO
 * will use gcc to invoke the assembler.
 */
extern void
lto_codegen_set_assembler_path(lto_code_gen_t cg, const char* path);

/**
 * Sets extra arguments that libLTO should pass to the assembler.
 */
extern void
lto_codegen_set_assembler_args(lto_code_gen_t cg, const char **args,
                               int nargs);

/**
 * Adds to a list of all global symbols that must exist in the final
 * generated code.  If a function is not listed, it might be
 * inlined into every usage and optimized away.
 */
extern void
lto_codegen_add_must_preserve_symbol(lto_code_gen_t cg, const char* symbol);

/**
 * Writes a new object file at the specified path that contains the
 * merged contents of all modules added so far.
 * Returns true on error (check lto_get_error_message() for details).
 */
extern bool
lto_codegen_write_merged_modules(lto_code_gen_t cg, const char* path);

/**
 * Generates code for all added modules into one native object file.
 * On success returns a pointer to a generated mach-o/ELF buffer and
 * length set to the buffer size.  The buffer is owned by the
 * lto_code_gen_t and will be freed when lto_codegen_dispose()
 * is called, or lto_codegen_compile() is called again.
 * On failure, returns NULL (check lto_get_error_message() for details).
 */
extern const void*
lto_codegen_compile(lto_code_gen_t cg, size_t* length);

/**
 * Generates code for all added modules into one native object file.
 * The name of the file is written to name. Returns true on error.
 */
extern bool
lto_codegen_compile_to_file(lto_code_gen_t cg, const char** name);


/**
 * Sets options to help debug codegen bugs.
 */
extern void
lto_codegen_debug_options(lto_code_gen_t cg, const char *);

#ifdef __cplusplus
}
#endif

/**
 * @}
 */

#endif
