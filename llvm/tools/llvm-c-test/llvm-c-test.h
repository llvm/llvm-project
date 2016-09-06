/*===-- llvm-c-test.h - tool for testing libLLVM and llvm-c API -----------===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* Header file for llvm-c-test                                                *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/
#ifndef LLVM_C_TEST_H
#define LLVM_C_TEST_H

#include <stdbool.h>
#include "llvm-c/Core.h"

#ifdef __cplusplus
extern "C" {
#endif

// helpers.c
void llvm_tokenize_stdin(void (*cb)(char **tokens, int ntokens));

// module.c
LLVMModuleRef llvm_load_module(bool Lazy, bool New);
int llvm_module_dump(bool Lazy, bool New);
int llvm_module_list_functions(void);
int llvm_module_list_globals(void);

// calc.c
int llvm_calc(void);

// disassemble.c
int llvm_disassemble(void);

// metadata.c
int llvm_add_named_metadata_operand(void);
int llvm_set_metadata(void);

// object.c
int llvm_object_list_sections(void);
int llvm_object_list_symbols(void);

// targets.c
int llvm_targets_list(void);

// echo.c
int llvm_echo(void);

// diagnostic.c
int llvm_test_diagnostic_handler(void);

#ifdef __cplusplus
}
#endif /* !defined(__cplusplus) */

#endif
