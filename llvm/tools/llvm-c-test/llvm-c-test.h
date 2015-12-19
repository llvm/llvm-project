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

// helpers.c
void tokenize_stdin(void (*cb)(char **tokens, int ntokens));

// module.c
int module_dump(bool Lazy, bool New);
int module_list_functions(void);
int module_list_globals(void);

// calc.c
int calc(void);

// disassemble.c
int disassemble(void);

// metadata.c
int add_named_metadata_operand(void);
int set_metadata(void);

// object.c
int object_list_sections(void);
int object_list_symbols(void);

// targets.c
int targets_list(void);

#endif
