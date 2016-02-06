/*===-- object.c - tool for testing libLLVM and llvm-c API ----------------===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file implements the --object-list-sections and --object-list-symbols  *|
|* commands in llvm-c-test.                                                   *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include "llvm-c-test.h"
#include "llvm-c/Object.h"
#include <stdio.h>
#include <stdlib.h>

int llvm_object_list_sections(void) {
  LLVMMemoryBufferRef MB;
  LLVMObjectFileRef O;
  LLVMSectionIteratorRef sect;
  char *msg = NULL;

  if (LLVMCreateMemoryBufferWithSTDIN(&MB, &msg)) {
    fprintf(stderr, "Error reading file: %s\n", msg);
    exit(1);
  }

  O = LLVMCreateObjectFile(MB);
  if (!O) {
    fprintf(stderr, "Error reading object\n");
    exit(1);
  }

  sect = LLVMGetSections(O);
  while (!LLVMIsSectionIteratorAtEnd(O, sect)) {
    printf("'%s': @0x%08" PRIx64 " +%" PRIu64 "\n", LLVMGetSectionName(sect),
           LLVMGetSectionAddress(sect), LLVMGetSectionSize(sect));

    LLVMMoveToNextSection(sect);
  }

  LLVMDisposeSectionIterator(sect);

  LLVMDisposeObjectFile(O);

  return 0;
}

int llvm_object_list_symbols(void) {
  LLVMMemoryBufferRef MB;
  LLVMObjectFileRef O;
  LLVMSectionIteratorRef sect;
  LLVMSymbolIteratorRef sym;
  char *msg = NULL;

  if (LLVMCreateMemoryBufferWithSTDIN(&MB, &msg)) {
    fprintf(stderr, "Error reading file: %s\n", msg);
    exit(1);
  }

  O = LLVMCreateObjectFile(MB);
  if (!O) {
    fprintf(stderr, "Error reading object\n");
    exit(1);
  }

  sect = LLVMGetSections(O);
  sym = LLVMGetSymbols(O);
  while (!LLVMIsSymbolIteratorAtEnd(O, sym)) {

    LLVMMoveToContainingSection(sect, sym);
    printf("%s @0x%08" PRIx64 " +%" PRIu64 " (%s)\n", LLVMGetSymbolName(sym),
           LLVMGetSymbolAddress(sym), LLVMGetSymbolSize(sym),
           LLVMGetSectionName(sect));

    LLVMMoveToNextSymbol(sym);
  }

  LLVMDisposeSymbolIterator(sym);

  LLVMDisposeObjectFile(O);

  return 0;
}
