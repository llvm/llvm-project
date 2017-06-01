/*===-- disassemble.c - tool for testing libLLVM and llvm-c API -----------===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file implements the --disassemble command in llvm-c-test.             *|
|* --disassemble reads lines from stdin, parses them as a triple and hex      *|
|*  machine code, and prints disassembly of the machine code.                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include "llvm-c-test.h"
#include "llvm-c/Disassembler.h"
#include "llvm-c/Target.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void pprint(int pos, unsigned char *buf, int len, const char *disasm) {
  int i;
  printf("%04x:  ", pos);
  for (i = 0; i < 8; i++) {
    if (i < len) {
      printf("%02x ", buf[i]);
    } else {
      printf("   ");
    }
  }

  printf("   %s\n", disasm);
}

static void do_disassemble(const char *triple, const char *features,
                           unsigned char *buf, int siz) {
  LLVMDisasmContextRef D = LLVMCreateDisasmCPUFeatures(triple, "", features,
                                                       NULL, 0, NULL, NULL);
  char outline[1024];
  int pos;

  if (!D) {
    printf("ERROR: Couldn't create disassembler for triple %s\n", triple);
    return;
  }

  pos = 0;
  while (pos < siz) {
    size_t l = LLVMDisasmInstruction(D, buf + pos, siz - pos, 0, outline,
                                     sizeof(outline));
    if (!l) {
      pprint(pos, buf + pos, 1, "\t???");
      pos++;
    } else {
      pprint(pos, buf + pos, l, outline);
      pos += l;
    }
  }

  LLVMDisasmDispose(D);
}

static void handle_line(char **tokens, int ntokens) {
  unsigned char disbuf[128];
  size_t disbuflen = 0;
  const char *triple = tokens[0];
  const char *features = tokens[1];
  int i;

  printf("triple: %s, features: %s\n", triple, features);
  if (!strcmp(features, "NULL"))
    features = "";

  for (i = 2; i < ntokens; i++) {
    disbuf[disbuflen++] = strtol(tokens[i], NULL, 16);
    if (disbuflen >= sizeof(disbuf)) {
      fprintf(stderr, "Warning: Too long line, truncating\n");
      break;
    }
  }
  do_disassemble(triple, features, disbuf, disbuflen);
}

int llvm_disassemble(void) {
  LLVMInitializeAllTargetInfos();
  LLVMInitializeAllTargetMCs();
  LLVMInitializeAllDisassemblers();

  llvm_tokenize_stdin(handle_line);

  return 0;
}
