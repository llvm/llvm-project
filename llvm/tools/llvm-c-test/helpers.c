/*===-- helpers.c - tool for testing libLLVM and llvm-c API ---------------===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* Helper functions                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include <stdio.h>
#include <string.h>

#define MAX_TOKENS 512
#define MAX_LINE_LEN 1024

void llvm_tokenize_stdin(void (*cb)(char **tokens, int ntokens)) {
  char line[MAX_LINE_LEN];
  char *tokbuf[MAX_TOKENS];

  while (fgets(line, sizeof(line), stdin)) {
    int c = 0;

    if (line[0] == ';' || line[0] == '\n')
      continue;

    while (c < MAX_TOKENS) {
      tokbuf[c] = strtok(c ? NULL : line, " \n");
      if (!tokbuf[c])
        break;
      c++;
    }
    if (c)
      cb(tokbuf, c);
  }
}
