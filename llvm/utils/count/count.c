/*===- count.c - The 'count' testing tool ---------------------------------===*\
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
\*===----------------------------------------------------------------------===*/

#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv) {
  unsigned Count, NumLines, NumRead;
  char Buffer[4096], *End;

  if (argc != 2) {
    fprintf(stderr, "usage: %s <expected line count>\n", argv[0]);
    return 2;
  }

  Count = strtol(argv[1], &End, 10);
  if (*End != '\0' && End != argv[1]) {
    fprintf(stderr, "%s: invalid count argument '%s'\n", argv[0], argv[1]);
    return 2;
  }

  NumLines = 0;
  do {
    unsigned i;

    NumRead = fread(Buffer, 1, sizeof(Buffer), stdin);

    for (i = 0; i != NumRead; ++i)
      if (Buffer[i] == '\n')
        ++NumLines;
  } while (NumRead == sizeof(Buffer));
    
  if (!feof(stdin)) {
    fprintf(stderr, "%s: error reading stdin\n", argv[0]);
    return 3;
  }

  if (Count != NumLines) {
    fprintf(stderr, "Expected %d lines, got %d.\n", Count, NumLines);
    return 1;
  }

  return 0;
}
