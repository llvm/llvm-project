// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

#include <stdio.h>

union U{
  int i[8];
  char s[80];
};

void format_message(char *buffer, union U *u) {
  sprintf(buffer, u->s);
}

