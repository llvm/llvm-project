// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

#include <wchar.h>

struct {
  wchar_t *name;
} syms = { L"NUL" };
