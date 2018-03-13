//===-- thread_inferior.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <string>
#include <cstdlib>

int main() {
  const char *value = std::getenv("LLDB_TEST_MAGIC_VARIABLE");
  if (!value)
    return 1;
  if (std::string(value) != "LLDB_TEST_MAGIC_VALUE")
    return 2;
  return 0;
}
