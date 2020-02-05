//===-- thread_inferior.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
