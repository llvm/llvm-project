// clang-format off
// REQUIRES: msvc

// RUN: %build --compiler=msvc --nodefaultlib -o %t.exe -- %S/ast-functions.cpp

// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -s \
// RUN:     %p/Inputs/ast-functions.lldbinit 2>&1 | FileCheck %S/ast-functions.cpp
