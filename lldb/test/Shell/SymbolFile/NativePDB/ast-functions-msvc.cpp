// clang-format off
// REQUIRES: msvc

// RUN: %build --compiler=msvc --nodefaultlib -o %t.exe -- %S/ast-functions.cpp

// RUN: %lldb -f %t.exe -s \
// RUN:     %p/Inputs/ast-functions.lldbinit 2>&1 | FileCheck %S/ast-functions.cpp
