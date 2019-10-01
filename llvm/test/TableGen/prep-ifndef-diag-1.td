// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s

// CHECK: error: Expected macro name after #ifndef
#ifndef 1
