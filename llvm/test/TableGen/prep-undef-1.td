// RUN: llvm-tblgen %s | FileCheck %s

// CHECK: def TEST_MACRONAME
#define MACRONAME
#undef MACRONAME

#ifndef MACRONAME
def TEST_MACRONAME;
#endif
