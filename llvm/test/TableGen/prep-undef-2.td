// RUN: llvm-tblgen %s | FileCheck %s

// CHECK: def TEST_MACRONAME
#undef MACRONAME
#define MACRONAME
#ifdef MACRONAME
def TEST_MACRONAME;
#endif
