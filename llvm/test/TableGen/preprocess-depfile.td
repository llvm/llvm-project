// Check that -E writes a dependency file listing the main input file and every
// file it includes, directly or indirectly. Files included more than once are
// only listed once, and files included from inactive regions are not listed.

// RUN: rm -rf %t && split-file %s %t
// RUN: llvm-tblgen -E -I %t/inc %t/main.td -o %t/out.td -d %t/out.d
// RUN: FileCheck %s --input-file=%t/out.d --match-full-lines -DDIR=%t

// CHECK:[[DIR]]{{/|\\}}out.td: [[DIR]]{{/|\\}}main.td [[DIR]]{{/|\\}}inc{{/|\\}}a.td [[DIR]]{{/|\\}}inc{{/|\\}}b.td

// Check that the dependency file is the same as without -E.

// RUN: llvm-tblgen -I %t/inc %t/main.td -o %t/out.td -d %t/out2.d
// RUN: cmp %t/out.d %t/out2.d

// RUN: not llvm-tblgen -E -I %t/inc %t/main.td -d %t/out.d 2>&1 | FileCheck %s --check-prefix=NO-OUTPUT
// NO-OUTPUT: llvm-tblgen{{.*}}: the option -d must be used together with -o

//--- main.td
include "a.td"
include "b.td"
#ifdef UNDEFINED
include "unused.td"
#endif
def m;

//--- inc/a.td
include "b.td"
def a;

//--- inc/b.td
#ifndef B_TD
#define B_TD
def b;
#endif

//--- inc/unused.td
def unused;
