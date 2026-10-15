// Check how -E handles include directives. Each switch into or out of an
// included file starts a new output line, and the first token on each line is
// indented to its original column. -E does not parse records, so it does not
// matter that including the same file twice creates duplicate definitions.

// RUN: rm -rf %t && split-file %s %t
// RUN: llvm-tblgen -E -I %t %t/main.td | FileCheck %s --match-full-lines --strict-whitespace

// CHECK:{{^}}def main_first ;
// CHECK-NEXT:  def a_first ;
// CHECK-NEXT:def b ;
// CHECK-NEXT:def a_last ;
// CHECK-NEXT:def main_middle ;
// CHECK-NEXT:def b ;
// CHECK-NEXT:def from_b ;
// CHECK-NEXT:  def a_first ;
// CHECK-NEXT:def b ;
// CHECK-NEXT:def a_last ;
// CHECK-NEXT:               def main_last ;
// CHECK-NOT:{{.}}

//--- main.td
def main_first; include "a.td"
def main_middle;
  include "b.td"
#ifdef FROM_B
def from_b;
#endif
include "a.td" def main_last;

//--- a.td
  def a_first;
include "b.td"
def a_last;

//--- b.td
#ifndef B_GUARD
#define B_GUARD
#define FROM_B
#endif
def b;
