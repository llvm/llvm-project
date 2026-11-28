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

// Check more than one include on a single line, and multi-line tokens before,
// inside and after included files. A token that follows a multi-line token on
// the same line stays on the same output line, even if there are includes in
// between, as long as they do not output any tokens.

// RUN: llvm-tblgen -E -I %t %t/multi.td | FileCheck %s --check-prefix=MULTI --match-full-lines --strict-whitespace

// MULTI:{{^}}def p ;
// MULTI-NEXT:def c ; code cc = [{
// MULTI-NEXT:  c }] ;
// MULTI-NEXT:                      def q ;
// MULTI-NEXT:def c ; code cc = [{
// MULTI-NEXT:  c }] ;
// MULTI-NEXT:                                            def r ;
// MULTI-NEXT:code x = [{
// MULTI-NEXT:  x }] ; def s ;
// MULTI-NEXT:def c ; code cc = [{
// MULTI-NEXT:  c }] ;
// MULTI-NEXT:                                                                    def t ;
// MULTI-NEXT:def c ; code cc = [{
// MULTI-NEXT:  c }] ;
// MULTI-NEXT:def c ; code cc = [{
// MULTI-NEXT:  c }] ;
// MULTI-NEXT:def u ;
// MULTI-NOT:{{.}}

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

//--- multi.td
def p; include "c.td" def q; include "c.td" def r;
code x = [{
  x }]; include "empty.td" def s; include "empty.td" include "c.td" def t;
include "c.td" include "empty.td" include "c.td"
def u;

//--- c.td
def c; code cc = [{
  c }];

//--- empty.td
