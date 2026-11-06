// Check the exact output format of -E. Comments are removed, tokens on the
// same line are separated by a single space, and each output line is indented
// to the original column of its first token. Tokens are otherwise copied
// verbatim, including multi-line tokens like code blocks.

// RUN: llvm-tblgen -E %s | FileCheck %s --match-full-lines --strict-whitespace

// Check that the preprocessed output defines the same records as the original.

// RUN: llvm-tblgen -E %s -o %t.td
// RUN: llvm-tblgen %s -o %t1
// RUN: llvm-tblgen %t.td -o %t2
// RUN: cmp %t1 %t2

// Check that input with no tokens gives empty output.

// RUN: echo '// Only a comment.' | llvm-tblgen -E | count 0

// CHECK:{{^}}def ops ;
// CHECK-NEXT:def i32 ;
// CHECK-NEXT:class C < int n > {
// CHECK-NEXT:  bits < 8 > B = 0b10100101 ;
// CHECK-NEXT:  bits < 4 > Lo = B { 0 -3 } ;
// CHECK-NEXT:  bits < 4 > Hi = B { 7 ... 4 } ;
// CHECK-NEXT:  list < int > L = [ -1 , +2 , 0x1F , 0b101 ] ;
// CHECK-NEXT:  string S = "a\"b\\c" # "d" ;
// CHECK-NEXT:  code K = [{
// CHECK-NEXT:    multi
// CHECK-NEXT:      line }] ;
// CHECK-NEXT:  int I = !if ( !eq ( n , 0 ) , 1 , !add ( n , -1 ) ) ;
// CHECK-NEXT:  dag D = ( ops $x , i32 : $y ) ;
// CHECK-NEXT:  bit Q = ? ;
// CHECK-NEXT:}
// CHECK-NEXT:defvar V = 5 ;
// CHECK-NEXT:def X # V : C < V > ;
// CHECK-NEXT:foreach i = 1 -2 in def Y # i : C < i > ;
// CHECK-NEXT:def 1a : C < 0 > ;
// CHECK-NEXT:def Z : C < 0 > { let I = 3 ; }
// CHECK-NOT:{{.}}

def ops;
def i32; // Trailing comment.
class C<int n> {
  bits<8> B = 0b10100101;
  bits<4> Lo = B{0-3};
  bits<4> Hi = B{7...4};
  list<int> L = [-1,+2,  0x1F,   0b101];
  string S = "a\"b\\c" # "d";
  code K = [{
    multi
      line }];

  /* Blank lines and
     block comments are removed. */

  int I = !if(!eq(n, 0), 1, !add(n, -1));
  dag D = (ops $x, i32:$y);
  bit Q = ?;
}
defvar V = 5;
def X#V : C<V>;
foreach i = 1-2 in def Y#i : C<i>;
def 1a : C<0>;
def Z : C<0>{let I=3;}
