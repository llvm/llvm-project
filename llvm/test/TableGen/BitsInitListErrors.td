// RUN: not llvm-tblgen -DNOT_A_BIT %s 2>&1 | FileCheck %s -check-prefix=NOT_A_BIT
// RUN: not llvm-tblgen -DAFTER_BITS %s 2>&1 | FileCheck %s -check-prefix=AFTER_BITS
// RUN: not llvm-tblgen -DNO_RBRACE %s 2>&1 | FileCheck %s -check-prefix=NO_RBRACE

// Check errors when parsing a bit list value like { a, b, c }.

#ifdef NOT_A_BIT
def A {
  // NOT_A_BIT: :[[@LINE+1]]:15: error: Element #1 ("x") is not convertable to a bit
  bits<2> X = { 1, "x" };
}
#endif

// The element number counts elements of the list, not bits.
#ifdef AFTER_BITS
def B {
  bits<2> Two = 0b10;
  // AFTER_BITS: :[[@LINE+1]]:15: error: Element #3 ([1]) is not convertable to a bit
  bits<6> X = { 0b10, Two, 1, [1] };
}
#endif

#ifdef NO_RBRACE
def C {
  // NO_RBRACE: :[[@LINE+1]]:22: error: expected '}' at end of bit list value
  bits<2> X = { 1, 0 ;
}
#endif
