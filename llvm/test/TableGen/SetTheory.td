// Test evaluation of set operations in dags.
// RUN: llvm-tblgen -print-sets %s | FileCheck %s
// XFAIL: vg_leak
//
// The -print-sets driver configures a primitive SetTheory instance that
// understands these sets:

class Set<dag d> {
  dag Elements = d;
}

// It prints all Set instances and their ordered set interpretation.

// Define some elements.
def a;
def b;
def c;
def d;

// The 'add' operator evaluates and concatenates its arguments.
def add;
def S0a : Set<(add)>;
def S0b : Set<(add a)>;
def S0c : Set<(add a, b)>;
def S0d : Set<(add b, a)>;
def S0e : Set<(add a, a)>;
def S0f : Set<(add a, a, b, a, c, b, d, a)>;
def S0g : Set<(add b, a, b)>;
// CHECK: S0a = [ ]
// CHECK: S0b = [ a ]
// CHECK: S0c = [ a b ]
// CHECK: S0d = [ b a ]
// CHECK: S0e = [ a ]
// CHECK: S0f = [ a b c d ]
// CHECK: S0g = [ b a ]

// Defs of Set class expand into their elements.
// Mixed sets and elements are flattened.
def S1a : Set<(add S0a)>;
def S1b : Set<(add S0a, S0a)>;
def S1c : Set<(add S0d, S0f)>;
def S1d : Set<(add d, S0d, S0f)>;
// CHECK: S1a = [ ]
// CHECK: S1b = [ ]
// CHECK: S1c = [ b a c d ]
// CHECK: S1d = [ d b a c ]

// The 'sub' operator returns the first argument with the following arguments
// removed.
def sub;
def S2a : Set<(sub S1a, S1c)>;
def S2b : Set<(sub S1c, S1d)>;
def S2c : Set<(sub S1c, b)>;
def S2d : Set<(sub S1c, S0c)>;
def S2e : Set<(sub S1c, S2d)>;
// CHECK: S2a = [ ]
// CHECK: S2b = [ ]
// CHECK: S2c = [ a c d ]
// CHECK: S2d = [ c d ]
// CHECK: S2e = [ b a ]

// The 'and' operator intersects two sets. The result has the same order as the
// first argument.
def and;
def S3a : Set<(and S2d, S2e)>;
def S3b : Set<(and S2d, S1d)>;
// CHECK: S3a = [ ]
// CHECK: S3b = [ c d ]

// The 'shl' operator removes the first N elements.
def shl;
def S4a : Set<(shl S0f, 0)>;
def S4b : Set<(shl S0f, 1)>;
def S4c : Set<(shl S0f, 3)>;
def S4d : Set<(shl S0f, 4)>;
def S4e : Set<(shl S0f, 5)>;
// CHECK: S4a = [ a b c d ]
// CHECK: S4b = [ b c d ]
// CHECK: S4c = [ d ]
// CHECK: S4d = [ ]
// CHECK: S4e = [ ]

// The 'trunc' operator truncates after the first N elements.
def trunc;
def S5a : Set<(trunc S0f, 0)>;
def S5b : Set<(trunc S0f, 1)>;
def S5c : Set<(trunc S0f, 3)>;
def S5d : Set<(trunc S0f, 4)>;
def S5e : Set<(trunc S0f, 5)>;
// CHECK: S5a = [ ]
// CHECK: S5b = [ a ]
// CHECK: S5c = [ a b c ]
// CHECK: S5d = [ a b c d ]
// CHECK: S5e = [ a b c d ]

// The 'rotl' operator rotates left, but also accepts a negative shift.
def rotl;
def S6a : Set<(rotl S0f, 0)>;
def S6b : Set<(rotl S0f, 1)>;
def S6c : Set<(rotl S0f, 3)>;
def S6d : Set<(rotl S0f, 4)>;
def S6e : Set<(rotl S0f, 5)>;
def S6f : Set<(rotl S0f, -1)>;
def S6g : Set<(rotl S0f, -4)>;
def S6h : Set<(rotl S0f, -5)>;
// CHECK: S6a = [ a b c d ]
// CHECK: S6b = [ b c d a ]
// CHECK: S6c = [ d a b c ]
// CHECK: S6d = [ a b c d ]
// CHECK: S6e = [ b c d a ]
// CHECK: S6f = [ d a b c ]
// CHECK: S6g = [ a b c d ]
// CHECK: S6h = [ d a b c ]

// The 'rotr' operator rotates right, but also accepts a negative shift.
def rotr;
def S7a : Set<(rotr S0f, 0)>;
def S7b : Set<(rotr S0f, 1)>;
def S7c : Set<(rotr S0f, 3)>;
def S7d : Set<(rotr S0f, 4)>;
def S7e : Set<(rotr S0f, 5)>;
def S7f : Set<(rotr S0f, -1)>;
def S7g : Set<(rotr S0f, -4)>;
def S7h : Set<(rotr S0f, -5)>;
// CHECK: S7a = [ a b c d ]
// CHECK: S7b = [ d a b c ]
// CHECK: S7c = [ b c d a ]
// CHECK: S7d = [ a b c d ]
// CHECK: S7e = [ d a b c ]
// CHECK: S7f = [ b c d a ]
// CHECK: S7g = [ a b c d ]
// CHECK: S7h = [ b c d a ]

// The 'decimate' operator picks every N'th element.
def decimate;
def e0;
def e1;
def e2;
def e3;
def e4;
def e5;
def e6;
def e7;
def e8;
def e9;
def E : Set<(add e0, e1, e2, e3, e4, e5, e6, e7, e8, e9)>;
def S8a : Set<(decimate E, 3)>;
def S8b : Set<(decimate E, 9)>;
def S8c : Set<(decimate E, 10)>;
def S8d : Set<(decimate (rotl E, 1), 2)>;
def S8e : Set<(add (decimate E, 2), (decimate (rotl E, 1), 2))>;
// CHECK: S8a = [ e0 e3 e6 e9 ]
// CHECK: S8b = [ e0 e9 ]
// CHECK: S8c = [ e0 ]
// CHECK: S8d = [ e1 e3 e5 e7 e9 ]
// CHECK: S8e = [ e0 e2 e4 e6 e8 e1 e3 e5 e7 e9 ]

// The 'sequence' operator finds a sequence of records from their name.
def sequence;
def S9a : Set<(sequence "e%u", 3, 7)>;
def S9b : Set<(sequence "e%u", 7, 3)>;
def S9c : Set<(sequence "e%u", 0, 0)>;
def S9d : Set<(sequence "S%ua", 7, 9)>;
def S9e : Set<(sequence "e%u", 3, 6, 2)>;
// CHECK: S9a = [ e3 e4 e5 e6 e7 ]
// CHECK: S9b = [ e7 e6 e5 e4 e3 ]
// CHECK: S9c = [ e0 ]
// CHECK: S9d = [ a b c d e0 e3 e6 e9 e4 e5 e7 ]
// CHECK: S9e = [ e3 e5 ]

// The 'interleave' operator is almost the inverse of 'decimate'.
def interleave;
def T0a : Set<(interleave S9a, S9b)>;
def T0b : Set<(interleave S8e, S8d)>;
// CHECK: T0a = [ e3 e7 e4 e6 e5 ]
// CHECK: T0b = [ e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 ]
