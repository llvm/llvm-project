// Test that multiline, nested, comments work correctly.
//
// RUN: llvm-tblgen < %s
// XFAIL: vg_leak

/* Foo
  bar
  /* 
  blah
  */

  stuff
  */

def x;
