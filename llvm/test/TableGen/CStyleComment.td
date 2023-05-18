// Test that multiline, nested, comments work correctly.
//
// RUN: llvm-tblgen < %s

/* Foo
  bar
  /* 
  blah
  */

  stuff
  */

def x;
