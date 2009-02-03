// Test that multiline, nested, comments work correctly.
//
// RUN: tblgen < %s

/* Foo
  bar
  /* 
  blah
  */

  stuff
  */

def x;
