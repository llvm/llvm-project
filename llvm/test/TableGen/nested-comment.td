// RUN: llvm-tblgen < %s
// XFAIL: vg_leak

/* foo

/ foo

     /*NoReg*/, baz
     
      
      */

def X;
