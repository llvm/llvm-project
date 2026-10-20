// RUN: llvm-tblgen %s | FileCheck %s

// Operator name "frg" in one_frag:$frg was not retained during !subst.

def one_frag;
def FPR32;
def ops;
def node;
def GPR;
def cond;
def set;
def FPR32_NEW;
def a {
  dag d = (set FPR32:$dst, (one_frag:$frg FPR32:$a, FPR32:$b));
  dag n = !foreach(i, d, !subst(FPR32, FPR32_NEW, i));
}

// CHECK: dag n = (set FPR32_NEW:$dst, (one_frag:$frg FPR32_NEW:$a, FPR32_NEW:$b));
