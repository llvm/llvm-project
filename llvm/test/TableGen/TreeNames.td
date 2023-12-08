// This tests to make sure we can parse tree patterns with names.
// RUN: llvm-tblgen %s
// XFAIL: vg_leak

class TreeNode;
class RegisterClass;

def set  : TreeNode;
def plus : TreeNode;
def imm  : TreeNode;
def R32  : RegisterClass;

class Inst<dag T> {
  dag Pattern = T;
}

def ADDrr32 : Inst<(set R32, (plus R32:$A, R32:$def))>;

