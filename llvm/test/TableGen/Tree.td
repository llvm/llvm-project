// This tests to make sure we can parse tree patterns.
// RUN: tblgen %s

class TreeNode;
class RegisterClass;

def set  : TreeNode;
def plus : TreeNode;
def imm  : TreeNode;
def R32  : RegisterClass;

class Inst<dag T> {
  dag Pattern = T;
}

def ADDrr32 : Inst<(set R32, (plus R32, R32))>;  // a = b + c
def ADDri32 : Inst<(set R32, (plus R32, imm))>;  // a = b + imm

