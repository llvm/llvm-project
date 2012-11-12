// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

class XD { bits<4> Prefix = 11; }
// CHECK: Prefix = { 1, 1, 0, 0 };
class XS { bits<4> Prefix = 12; }
class VEX { bit hasVEX_4VPrefix = 1; }

def xd : XD;

class BaseI {
  bits<4> Prefix = 0;
  bit hasVEX_4VPrefix = 0;
}

class I<bits<4> op> : BaseI {
  bits<4> opcode = op;
  int val = !if(!eq(Prefix, xd.Prefix), 7, 21);
  int check = !if(hasVEX_4VPrefix, 0, 10);
}

multiclass R {
  def rr : I<4>;
}

multiclass M {
  def rm : I<2>;
}

multiclass Y {
  defm SS : R, M, XD;
// CHECK: Prefix = { 1, 1, 0, 0 };
// CHECK: Prefix = { 1, 1, 0, 0 };
  defm SD : R, M, XS;
}

// CHECK: int check = 0;
defm Instr : Y, VEX;


// Anonymous defm.

multiclass SomeAnonymous<int x> {
  def rm;
  def mr;
}

// These multiclasses shouldn't conflict.
defm : SomeAnonymous<1>;
defm : SomeAnonymous<2>;