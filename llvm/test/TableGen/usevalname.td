// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

class Instr<list<dag> pat> {
  list<dag> Pattern = pat;
}

class Reg {
  int a = 3;
}

def VR128 : Reg;
def mem_frag;
def set;
def addr;
def shufp : Reg;

multiclass shuffle<Reg RC> {
  def rri : Instr<[(set RC:$dst, (shufp:$src3
                                       RC:$src1, RC:$src2))]>;
}

// CHECK: shufp:$src3
defm ADD : shuffle<VR128>;
