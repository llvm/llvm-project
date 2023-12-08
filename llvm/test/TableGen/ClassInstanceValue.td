// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

class Struct<int i> {
  int I = !shl(i, 1);
  int J = !shl(I, 1);
}

class Class<Struct s> {
    int Class_J = s.J;
}

multiclass MultiClass<int i> {
  def Def : Class<Struct<i>>;
// CHECK: Class_J = 8
// CHECK-NOT: Class_J = !shl(I, 1)
}

defm Defm : MultiClass<2>;
