// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// Tests evaluation of !foreach operator.

def d0;
def d1;
def d2;
def d3;
def d4;

class D<dag d> {
  int tmp;
  dag r1 = !foreach(tmp, d, !subst(d1, d0, !subst(d2, d0,
                                           !subst(d3, d0,
                                           !subst(d4, d0, tmp)))));
  dag tmp2;
  list<dag> dl = [d];
  list<dag> r2 = !foreach(tmp2, dl,
                          !foreach(tmp, tmp2, !subst(d1, d0,
                                              !subst(d2, d0,
                                              !subst(d3, d0,
                                              !subst(d4, d0, tmp))))));
}

// CHECK-LABEL: def d
// CHECK: dag r1 = (d0 d0, d0, d0, d0);
// CHECK: list<dag> r2 = [(d0 d0, d0, d0, d0)];
def d : D <(d0 d1, d2, d3, d4)>;

class I<list<int> i> {
  int tmp;
  list<int> r1 = !foreach(tmp, i, !add(3, !add(4, tmp)));

  list<int> tmp2;
  list<list<int>> li = [i];
  list<list<int>> r2 = !foreach(tmp2, li,
                                !foreach(tmp, tmp2, !add(3, !add(4, tmp))));
}

// CHECK-LABEL: def i
// CHECK: list<int> r1 = [8, 9, 10];
// CHECK: list<list<int>> r2 = [{{[[]}}8, 9, 10]];
def i : I<[1,2,3]>;

class Tmp {
  dag t0;
  int t1;
}
def tmp: Tmp;

class J0<list<dag> pattern> {
  list<dag> Pattern = pattern;
}
class J1<dag pattern>
      : J0<[!foreach(tmp.t1, pattern, !subst(d1, d0,
                                      !subst(d2, d0,
                                      !subst(d3, d0,
                                      !subst(d4, d0, tmp.t1)))))]>;
class J2<list<dag> patterns>
      : J0<!foreach(tmp.t0, patterns,
                    !foreach(tmp.t1, tmp.t0, !subst(d1, d0,
                                             !subst(d2, d0,
                                             !subst(d3, d0,
                                             !subst(d4, d0, tmp.t1))))))>;
// CHECK-LABEL: def j1
// CHECK: list<dag> Pattern = [(d0 d0:$dst, (d0 d0:$src1))];
def j1 : J1< (d1 d2:$dst, (d3 d4:$src1))>;
// CHECK-LABEL: def j2
// CHECK: list<dag> Pattern = [(d0 d0:$dst, (d0 d0:$src1))];
def j2 : J2< [(d1 d2:$dst, (d3 d4:$src1))]>;

