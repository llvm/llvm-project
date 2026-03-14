// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// Tests evaluation of !foreach operator.

def d0;
def d1;
def d2;
def d3;
def d4;

class D<dag d> {
  dag r1 = !foreach(tmp, d, !subst(d1, d0, !subst(d2, d0,
                                           !subst(d3, d0,
                                           !subst(d4, d0, tmp)))));
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
  list<int> r1 = !foreach(tmp, i, !add(3, !add(4, tmp)));

  list<list<int>> li = [i];
  list<list<int>> r2 = !foreach(tmp2, li,
                                !foreach(tmp, tmp2, !add(3, !add(4, tmp))));
}

// CHECK-LABEL: def i
// CHECK: list<int> r1 = [8, 9, 10];
// CHECK: list<list<int>> r2 = [{{[[]}}8, 9, 10]];
def i : I<[1,2,3]>;

class J0<list<dag> pattern> {
  list<dag> Pattern = pattern;
}
class J1<dag pattern>
      : J0<[!foreach(tmp, pattern, !subst(d1, d0,
                                   !subst(d2, d0,
                                   !subst(d3, d0,
                                   !subst(d4, d0, tmp)))))]>;
class J2<list<dag> patterns>
      : J0<!foreach(t0, patterns,
                    !foreach(t1, t0, !subst(d1, d0,
                                     !subst(d2, d0,
                                     !subst(d3, d0,
                                     !subst(d4, d0, t1))))))>;
// CHECK-LABEL: def j1
// CHECK: list<dag> Pattern = [(d0 d0:$dst, (d0 d0:$src1))];
def j1 : J1< (d1 d2:$dst, (d3 d4:$src1))>;
// CHECK-LABEL: def j2
// CHECK: list<dag> Pattern = [(d0 d0:$dst, (d0 d0:$src1))];
def j2 : J2< [(d1 d2:$dst, (d3 d4:$src1))]>;

