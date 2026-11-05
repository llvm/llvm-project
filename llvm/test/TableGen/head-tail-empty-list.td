// RUN: not llvm-tblgen -DHEAD %s 2>&1 | FileCheck --check-prefix=HEAD %s
// RUN: not llvm-tblgen -DTAIL %s 2>&1 | FileCheck --check-prefix=TAIL %s
// RUN: not llvm-tblgen -DFILTER %s 2>&1 | FileCheck --check-prefix=FILTER %s

// A literal empty list is rejected by the parser, but a list that only
// resolves to an empty list later used to hit an assert in UnOpInit::Fold.

#ifdef HEAD
// HEAD: error: !head requires a non-empty list
def HeadDeferred {
  list<int> e = [];
  int h = !head(e);
}
#endif

#ifdef TAIL
// TAIL: error: !tail requires a non-empty list
def TailDeferred {
  list<int> e = [];
  list<int> t = !tail(e);
}
#endif

#ifdef FILTER
// FILTER: error: !head requires a non-empty list
class C<int N> {
  int h = !head(!filter(x, [1, 2], !gt(x, N)));
}
def D : C<5>;
#endif
