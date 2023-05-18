// RUN: llvm-tblgen %s | FileCheck %s
// RUN: not llvm-tblgen -DERROR %s 2>&1 | FileCheck --check-prefix=ERROR %s

def op;
def otherop;

def test {
  // CHECK: dag d = (? "hello":$world);
  dag d = (? "hello":$world);

  // CHECK: dag undefNeither = (op 1, 2);
  dag undefNeither = !con((op 1), (op 2));
  // CHECK: dag undefFirst = (op 1, 2);
  dag undefFirst   = !con((?  1), (op 2));
  // CHECK: dag undefSecond = (op 1, 2);
  dag undefSecond  = !con((op 1), (?  2));
  // CHECK: dag undefBoth = (? 1, 2);
  dag undefBoth    = !con((?  1), (?  2));

#ifdef ERROR
  // ERROR: Concatenated Dag operators do not match: '(op 1)' vs. '(otherop 2)'
  dag mismatch     = !con((op 1), (otherop 2));
#endif
}
