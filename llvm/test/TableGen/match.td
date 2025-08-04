// RUN: llvm-tblgen %s | FileCheck %s
// RUN: not llvm-tblgen -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s
// RUN: not llvm-tblgen -DERROR2 %s 2>&1 | FileCheck --check-prefix=ERROR2 %s
// RUN: not llvm-tblgen -DERROR3 %s 2>&1 | FileCheck --check-prefix=ERROR3 %s
// RUN: not llvm-tblgen -DERROR4 %s 2>&1 | FileCheck --check-prefix=ERROR4 %s
// XFAIL: vg_leak

def test {
  bit test0 = !match("123 abc ABC", "[0-9 a-z A-Z]+");
  bit test1 = !match("abc", "[0-9]+");
}

// CHECK-LABEL: def test {
// CHECK-NEXT:    bit test0 = 1;
// CHECK-NEXT:    bit test1 = 0;
// CHECK-NEXT:  }

#ifdef ERROR1
defvar error1 = !match(123, ".*");
// ERROR1: error: expected value of type 'string', got 'int'
#endif

#ifdef ERROR2
defvar error2 = !match("abc", 123);
// ERROR2: error: expected value of type 'string', got 'int'
#endif

#ifdef ERROR3
defvar error3 = !match("abc", "abc", "abc");
// ERROR3: error: expected two operands to operator
#endif

#ifdef ERROR4
defvar error4 = !match("abc", "([)]");
// ERROR4: error: invalid regex '([)]'
#endif
