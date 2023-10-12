// RUN: llvm-tblgen %s -o -  2>&1 >/dev/null | FileCheck %s -DFILE=%s

def op;
class A {
  string A = "some text";
  dag X =(op op);
}
def a : A;
// CHECK: [[FILE]]:[[@LINE+1]]:1: warning: dump message is not a string - use `!repr`
dump a;
