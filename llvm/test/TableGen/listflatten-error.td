// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s  -DFILE=%s

// CHECK: [[FILE]]:[[@LINE+2]]:33: error: expected list type argument in unary operator
class Flatten<int A> {
    list<int> F = !listflatten(A);
}
