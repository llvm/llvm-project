// RUN: not llvm-tblgen %s 2>&1 | FileCheck -DFILE=%s %s
// Make sure there is no crash on undefined variable

// CHECK: [[FILE]]:[[@LINE+2]]:17: error: Variable not defined: 'foo'
// CHECK: [[FILE]]:[[@LINE+1]]:21: error: expected declaration in for
foreach Index = foo in {
  def arst#Index;
}

