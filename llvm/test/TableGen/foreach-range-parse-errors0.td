// RUN: not llvm-tblgen %s 2>&1 | FileCheck -DFILE=%s %s

class ConstantsImpl {
  int Zero = 0;
  int One = 1;
  int Two = 2;
  int Three = 3;
  int Five = 5;
}

def Constants : ConstantsImpl;

// CHECK-NOT:  error: Unknown or reserved token when parsing a value
// CHECK: [[FILE]]:[[@LINE+3]]:22: error: Unknown or reserved token when parsing a value
// CHECK: [[FILE]]:[[@LINE+2]]:22: error: expected integer value as end of range
// CHECK: [[FILE]]:[[@LINE+1]]:22: error: expected declaration in for
foreach Index = 0 -  in {

}
