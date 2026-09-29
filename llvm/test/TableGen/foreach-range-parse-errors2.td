// RUN: not llvm-tblgen %s 2>&1 | FileCheck -DFILE=%s %s

class ConstantsImpl {
  int NegOne = -1;
}

def Constants : ConstantsImpl;

// CHECK: [[FILE]]:[[@LINE+2]]:38: error: invalid range, cannot be negative
// CHECK: [[FILE]]:[[@LINE+1]]:38: error: expected declaration in for
foreach Index = 0 - Constants.NegOne in {

}
