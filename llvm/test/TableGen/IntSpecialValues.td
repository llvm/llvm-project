// RUN: llvm-tblgen %s | FileCheck %s

def TestRecord {
  // CHECK: int X = 9223372036854775807;
  int X = 0x7FFFFFFFFFFFFFFF;
  // CHECK: int Y = 9223372036854775806;
  int Y = 0x7FFFFFFFFFFFFFFE;
}
