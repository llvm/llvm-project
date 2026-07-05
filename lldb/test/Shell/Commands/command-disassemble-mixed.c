// invalid mixed disassembly line

// XFAIL: target-windows

// RUN: %clang_host -g %s -o %t
// RUN: %lldb %t -o "dis -m -n main" -o "exit" | FileCheck %s

// CHECK: int main
// CHECK: int i
// CHECK-NOT: invalid mixed disassembly line
// CHECK: return 0;

int main(int argc, char **argv)
{
  int i;

  for (i=0; i < 10; ++i) ;

  return 0;
}
