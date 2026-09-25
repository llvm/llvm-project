// Check that a trivial C program can be compiled and run under ogre.
//
// RUN: %{cc} -c -o %t.o %s
// RUN: %{jit} -show-jit-result %t.o | FileCheck %s

// CHECK: JIT result: 0

int main(void) { return 0; }
