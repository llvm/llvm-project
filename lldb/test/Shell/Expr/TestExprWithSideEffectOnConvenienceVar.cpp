// Tests evaluating expressions with side effects on convenience variable.
// Applied side effect should be visible to the debugger.

// UNSUPPORTED: system-windows

// RUN: %build %s -o %t
// RUN: %lldb %t \
// RUN:   -o "settings set target.process.track-memory-cache-changes false" \
// RUN:   -o "run" \
// RUN:   -o "expr int \$y = 11" \
// RUN:   -o "expr \$y" \
// RUN:   -o "expr \$y = 100" \
// RUN:   -o "expr \$y" \
// RUN:   -o "continue" \
// RUN:   -o "expr \$y" \
// RUN:   -o "expr X \$mine = {100, 200}" \
// RUN:   -o "expr \$mine.a = 300" \
// RUN:   -o "expr \$mine" \
// RUN:   -o "exit" | FileCheck %s -dump-input=fail

struct X {
  int a;
  int b;
};

int main() {
  X x;

  __builtin_debugtrap();
  __builtin_debugtrap();
  return 0;
}

// CHECK-LABEL: expr int $y = 11
// CHECK-LABEL: expr $y
// CHECK: (int) $y = 11

// CHECK-LABEL: expr $y = 100
// CHECK: (int) $0 = 100

// CHECK-LABEL: expr $y
// CHECK: (int) $y = 100

// CHECK-LABEL: continue
// CHECK-LABEL: expr $y
// CHECK: (int) $y = 100

// CHECK-LABEL: expr X $mine = {100, 200}
// CHECK-LABEL: expr $mine.a = 300
// CHECK: (int) $1 = 300
// CHECK-LABEL: expr $mine
// CHECK: (X) $mine = (a = 300, b = 200)
