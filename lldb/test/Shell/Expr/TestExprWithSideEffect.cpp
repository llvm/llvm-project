// Tests evaluating expressions with side effects.
// Applied side effect should be visible to the debugger.

// RUN: %build %s -o %t
// RUN: %lldb %t \
// RUN:   -o "settings set target.process.track-memory-cache-changes false" \
// RUN:   -o "run" \
// RUN:   -o "frame variable x" \
// RUN:   -o "expr x.inc()" \
// RUN:   -o "frame variable x" \
// RUN:   -o "continue" \
// RUN:   -o "frame variable x" \
// RUN:   -o "expr x.i = 10" \
// RUN:   -o "frame variable x" \
// RUN:   -o "continue" \
// RUN:   -o "frame variable x" \
// RUN:   -o "exit" | FileCheck %s -dump-input=fail

class X {
  int i = 0;

public:
  void inc() { ++i; }
};

int main() {
  X x;
  x.inc();

  __builtin_debugtrap();
  __builtin_debugtrap();
  __builtin_debugtrap();
  return 0;
}

// CHECK-LABEL: frame variable x
// CHECK: (X) x = (i = 1)

// CHECK-LABEL: expr x.inc()
// CHECK-LABEL: frame variable x
// CHECK: (X) x = (i = 2)

// CHECK-LABEL: continue
// CHECK-LABEL: frame variable x
// CHECK: (X) x = (i = 2)

// CHECK-LABEL: expr x.i = 10
// CHECK: (int) $0 = 10

// CHECK-LABEL: frame variable x
// CHECK: (X) x = (i = 10)

// CHECK-LABEL: continue
// CHECK-LABEL: frame variable x
// CHECK: (X) x = (i = 10)
