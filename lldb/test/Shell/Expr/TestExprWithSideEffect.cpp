// Tests evaluating expressions with side effects.
// Applied side effect should be visible to the debugger.

// RUN: %build %s -o %t
// RUN: %lldb %t -o run \
// RUN:   -o "frame variable x" \
// RUN:   -o "expr x.inc()" \
// RUN:   -o "frame variable x" \
// RUN:   -o "continue" \
// RUN:   -o "frame variable x" \
// RUN:   -o "expr x.i = 10" \
// RUN:   -o "frame variable x" \
// RUN:   -o "continue" \
// RUN:   -o "frame variable x" \
// RUN:   -o "expr int $y = 11" \
// RUN:   -o "expr $y" \
// RUN:   -o "expr $y = 100" \
// RUN:   -o "expr $y" \
// RUN:   -o "continue" \
// RUN:   -o "expr $y" \
// RUN:   -o exit | FileCheck %s -dump-input=fail

class X
{
  int i = 0;
public:
  void inc()
  {
    ++i;
  }
};

int main()
{
  X x;
  x.inc();

  __builtin_debugtrap();
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



// CHECK-LABEL: expr int $y = 11
// CHECK-LABEL: expr $y
// CHECK: (int) $y = 11

// CHECK-LABEL: expr $y = 100
// CHECK: (int) $1 = 100

// CHECK-LABEL: expr $y
// CHECK: (int) $y = 100

// CHECK-LABEL: continue
// CHECK-LABEL: expr $y
// CHECK: (int) $y = 100
