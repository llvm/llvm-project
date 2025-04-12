// Tests that ProcessModID.m_memory_id is not bumped when evaluating expressions without side effects.

// REQUIRES: target-windows
// Due to different implementations exact numbers (m_stop_id) are different on different OSs. So we lock this test to specific platform.

// RUN: %build %s -o %t
// RUN: %lldb %t \
// RUN:   -o "settings set target.process.track-memory-cache-changes false" \
// RUN:   -o "run" \
// RUN:   -o "process status -d" \
// RUN:   -o "expr x.i != 42" \
// RUN:   -o "process status -d" \
// RUN:   -o "expr x.get()" \
// RUN:   -o "process status -d" \
// RUN:   -o "expr x.i = 10" \
// RUN:   -o "process status -d" \
// RUN:   -o "continue" \
// RUN:   -o "process status -d" \
// RUN:   -o "exit" | FileCheck %s -dump-input=fail

class X {
  int i = 0;

public:
  int get() { return i; }
};

int main() {
  X x;
  x.get();

  __builtin_debugtrap();
  __builtin_debugtrap();
  return 0;
}

// CHECK-LABEL: process status -d
// CHECK: m_stop_id: 2
// CHECK: m_memory_id: 0

// CHECK-LABEL: expr x.i != 42
// IDs are not changed when executing simple expressions

// CHECK-LABEL: process status -d
// CHECK: m_stop_id: 2
// CHECK: m_memory_id: 0

// CHECK-LABEL: expr x.get()
// Expression causes ID to be bumped because LLDB has to execute function

// CHECK-LABEL: process status -d
// CHECK: m_stop_id: 3
// CHECK: m_memory_id: 1

// CHECK-LABEL: expr x.i = 10
// Expression causes MemoryID to be bumped because LLDB writes to non-cache memory

// CHECK-LABEL: process status -d
// CHECK: m_stop_id: 3
// CHECK: m_memory_id: 2

// CHECK-LABEL: continue
// Continue causes StopID to be bumped because process is resumed

// CHECK-LABEL: process status -d
// CHECK: m_stop_id: 4
// CHECK: m_memory_id: 2
