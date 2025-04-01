// Tests that ProcessModID.m_memory_id is not bumped when evaluating expressions without side effects.

// RUN: %build %s -o %t
// RUN: %lldb %t \
// RUN:   -o "settings set target.process.process-state-tracks-memory-cache false" \
// RUN:   -o "run" \
// RUN:   -o "process dump-mod-id" \
// RUN:   -o "expr x.i != 42" \
// RUN:   -o "process dump-mod-id" \
// RUN:   -o "expr x.get()" \
// RUN:   -o "process dump-mod-id" \
// RUN:   -o "expr x.i = 10" \
// RUN:   -o "process dump-mod-id" \
// RUN:   -o "continue" \
// RUN:   -o "process dump-mod-id" \
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

// CHECK-LABEL: process dump-mod-id
// CHECK: m_stop_id: 2
// CHECK: m_memory_id: 0

// CHECK-LABEL: expr x.i != 42
// IDs are not changed when executing simple expressions

// CHECK-LABEL: process dump-mod-id
// CHECK: m_stop_id: 2
// CHECK: m_memory_id: 0

// CHECK-LABEL: expr x.get()
// Expression causes ID to be bumped because LLDB has to execute function

// CHECK-LABEL: process dump-mod-id
// CHECK: m_stop_id: 3
// CHECK: m_memory_id: 1

// CHECK-LABEL: expr x.i = 10
// Expression causes MemoryID to be bumped because LLDB writes to non-cache memory

// CHECK-LABEL: process dump-mod-id
// CHECK: m_stop_id: 3
// CHECK: m_memory_id: 2

// CHECK-LABEL: continue
// Continue causes StopID to be bumped because process is resumed

// CHECK-LABEL: process dump-mod-id
// CHECK: m_stop_id: 4
// CHECK: m_memory_id: 2
