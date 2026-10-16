// RUN: not mlir-opt %s --mlir-disable-threading \
// RUN:   --mlir-debug-counter=unique-tag-for-my-action-skip=-1n 2>&1 \
// RUN: | FileCheck %s --check-prefix=BADNUM
//
// RUN: not mlir-opt %s --mlir-disable-threading \
// RUN:   --mlir-debug-counter=unique-tag-for-my-action-skip 2>&1 \
// RUN: | FileCheck %s --check-prefix=NOEQ
//
// RUN: not mlir-opt %s --mlir-disable-threading \
// RUN:   --mlir-debug-counter=unique-tag-for-my-action=-1 2>&1 \
// RUN: | FileCheck %s --check-prefix=BADSFX

func.func @foo() {
  return
}

// BADNUM-NOT: LLVM ERROR
// BADNUM-NOT: Stack dump:
// BADNUM: {{.*}}: for the {{-+}}mlir-debug-counter option: expected DebugCounter counter value to be numeric, but got `-1n`

// NOEQ-NOT: LLVM ERROR
// NOEQ-NOT: Stack dump:
// NOEQ: {{.*}}: for the {{-+}}mlir-debug-counter option: expected DebugCounter argument to have an `=` separating the counter name and value, but the provided argument was: `unique-tag-for-my-action-skip`

// BADSFX-NOT: LLVM ERROR
// BADSFX-NOT: Stack dump:
// BADSFX: {{.*}}: for the {{-+}}mlir-debug-counter option: expected DebugCounter counter name to end with either `-skip` or `-count`, but got `unique-tag-for-my-action`
