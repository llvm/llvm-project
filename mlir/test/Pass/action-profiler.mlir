// RUN: mlir-opt %s --profile-actions-to=- -test-stats-pass -test-module-pass | FileCheck %s

// CHECK: {"name": "pass-execution", "cat": "PERF", "ph": "B", "pid": 0, "tid": {{[0-9]+}}, "ts": {{[0-9]+}}, "args": {"desc": "`pass-execution` running `{{.*}}TestStatisticPass` on Operation `builtin.module`"}},
// CHECK-NEXT: {"name": "pass-execution", "cat": "PERF", "ph": "E", "pid": 0, "tid": {{[0-9]+}}, "ts": {{[0-9]+}}},
// CHECK-NEXT: {"name": "pass-execution", "cat": "PERF", "ph": "B", "pid": 0, "tid": {{[0-9]+}}, "ts": {{[0-9]+}}, "args": {"desc": "`pass-execution` running `{{.*}}TestModulePass` on Operation `builtin.module`"}},
// CHECK-NEXT: {"name": "pass-execution", "cat": "PERF", "ph": "E", "pid": 0, "tid": {{[0-9]+}}, "ts": {{[0-9]+}}}
