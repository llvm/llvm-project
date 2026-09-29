// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(test-remark))' --remarks-filter-passed=category-1-passed --remark-policy=final -o /dev/null 2>&1 | FileCheck %s --implicit-check-not="remark:"
// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(test-remark))' --remarks-filter-passed=category-1-passed --remark-policy=final --mlir-disable-threading -o /dev/null 2>&1 | FileCheck %s --implicit-check-not="remark:"
// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(test-remark))' --remarks-filter-passed=category-1-passed --remark-policy=final --remark-format=yaml --remarks-output-file=%t.yaml -o /dev/null
// RUN: FileCheck --check-prefix=CHECK-YAML %s --implicit-check-not="--- !" < %t.yaml

// test-remark runs on each function in parallel and reports one passed remark
// per operation. The final policy prints them sorted by source position, which
// differs from both the IR order and any order the threads can report in, and
// is the same with and without threading. Remarks without a file position come
// last.

func.func @c() {
  return loc("b.c":1:1)
} loc("a.c":30:1)
func.func @a() {
  return loc("a.c":11:1)
} loc("a.c":10:1)
func.func @b() {
  return loc("n"("a.c":20:1))
} loc(fused["a.c":10:5, "x"])
func.func @d() {
  return loc(unknown)
} loc("a.c":25:1)

// CHECK:      a.c:10:1: remark: [Passed]
// CHECK-NEXT: a.c:10:5: remark: [Passed]
// CHECK-NEXT: a.c:11:1: remark: [Passed]
// CHECK-NEXT: a.c:20:1: remark: [Passed]
// CHECK-NEXT: a.c:25:1: remark: [Passed]
// CHECK-NEXT: a.c:30:1: remark: [Passed]
// CHECK-NEXT: b.c:1:1: remark: [Passed]
// CHECK-NEXT: <unknown>:0: remark: [Passed]

// The YAML serializer writes the same file position the printer shows, also
// for the fused and named locations.
// CHECK-YAML:      --- !Passed
// CHECK-YAML:      DebugLoc: { File: a.c, Line: 10, Column: 1 }
// CHECK-YAML:      --- !Passed
// CHECK-YAML:      DebugLoc: { File: a.c, Line: 10, Column: 5 }
// CHECK-YAML:      --- !Passed
// CHECK-YAML:      DebugLoc: { File: a.c, Line: 11, Column: 1 }
// CHECK-YAML:      --- !Passed
// CHECK-YAML:      DebugLoc: { File: a.c, Line: 20, Column: 1 }
// CHECK-YAML:      --- !Passed
// CHECK-YAML:      DebugLoc: { File: a.c, Line: 25, Column: 1 }
// CHECK-YAML:      --- !Passed
// CHECK-YAML:      DebugLoc: { File: a.c, Line: 30, Column: 1 }
// CHECK-YAML:      --- !Passed
// CHECK-YAML:      DebugLoc: { File: b.c, Line: 1, Column: 1 }
// CHECK-YAML:      --- !Passed
// CHECK-YAML:      DebugLoc: { File: '<unknown file>', Line: 0, Column: 0 }
