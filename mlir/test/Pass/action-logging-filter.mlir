// Run the canonicalize on each function, use the --log-mlir-actions-filter= option
// to filter which action should be logged.

func.func @a() {
    return
}

func.func @b() {
    return
}

func.func @c() {
    return
}

////////////////////////////////////
/// 1. All actions should be logged.

// RUN: mlir-opt %s --log-actions-to=- -pass-pipeline="builtin.module(func.func(test-stats-pass))" -o %t --mlir-disable-threading | FileCheck %s
// Specify the current file as filter, expect to see all actions.
// RUN: mlir-opt %s --log-mlir-actions-filter=%s --log-actions-to=- -pass-pipeline="builtin.module(func.func(test-stats-pass))" -o %t --mlir-disable-threading | FileCheck %s

// CHECK: [thread {{.*}}] begins (no breakpoint) Action `pass-execution`  running `{{.*}}TestStatisticPass` on Operation `func.func` (func.func @a() {...}
// CHECK-NEXT: [thread {{.*}}] completed `pass-execution`
// CHECK-NEXT: [thread {{.*}}] begins (no breakpoint) Action `pass-execution`  running `{{.*}}TestStatisticPass` on Operation `func.func` (func.func @b() {...}
// CHECK-NEXT: [thread {{.*}}] completed `pass-execution`
// CHECK-NEXT: [thread {{.*}}] begins (no breakpoint) Action `pass-execution`  running `{{.*}}TestStatisticPass` on Operation `func.func` (func.func @c() {...}
// CHECK-NEXT: [thread {{.*}}] completed `pass-execution`

////////////////////////////////////
/// 2. No match

// Specify a non-existing file as filter, expect to see no actions.
// RUN: mlir-opt %s --log-mlir-actions-filter=foo.mlir --log-actions-to=- -pass-pipeline="builtin.module(func.func(test-stats-pass))" -o %t --mlir-disable-threading | FileCheck %s --check-prefix=CHECK-NONE --allow-empty
// Filter on a non-matching line, expect to see no actions.
// RUN: mlir-opt %s --log-mlir-actions-filter=%s:1 --log-actions-to=- -pass-pipeline="builtin.module(func.func(test-stats-pass))" -o %t --mlir-disable-threading | FileCheck %s --check-prefix=CHECK-NONE --allow-empty

// Invalid Filter
// CHECK-NONE-NOT: Canonicalizer

////////////////////////////////////
/// 3. Matching filters

// Filter the second function only
// RUN: mlir-opt %s --log-mlir-actions-filter=%s:8 --log-actions-to=- -pass-pipeline="builtin.module(func.func(test-stats-pass))" -o %t --mlir-disable-threading | FileCheck %s --check-prefix=CHECK-SECOND

// CHECK-SECOND-NOT: @a
// CHECK-SECOND-NOT: @c
// CHECK-SECOND: [thread {{.*}}] begins (no breakpoint) Action `pass-execution`  running `{{.*}}TestStatisticPass` on Operation `func.func` (func.func @b() {...}
// CHECK-SECOND-NEXT: [thread {{.*}}] completed `pass-execution`

// Filter the first and third functions
// RUN: mlir-opt %s --log-mlir-actions-filter=%s:4,%s:12 --log-actions-to=- -pass-pipeline="builtin.module(func.func(test-stats-pass))" -o %t --mlir-disable-threading | FileCheck %s  --check-prefix=CHECK-FIRST-THIRD

// CHECK-FIRST-THIRD-NOT: Canonicalizer
// CHECK-FIRST-THIRD: [thread {{.*}}] begins (no breakpoint) Action `pass-execution`  running `{{.*}}TestStatisticPass` on Operation `func.func` (func.func @a() {...}
// CHECK-FIRST-THIRD-NEXT: [thread {{.*}}] completed `pass-execution`
// CHECK-FIRST-THIRD-NEXT: [thread {{.*}}] begins (no breakpoint) Action `pass-execution`  running `{{.*}}TestStatisticPass` on Operation `func.func` (func.func @c() {...}
// CHECK-FIRST-THIRD-NEXT: [thread {{.*}}] completed `pass-execution`
// CHECK-FIRST-THIRD-NOT: Canonicalizer
