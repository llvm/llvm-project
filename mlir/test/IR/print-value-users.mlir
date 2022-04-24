// RUN: mlir-opt -allow-unregistered-dialect -mlir-print-value-users -split-input-file %s | FileCheck %s

module {
    // CHECK: %[[ARG0:.+]]: i32, %[[ARG1:.+]]: i32, %[[ARG2:.+]]: i32
    func @foo(%arg0: i32, %arg1: i32, %arg3: i32) -> i32 {
        // CHECK-NEXT: // %[[ARG0]] is used by %[[ARG0U1:.+]], %[[ARG0U2:.+]], %[[ARG0U3:.+]]
        // CHECK-NEXT: // %[[ARG1]] is used by %[[ARG1U1:.+]], %[[ARG1U2:.+]]
        // CHECK-NEXT: // %[[ARG2]] is unused
        // CHECK-NEXT: test.noop
        // CHECK-NOT: // unused
        "test.noop"() : () -> ()
        // When no result is produced, an id should be printed.
        // CHECK-NEXT: // id: %[[ARG0U3]]
        "test.no_result"(%arg0) {} : (i32) -> ()
        // Check for unused result.
        // CHECK-NEXT: %[[ARG0U2]] = 
        // CHECK-SAME: // unused
        %1 = "test.unused_result"(%arg0, %arg1) {} : (i32, i32) -> i32
        // Check that both users are printed.
        // CHECK-NEXT: %[[ARG0U1]] = 
        // CHECK-SAME: // users: %[[A:.+]]#0, %[[A]]#1
        %2 = "test.one_result"(%arg0, %arg1) {} : (i32, i32) -> i32
        // For multiple results, users should be grouped per result.
        // CHECK-NEXT: %[[A]]:2 = 
        // CHECK-SAME: // users: (%[[B:.+]], %[[C:.+]]), (%[[B]], %[[D:.+]])
        %3:2 = "test.many_results"(%2) {} : (i32) -> (i32, i32)
        // Two results are produced, but there is only one user.
        // CHECK-NEXT: // users:
        %7:2 = "test.many_results"() : () -> (i32, i32)
        // CHECK-NEXT: %[[C]] =
        // Result is used twice in next operation but it produces only one result.
        // CHECK-SAME: // user:
        %4 = "test.foo"(%3#0) {} : (i32) -> i32
        // CHECK-NEXT: %[[D]] =
        %5 = "test.foo"(%3#1, %4, %4) {} : (i32, i32, i32) -> i32
        // CHECK-NEXT: %[[B]] =
        // Result is not used in any other result but in two operations.
        // CHECK-SAME: // users:
        %6 = "test.foo"(%3#0, %3#1) {} : (i32, i32) -> i32
        "test.no_result"(%6) {} : (i32) -> ()
        "test.no_result"(%7#0) : (i32) -> ()
        return %6: i32
    }
}

// -----

module {
    // Check with nested operation.
    // CHECK: %[[CONSTNAME:.+]] = arith.constant
    %0 = arith.constant 42 : i32
    %test = "test.outerop"(%0) ({
        // CHECK: "test.innerop"(%[[CONSTNAME]]) : (i32) -> () // id: %
        "test.innerop"(%0) : (i32) -> ()
    // CHECK: (i32) -> i32 // users: %r, %s, %p, %p_0, %q
    }): (i32) -> i32

    // Check named results.
    // CHECK-NEXT: // users: (%u, %v), (unused), (%u, %v, %r, %s)
    %p:2, %q = "test.custom_result_name"(%test) {names = ["p", "p", "q"]} : (i32) -> (i32, i32, i32)
    // CHECK-NEXT: // users: (unused), (%u, %v)
    %r, %s = "test.custom_result_name"(%q#0, %q#0, %test) {names = ["r", "s"]} : (i32, i32, i32) -> (i32, i32)
    // CHECK-NEXT: // unused
    %u, %v = "test.custom_result_name"(%s, %q#0, %p) {names = ["u", "v"]} : (i32, i32, i32) -> (i32, i32)
}
