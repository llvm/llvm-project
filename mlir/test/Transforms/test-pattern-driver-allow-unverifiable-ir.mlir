// Tests interactions between `allowUnverifiableIR` rewriter option and
// `MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS` compile time option for both
// greedy and walk-based pattern rewriters. All RUN lines use an unverifiable
// input and either print it successfully or fail with a crash.

// Check that in both drivers, if `allow-unverifiable-ir=true`, the driver does
// not verify IR, independently of the compile time option.
// RUN: mlir-opt %s --mlir-very-unsafe-disable-verifier-on-parsing --verify-each=false \
// RUN:   --test-greedy-patterns="allow-unverifiable-ir=true" | FileCheck %s
// RUN: mlir-opt %s --mlir-very-unsafe-disable-verifier-on-parsing --verify-each=false \
// RUN:   --test-walk-pattern-rewrite-driver="allow-unverifiable-ir=true" | FileCheck %s

// If the expensive checks are *disabled* at compile time, the driver does not
// verify IR even if `allow-unverifiable-ir=false` (or default).
// RUN: %if !mlir-expensive-checks %{ mlir-opt %s --mlir-very-unsafe-disable-verifier-on-parsing --verify-each=false --test-greedy-patterns="allow-unverifiable-ir=false" | FileCheck %s %}
// RUN: %if !mlir-expensive-checks %{ mlir-opt %s --mlir-very-unsafe-disable-verifier-on-parsing --verify-each=false --test-greedy-patterns | FileCheck %s %}
// RUN: %if !mlir-expensive-checks %{ mlir-opt %s --mlir-very-unsafe-disable-verifier-on-parsing --verify-each=false --test-walk-pattern-rewrite-driver="allow-unverifiable-ir=false" | FileCheck %s %}
// RUN: %if !mlir-expensive-checks %{ mlir-opt %s --mlir-very-unsafe-disable-verifier-on-parsing --verify-each=false --test-walk-pattern-rewrite-driver | FileCheck %s %}

// If the expensive checks are *enabled* at compile time and
// `allow-unverifiable-ir=false` (or default), the driver fails.
// RUN: %if mlir-expensive-checks %{ not --crash mlir-opt %s --mlir-very-unsafe-disable-verifier-on-parsing --verify-each=false --test-greedy-patterns="allow-unverifiable-ir=false" 2>&1 | FileCheck %s --check-prefix=ERR-GREEDY %}
// RUN: %if mlir-expensive-checks %{ not --crash mlir-opt %s --mlir-very-unsafe-disable-verifier-on-parsing --verify-each=false --test-greedy-patterns 2>&1 | FileCheck %s --check-prefix=ERR-GREEDY %}
// RUN: %if mlir-expensive-checks %{ not --crash mlir-opt %s --mlir-very-unsafe-disable-verifier-on-parsing --verify-each=false --test-walk-pattern-rewrite-driver="allow-unverifiable-ir=false" 2>&1 | FileCheck %s --check-prefix=ERR-WALK %}
// RUN: %if mlir-expensive-checks %{ not --crash mlir-opt %s --mlir-very-unsafe-disable-verifier-on-parsing --verify-each=false --test-walk-pattern-rewrite-driver 2>&1 | FileCheck %s --check-prefix=ERR-WALK %}

// ERR-GREEDY: LLVM ERROR: greedy pattern rewriter input IR failed to verify
// ERR-WALK: LLVM ERROR: walk pattern rewriter input IR failed to verify

// CHECK-LABEL: sym_name = "test_allow_unverifiable_ir"
// CHECK: "test.unverifiable_op"() {test.invalid_attr} : () -> ()
func.func @test_allow_unverifiable_ir() {
  "test.unverifiable_op"() {test.invalid_attr} : () -> ()
  return
}
