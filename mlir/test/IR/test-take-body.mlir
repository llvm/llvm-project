// RUN: mlir-opt -allow-unregistered-dialect %s --test-take-body -split-input-file

func.func @foo() {
  %0 = "test.foo"() : () -> i32
  cf.br ^header

^header:
  cf.br ^body

^body:
  "test.use"(%0) : (i32) -> ()
  cf.br ^header
}

func.func private @bar() {
  return
}

// CHECK-LABEL: func @foo
// CHECK-NEXT: return

// CHECK-LABEL: func private @bar()
// CHECK-NOT: {
