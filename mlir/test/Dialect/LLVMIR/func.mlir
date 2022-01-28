// RUN: mlir-opt -split-input-file -verify-diagnostics %s | mlir-opt | FileCheck %s
// RUN: mlir-opt -split-input-file -verify-diagnostics -mlir-print-op-generic %s | FileCheck %s --check-prefix=GENERIC

module {
  // GENERIC: "llvm.func"
  // GENERIC: sym_name = "foo"
  // GENERIC-SAME: type = !llvm.func<void ()>
  // GENERIC-SAME: () -> ()
  // CHECK: llvm.func @foo()
  "llvm.func"() ({
  }) {sym_name = "foo", type = !llvm.func<void ()>} : () -> ()

  // GENERIC: "llvm.func"
  // GENERIC: sym_name = "bar"
  // GENERIC-SAME: type = !llvm.func<i64 (i64, i64)>
  // GENERIC-SAME: () -> ()
  // CHECK: llvm.func @bar(i64, i64) -> i64
  "llvm.func"() ({
  }) {sym_name = "bar", type = !llvm.func<i64 (i64, i64)>} : () -> ()

  // GENERIC: "llvm.func"
  // CHECK: llvm.func @baz(%{{.*}}: i64) -> i64
  "llvm.func"() ({
  // GENERIC: ^bb0
  ^bb0(%arg0: i64):
    // GENERIC: llvm.return
    llvm.return %arg0 : i64

  // GENERIC: sym_name = "baz"
  // GENERIC-SAME: type = !llvm.func<i64 (i64)>
  // GENERIC-SAME: () -> ()
  }) {sym_name = "baz", type = !llvm.func<i64 (i64)>} : () -> ()

  // CHECK: llvm.func @qux(!llvm.ptr<i64> {llvm.noalias}, i64)
  // CHECK: attributes {xxx = {yyy = 42 : i64}}
  "llvm.func"() ({
  }) {sym_name = "qux", type = !llvm.func<void (ptr<i64>, i64)>,
      arg_attrs = [{llvm.noalias}, {}], xxx = {yyy = 42}} : () -> ()

  // CHECK: llvm.func @roundtrip1()
  llvm.func @roundtrip1()

  // CHECK: llvm.func @roundtrip2(i64, f32) -> f64
  llvm.func @roundtrip2(i64, f32) -> f64

  // CHECK: llvm.func @roundtrip3(i32, i1)
  llvm.func @roundtrip3(%a: i32, %b: i1)

  // CHECK: llvm.func @roundtrip4(%{{.*}}: i32, %{{.*}}: i1) {
  llvm.func @roundtrip4(%a: i32, %b: i1) {
    llvm.return
  }

  // CHECK: llvm.func @roundtrip5()
  // CHECK: attributes {baz = 42 : i64, foo = "bar"}
  llvm.func @roundtrip5() attributes {foo = "bar", baz = 42}

  // CHECK: llvm.func @roundtrip6()
  // CHECK: attributes {baz = 42 : i64, foo = "bar"}
  llvm.func @roundtrip6() attributes {foo = "bar", baz = 42} {
    llvm.return
  }

  // CHECK: llvm.func @roundtrip7() {
  llvm.func @roundtrip7() attributes {} {
    llvm.return
  }

  // CHECK: llvm.func @roundtrip8() -> i32
  llvm.func @roundtrip8() -> i32 attributes {}

  // CHECK: llvm.func @roundtrip9(!llvm.ptr<i32> {llvm.noalias})
  llvm.func @roundtrip9(!llvm.ptr<i32> {llvm.noalias})

  // CHECK: llvm.func @roundtrip10(!llvm.ptr<i32> {llvm.noalias})
  llvm.func @roundtrip10(%arg0: !llvm.ptr<i32> {llvm.noalias})

  // CHECK: llvm.func @roundtrip11(%{{.*}}: !llvm.ptr<i32> {llvm.noalias}) {
  llvm.func @roundtrip11(%arg0: !llvm.ptr<i32> {llvm.noalias}) {
    llvm.return
  }

  // CHECK: llvm.func @roundtrip12(%{{.*}}: !llvm.ptr<i32> {llvm.noalias})
  // CHECK: attributes {foo = 42 : i32}
  llvm.func @roundtrip12(%arg0: !llvm.ptr<i32> {llvm.noalias})
  attributes {foo = 42 : i32} {
    llvm.return
  }

  // CHECK: llvm.func @byvalattr(%{{.*}}: !llvm.ptr<i32> {llvm.byval})
  llvm.func @byvalattr(%arg0: !llvm.ptr<i32> {llvm.byval}) {
    llvm.return
  }

  // CHECK: llvm.func @sretattr(%{{.*}}: !llvm.ptr<i32> {llvm.sret})
  llvm.func @sretattr(%arg0: !llvm.ptr<i32> {llvm.sret}) {
    llvm.return
  }

  // CHECK: llvm.func @variadic(...)
  llvm.func @variadic(...)

  // CHECK: llvm.func @variadic_args(i32, i32, ...)
  llvm.func @variadic_args(i32, i32, ...)

  //
  // Check that functions can have linkage attributes.
  //

  // CHECK: llvm.func internal
  llvm.func internal @internal_func() {
    llvm.return
  }

  // CHECK: llvm.func weak
  llvm.func weak @weak_linkage() {
    llvm.return
  }

  // Omit the `external` linkage, which is the default, in the custom format.
  // Check that it is present in the generic format using its numeric value.
  //
  // CHECK: llvm.func @external_func
  // GENERIC: linkage = #llvm.linkage<external>
  llvm.func external @external_func()
}

// -----

module {
  // expected-error@+1 {{requires one region}}
  "llvm.func"() {sym_name = "no_region", type = !llvm.func<void ()>} : () -> ()
}

// -----

module {
  // expected-error@+1 {{requires a type attribute 'type'}}
  "llvm.func"() ({}) {sym_name = "missing_type"} : () -> ()
}

// -----

module {
  // expected-error@+1 {{requires 'type' attribute of wrapped LLVM function type}}
  "llvm.func"() ({}) {sym_name = "non_llvm_type", type = i64} : () -> ()
}

// -----

module {
  // expected-error@+1 {{requires 'type' attribute of wrapped LLVM function type}}
  "llvm.func"() ({}) {sym_name = "non_function_type", type = i64} : () -> ()
}

// -----

module {
  // expected-error@+1 {{entry block must have 0 arguments}}
  "llvm.func"() ({
  ^bb0(%arg0: i64):
    llvm.return
  }) {sym_name = "wrong_arg_number", type = !llvm.func<void ()>} : () -> ()
}

// -----

module {
  // expected-error@+1 {{entry block argument #0 is not of LLVM type}}
  "llvm.func"() ({
  ^bb0(%arg0: tensor<*xf32>):
    llvm.return
  }) {sym_name = "wrong_arg_number", type = !llvm.func<void (i64)>} : () -> ()
}

// -----

module {
  // expected-error@+1 {{entry block argument #0 does not match the function signature}}
  "llvm.func"() ({
  ^bb0(%arg0: i32):
    llvm.return
  }) {sym_name = "wrong_arg_number", type = !llvm.func<void (i64)>} : () -> ()
}

// -----

module {
  // expected-error@+1 {{failed to construct function type: expected LLVM type for function arguments}}
  llvm.func @foo(tensor<*xf32>)
}

// -----

module {
  // expected-error@+1 {{failed to construct function type: expected LLVM type for function results}}
  llvm.func @foo() -> tensor<*xf32>
}

// -----

module {
  // expected-error@+1 {{failed to construct function type: expected zero or one function result}}
  llvm.func @foo() -> (i64, i64)
}

// -----

module {
  // expected-error@+1 {{only external functions can be variadic}}
  llvm.func @variadic_def(...) {
    llvm.return
  }
}

// -----

module {
  // expected-error@+1 {{cannot attach result attributes to functions with a void return}}
  llvm.func @variadic_def() -> (!llvm.void {llvm.noalias})
}

// -----

module {
  // expected-error@+1 {{variadic arguments must be in the end of the argument list}}
  llvm.func @variadic_inside(%arg0: i32, ..., %arg1: i32)
}

// -----

module {
  // expected-error@+1 {{external functions must have 'external' or 'extern_weak' linkage}}
  llvm.func internal @internal_external_func()
}

// -----

module {
  // expected-error@+1 {{functions cannot have 'common' linkage}}
  llvm.func common @common_linkage_func()
}
