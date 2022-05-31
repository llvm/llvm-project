// RUN: mlir-opt -split-input-file -verify-diagnostics %s | mlir-opt | FileCheck %s
// RUN: mlir-opt -split-input-file -verify-diagnostics -mlir-print-op-generic %s | FileCheck %s --check-prefix=GENERIC
// RUN: mlir-opt -split-input-file -verify-diagnostics %s -mlir-print-debuginfo | mlir-opt -mlir-print-debuginfo | FileCheck %s --check-prefix=LOCINFO

module {
  // GENERIC: "llvm.func"
  // GENERIC: function_type = !llvm.func<void ()>
  // GENERIC-SAME: sym_name = "foo"
  // GENERIC-SAME: () -> ()
  // CHECK: llvm.func @foo()
  "llvm.func"() ({
  }) {sym_name = "foo", function_type = !llvm.func<void ()>} : () -> ()

  // GENERIC: "llvm.func"
  // GENERIC: function_type = !llvm.func<i64 (i64, i64)>
  // GENERIC-SAME: sym_name = "bar"
  // GENERIC-SAME: () -> ()
  // CHECK: llvm.func @bar(i64, i64) -> i64
  "llvm.func"() ({
  }) {sym_name = "bar", function_type = !llvm.func<i64 (i64, i64)>} : () -> ()

  // GENERIC: "llvm.func"
  // CHECK: llvm.func @baz(%{{.*}}: i64) -> i64
  "llvm.func"() ({
  // GENERIC: ^bb0
  ^bb0(%arg0: i64):
    // GENERIC: llvm.return
    llvm.return %arg0 : i64

  // GENERIC: function_type = !llvm.func<i64 (i64)>
  // GENERIC-SAME: sym_name = "baz"
  // GENERIC-SAME: () -> ()
  }) {sym_name = "baz", function_type = !llvm.func<i64 (i64)>} : () -> ()

  // CHECK: llvm.func @qux(!llvm.ptr<i64> {llvm.noalias}, i64)
  // CHECK: attributes {xxx = {yyy = 42 : i64}}
  "llvm.func"() ({
  }) {sym_name = "qux", function_type = !llvm.func<void (ptr<i64>, i64)>,
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
  // LOCINFO: llvm.func @sretattr(%{{.*}}: !llvm.ptr<i32> {llvm.sret} loc("some_source_loc"))
  llvm.func @sretattr(%arg0: !llvm.ptr<i32> {llvm.sret} loc("some_source_loc")) {
    llvm.return
  }

  // CHECK: llvm.func @nestattr(%{{.*}}: !llvm.ptr<i32> {llvm.nest})
  llvm.func @nestattr(%arg0: !llvm.ptr<i32> {llvm.nest}) {
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

  // CHECK-LABEL: llvm.func @arg_struct_attr(
  // CHECK-SAME: %{{.*}}: !llvm.struct<(i32)> {llvm.struct_attrs = [{llvm.noalias}]}) {
  llvm.func @arg_struct_attr(
      %arg0 : !llvm.struct<(i32)> {llvm.struct_attrs = [{llvm.noalias}]}) {
    llvm.return
  }

   // CHECK-LABEL: llvm.func @res_struct_attr(%{{.*}}: !llvm.struct<(i32)>)
   // CHECK-SAME:-> (!llvm.struct<(i32)> {llvm.struct_attrs = [{llvm.noalias}]}) {
  llvm.func @res_struct_attr(%arg0 : !llvm.struct<(i32)>)
      -> (!llvm.struct<(i32)> {llvm.struct_attrs = [{llvm.noalias}]}) {
    llvm.return %arg0 : !llvm.struct<(i32)>
  }

  // CHECK: llvm.func @cconv1
  llvm.func ccc @cconv1() {
    llvm.return
  }

  // CHECK: llvm.func weak @cconv2
  llvm.func weak ccc @cconv2() {
    llvm.return
  }

  // CHECK: llvm.func weak fastcc @cconv3
  llvm.func weak fastcc @cconv3() {
    llvm.return
  }
}

// -----

module {
  // expected-error@+1 {{requires one region}}
  "llvm.func"() {function_type = !llvm.func<void ()>, sym_name = "no_region"} : () -> ()
}

// -----

module {
  // expected-error@+1 {{requires attribute 'function_type'}}
  "llvm.func"() ({}) {sym_name = "missing_type"} : () -> ()
}

// -----

module {
  // expected-error@+1 {{attribute 'function_type' failed to satisfy constraint: type attribute of LLVM function type}}
  "llvm.func"() ({}) {sym_name = "non_llvm_type", function_type = i64} : () -> ()
}

// -----

module {
  // expected-error@+1 {{attribute 'function_type' failed to satisfy constraint: type attribute of LLVM function type}}
  "llvm.func"() ({}) {sym_name = "non_function_type", function_type = i64} : () -> ()
}

// -----

module {
  // expected-error@+1 {{entry block must have 0 arguments}}
  "llvm.func"() ({
  ^bb0(%arg0: i64):
    llvm.return
  }) {function_type = !llvm.func<void ()>, sym_name = "wrong_arg_number"} : () -> ()
}

// -----

module {
  // expected-error@+1 {{entry block argument #0('tensor<*xf32>') must match the type of the corresponding argument in function signature('i64')}}
  "llvm.func"() ({
  ^bb0(%arg0: tensor<*xf32>):
    llvm.return
  }) {function_type = !llvm.func<void (i64)>, sym_name = "wrong_arg_number"} : () -> ()
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

// -----

module {
  // expected-error@+1 {{custom op 'llvm.func' expected valid '@'-identifier for symbol name}}
  llvm.func cc_12 @unknown_calling_convention()
}

// -----

module {
  // expected-error@+2 {{unknown calling convention: cc_12}}
  "llvm.func"() ({
  }) {sym_name = "generic_unknown_calling_convention", CConv = #llvm.cconv<cc_12>, function_type = !llvm.func<i64 (i64, i64)>} : () -> ()
}
