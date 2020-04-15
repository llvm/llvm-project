// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK: llvm.mlir.global external @default_external
llvm.mlir.global @default_external() : !llvm.i64

// CHECK: llvm.mlir.global external constant @default_external_constant
llvm.mlir.global constant @default_external_constant(42) : !llvm.i64

// CHECK: llvm.mlir.global internal @global(42 : i64) : !llvm.i64
llvm.mlir.global internal @global(42 : i64) : !llvm.i64

// CHECK: llvm.mlir.global internal constant @constant(3.700000e+01 : f64) : !llvm.float
llvm.mlir.global internal constant @constant(37.0) : !llvm.float

// CHECK: llvm.mlir.global internal constant @".string"("foobar")
llvm.mlir.global internal constant @".string"("foobar") : !llvm<"[6 x i8]">

// CHECK: llvm.mlir.global internal @string_notype("1234567")
llvm.mlir.global internal @string_notype("1234567")

// CHECK: llvm.mlir.global internal @global_undef()
llvm.mlir.global internal @global_undef() : !llvm.i64

// CHECK: llvm.mlir.global internal @global_mega_initializer() : !llvm.i64 {
// CHECK-NEXT:  %[[c:[0-9]+]] = llvm.mlir.constant(42 : i64) : !llvm.i64
// CHECK-NEXT:  llvm.return %[[c]] : !llvm.i64
// CHECK-NEXT: }
llvm.mlir.global internal @global_mega_initializer() : !llvm.i64 {
  %c = llvm.mlir.constant(42 : i64) : !llvm.i64
  llvm.return %c : !llvm.i64
}

// Check different linkage types.
// CHECK: llvm.mlir.global private
llvm.mlir.global private @private() : !llvm.i64
// CHECK: llvm.mlir.global internal
llvm.mlir.global internal @internal() : !llvm.i64
// CHECK: llvm.mlir.global available_externally
llvm.mlir.global available_externally @available_externally() : !llvm.i64
// CHECK: llvm.mlir.global linkonce
llvm.mlir.global linkonce @linkonce() : !llvm.i64
// CHECK: llvm.mlir.global weak
llvm.mlir.global weak @weak() : !llvm.i64
// CHECK: llvm.mlir.global common
llvm.mlir.global common @common() : !llvm.i64
// CHECK: llvm.mlir.global appending
llvm.mlir.global appending @appending() : !llvm.i64
// CHECK: llvm.mlir.global extern_weak
llvm.mlir.global extern_weak @extern_weak() : !llvm.i64
// CHECK: llvm.mlir.global linkonce_odr
llvm.mlir.global linkonce_odr @linkonce_odr() : !llvm.i64
// CHECK: llvm.mlir.global weak_odr
llvm.mlir.global weak_odr @weak_odr() : !llvm.i64

// CHECK-LABEL: references
func @references() {
  // CHECK: llvm.mlir.addressof @global : !llvm<"i64*">
  %0 = llvm.mlir.addressof @global : !llvm<"i64*">

  // CHECK: llvm.mlir.addressof @".string" : !llvm<"[6 x i8]*">
  %1 = llvm.mlir.addressof @".string" : !llvm<"[6 x i8]*">

  llvm.return
}

// -----

// expected-error @+1 {{op requires string attribute 'sym_name'}}
"llvm.mlir.global"() ({}) {type = !llvm.i64, constant, value = 42 : i64} : () -> ()

// -----

// expected-error @+1 {{op requires attribute 'type'}}
"llvm.mlir.global"() ({}) {sym_name = "foo", constant, value = 42 : i64} : () -> ()

// -----

// expected-error @+1 {{expects type to be a valid element type for an LLVM pointer}}
llvm.mlir.global internal constant @constant(37.0) : !llvm<"label">

// -----

// expected-error @+1 {{'addr_space' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative}}
"llvm.mlir.global"() ({}) {sym_name = "foo", type = !llvm.i64, value = 42 : i64, addr_space = -1 : i32, linkage = 0} : () -> ()

// -----

// expected-error @+1 {{'addr_space' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative}}
"llvm.mlir.global"() ({}) {sym_name = "foo", type = !llvm.i64, value = 42 : i64, addr_space = 1.0 : f32, linkage = 0} : () -> ()

// -----

func @foo() {
  // expected-error @+1 {{must appear at the module level}}
  llvm.mlir.global internal @bar(42) : !llvm.i32
}

// -----

// expected-error @+1 {{requires an i8 array type of the length equal to that of the string}}
llvm.mlir.global internal constant @string("foobar") : !llvm<"[42 x i8]">

// -----

// expected-error @+1 {{type can only be omitted for string globals}}
llvm.mlir.global internal @i64_needs_type(0: i64)

// -----

// expected-error @+1 {{expected zero or one type}}
llvm.mlir.global internal @more_than_one_type(0) : !llvm.i64, !llvm.i32

// -----

llvm.mlir.global internal @foo(0: i32) : !llvm.i32

func @bar() {
  // expected-error @+2{{expected ':'}}
  llvm.mlir.addressof @foo
}

// -----

func @foo() {
  // The attribute parser will consume the first colon-type, so we put two of
  // them to trigger the attribute type mismatch error.
  // expected-error @+1 {{invalid kind of attribute specified}}
  llvm.mlir.addressof "foo" : i64 : !llvm<"void ()*">
}

// -----

func @foo() {
  // expected-error @+1 {{must reference a global defined by 'llvm.mlir.global'}}
  llvm.mlir.addressof @foo : !llvm<"void ()*">
}

// -----

llvm.mlir.global internal @foo(0: i32) : !llvm.i32

func @bar() {
  // expected-error @+1 {{the type must be a pointer to the type of the referred global}}
  llvm.mlir.addressof @foo : !llvm<"i64*">
}

// -----

// expected-error @+2 {{'llvm.mlir.global' op expects regions to end with 'llvm.return', found 'llvm.mlir.constant'}}
// expected-note @+1 {{in custom textual format, the absence of terminator implies 'llvm.return'}}
llvm.mlir.global internal @g() : !llvm.i64 {
  %c = llvm.mlir.constant(42 : i64) : !llvm.i64
}

// -----

// expected-error @+1 {{'llvm.mlir.global' op initializer region type '!llvm.i64' does not match global type '!llvm.i32'}}
llvm.mlir.global internal @g() : !llvm.i32 {
  %c = llvm.mlir.constant(42 : i64) : !llvm.i64
  llvm.return %c : !llvm.i64
}

// -----

// expected-error @+1 {{'llvm.mlir.global' op cannot have both initializer value and region}}
llvm.mlir.global internal @g(43 : i64) : !llvm.i64 {
  %c = llvm.mlir.constant(42 : i64) : !llvm.i64
  llvm.return %c : !llvm.i64
}

// -----

llvm.mlir.global internal @g(32 : i64) {addr_space = 3: i32} : !llvm.i64
func @mismatch_addr_space_implicit_global() {
  // expected-error @+1 {{op the type must be a pointer to the type of the referred global}}
  llvm.mlir.addressof @g : !llvm<"i64*">
}

// -----

llvm.mlir.global internal @g(32 : i64) {addr_space = 3: i32} : !llvm.i64
func @mismatch_addr_space() {
  // expected-error @+1 {{op the type must be a pointer to the type of the referred global}}
  llvm.mlir.addressof @g : !llvm<"i64 addrspace(4)*">
}
