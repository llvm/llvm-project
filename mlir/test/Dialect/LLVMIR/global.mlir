// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK: llvm.mlir.global external @default_external
llvm.mlir.global @default_external() : i64

// CHECK: llvm.mlir.global external constant @default_external_constant
llvm.mlir.global constant @default_external_constant(42) : i64

// CHECK: llvm.mlir.global internal @global(42 : i64) : i64
llvm.mlir.global internal @global(42 : i64) : i64

// CHECK: llvm.mlir.global private @aligned_global(42 : i64) {aligned = 64 : i64} : i64
llvm.mlir.global private @aligned_global(42 : i64) {aligned = 64} : i64

// CHECK: llvm.mlir.global private constant @aligned_global_const(42 : i64) {aligned = 32 : i64} : i64
llvm.mlir.global private constant @aligned_global_const(42 : i64) {aligned = 32} : i64

// CHECK: llvm.mlir.global internal constant @constant(3.700000e+01 : f64) : f32
llvm.mlir.global internal constant @constant(37.0) : f32

// CHECK: llvm.mlir.global internal constant @".string"("foobar")
llvm.mlir.global internal constant @".string"("foobar") : !llvm.array<6 x i8>

// CHECK: llvm.mlir.global internal @string_notype("1234567")
llvm.mlir.global internal @string_notype("1234567")

// CHECK: llvm.mlir.global internal @global_undef()
llvm.mlir.global internal @global_undef() : i64

// CHECK: llvm.mlir.global internal @global_mega_initializer() : i64 {
// CHECK-NEXT:  %[[c:[0-9]+]] = llvm.mlir.constant(42 : i64) : i64
// CHECK-NEXT:  llvm.return %[[c]] : i64
// CHECK-NEXT: }
llvm.mlir.global internal @global_mega_initializer() : i64 {
  %c = llvm.mlir.constant(42 : i64) : i64
  llvm.return %c : i64
}

// Check different linkage types.
// CHECK: llvm.mlir.global private
llvm.mlir.global private @private() : i64
// CHECK: llvm.mlir.global internal
llvm.mlir.global internal @internal() : i64
// CHECK: llvm.mlir.global available_externally
llvm.mlir.global available_externally @available_externally() : i64
// CHECK: llvm.mlir.global linkonce
llvm.mlir.global linkonce @linkonce() : i64
// CHECK: llvm.mlir.global weak
llvm.mlir.global weak @weak() : i64
// CHECK: llvm.mlir.global common
llvm.mlir.global common @common() : i64
// CHECK: llvm.mlir.global appending
llvm.mlir.global appending @appending() : !llvm.array<2 x i64>
// CHECK: llvm.mlir.global extern_weak
llvm.mlir.global extern_weak @extern_weak() : i64
// CHECK: llvm.mlir.global linkonce_odr
llvm.mlir.global linkonce_odr @linkonce_odr() : i64
// CHECK: llvm.mlir.global weak_odr
llvm.mlir.global weak_odr @weak_odr() : i64
// CHECK: llvm.mlir.global external @has_thr_local(42 : i64) {thr_local} : i64
llvm.mlir.global external @has_thr_local(42 : i64) {thr_local} : i64
// CHECK: llvm.mlir.global external @has_dso_local(42 : i64) {dso_local} : i64
llvm.mlir.global external @has_dso_local(42 : i64) {dso_local} : i64
// CHECK: llvm.mlir.global external @has_addr_space(32 : i64) {addr_space = 3 : i32} : i64
llvm.mlir.global external @has_addr_space(32 : i64) {addr_space = 3: i32} : i64

// CHECK-LABEL: references
func.func @references() {
  // CHECK: llvm.mlir.addressof @global : !llvm.ptr<i64>
  %0 = llvm.mlir.addressof @global : !llvm.ptr<i64>

  // CHECK: llvm.mlir.addressof @".string" : !llvm.ptr<array<6 x i8>>
  %1 = llvm.mlir.addressof @".string" : !llvm.ptr<array<6 x i8>>

  // CHECK: llvm.mlir.addressof @global : !llvm.ptr
  %2 = llvm.mlir.addressof @global : !llvm.ptr

  // CHECK: llvm.mlir.addressof @has_addr_space : !llvm.ptr<3>
  %3 = llvm.mlir.addressof @has_addr_space : !llvm.ptr<3>

  llvm.return
}

// CHECK: llvm.mlir.global private local_unnamed_addr constant @local(42 : i64) : i64
llvm.mlir.global private local_unnamed_addr constant @local(42 : i64) : i64

// CHECK: llvm.mlir.global private unnamed_addr constant @foo(42 : i64) : i64
llvm.mlir.global private unnamed_addr constant @foo(42 : i64) : i64

// CHECK: llvm.mlir.global internal constant @sectionvar("teststring")  {section = ".mysection"}
llvm.mlir.global internal constant @sectionvar("teststring")  {section = ".mysection"}: !llvm.array<10 x i8>

// -----

// expected-error @+1 {{op requires attribute 'sym_name'}}
"llvm.mlir.global"() ({}) {linkage = "private", type = i64, constant, global_type = i64, value = 42 : i64} : () -> ()

// -----

// expected-error @+1 {{requires attribute 'global_type'}}
"llvm.mlir.global"() ({}) {sym_name = "foo", constant, value = 42 : i64} : () -> ()

// -----

// expected-error @+1 {{expects type to be a valid element type for an LLVM pointer}}
llvm.mlir.global internal constant @constant(37.0) : !llvm.label

// -----

// expected-error @+1 {{'addr_space' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative}}
"llvm.mlir.global"() ({}) {sym_name = "foo", global_type = i64, value = 42 : i64, addr_space = -1 : i32, linkage = #llvm.linkage<private>} : () -> ()

// -----

// expected-error @+1 {{'addr_space' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative}}
"llvm.mlir.global"() ({}) {sym_name = "foo", global_type = i64, value = 42 : i64, addr_space = 1.0 : f32, linkage = #llvm.linkage<private>} : () -> ()

// -----

func.func @foo() {
  // expected-error @+1 {{must appear at the module level}}
  llvm.mlir.global internal @bar(42) : i32

  return
}

// -----

// expected-error @+1 {{requires an i8 array type of the length equal to that of the string}}
llvm.mlir.global internal constant @string("foobar") : !llvm.array<42 x i8>

// -----

// expected-error @+1 {{type can only be omitted for string globals}}
llvm.mlir.global internal @i64_needs_type(0: i64)

// -----

// expected-error @+1 {{expected zero or one type}}
llvm.mlir.global internal @more_than_one_type(0) : i64, i32

// -----

llvm.mlir.global internal @foo(0: i32) : i32

func.func @bar() {
  // expected-error @+1{{expected ':'}}
  llvm.mlir.addressof @foo
}

// -----

func.func @foo() {
  // The attribute parser will consume the first colon-type, so we put two of
  // them to trigger the attribute type mismatch error.
  // expected-error @+1 {{invalid kind of attribute specified}}
  llvm.mlir.addressof "foo" : i64 : !llvm.ptr<func<void ()>>
  llvm.return
}

// -----

func.func @foo() {
  // expected-error @+1 {{must reference a global defined by 'llvm.mlir.global'}}
  llvm.mlir.addressof @foo : !llvm.ptr<func<void ()>>
  llvm.return
}

// -----

llvm.mlir.global internal @foo(0: i32) : i32

func.func @bar() {
  // expected-error @+1 {{the type must be a pointer to the type of the referenced global}}
  llvm.mlir.addressof @foo : !llvm.ptr<i64>
  llvm.return
}

// -----

llvm.func @foo()

llvm.func @bar() {
  // expected-error @+1 {{the type must be a pointer to the type of the referenced function}}
  llvm.mlir.addressof @foo : !llvm.ptr<i8>
  llvm.return
}

// -----

// expected-error @+2 {{block with no terminator}}
llvm.mlir.global internal @g() : i64 {
  %c = llvm.mlir.constant(42 : i64) : i64
}

// -----

// expected-error @+1 {{'llvm.mlir.global' op initializer region type 'i64' does not match global type 'i32'}}
llvm.mlir.global internal @g() : i32 {
  %c = llvm.mlir.constant(42 : i64) : i64
  llvm.return %c : i64
}

// -----

// expected-error @+1 {{'llvm.mlir.global' op cannot have both initializer value and region}}
llvm.mlir.global internal @g(43 : i64) : i64 {
  %c = llvm.mlir.constant(42 : i64) : i64
  llvm.return %c : i64
}

// -----

llvm.mlir.global internal @g(32 : i64) {addr_space = 3: i32} : i64
func.func @mismatch_addr_space_implicit_global() {
  // expected-error @+1 {{pointer address space must match address space of the referenced global}}
  llvm.mlir.addressof @g : !llvm.ptr<i64>
  llvm.return
}

// -----

llvm.mlir.global internal @g(32 : i64) {addr_space = 3: i32} : i64
func.func @mismatch_addr_space() {
  // expected-error @+1 {{pointer address space must match address space of the referenced global}}
  llvm.mlir.addressof @g : !llvm.ptr<i64, 4>
  llvm.return
}
// -----

llvm.mlir.global internal @g(32 : i64) {addr_space = 3: i32} : i64

func.func @mismatch_addr_space_opaque() {
  // expected-error @+1 {{pointer address space must match address space of the referenced global}}
  llvm.mlir.addressof @g : !llvm.ptr<4>
  llvm.return
}

// -----

llvm.func @ctor() {
  llvm.return
}

// CHECK: llvm.mlir.global_ctors {ctors = [@ctor], priorities = [0 : i32]}
llvm.mlir.global_ctors { ctors = [@ctor], priorities = [0 : i32]}

// -----

llvm.func @dtor() {
  llvm.return
}

// CHECK: llvm.mlir.global_dtors {dtors = [@dtor], priorities = [0 : i32]}
llvm.mlir.global_dtors { dtors = [@dtor], priorities = [0 : i32]}
