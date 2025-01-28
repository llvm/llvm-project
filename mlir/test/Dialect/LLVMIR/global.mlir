// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK: llvm.mlir.global external @default_external
llvm.mlir.global @default_external() : i64

// CHECK: llvm.mlir.global external constant @default_external_constant
llvm.mlir.global constant @default_external_constant(42) : i64

// CHECK: llvm.mlir.global internal @global(42 : i64) {addr_space = 0 : i32} : i64
llvm.mlir.global internal @global(42 : i64) : i64

// CHECK: llvm.mlir.global private @aligned_global(42 : i64) {addr_space = 0 : i32, aligned = 64 : i64} : i64
llvm.mlir.global private @aligned_global(42 : i64) {aligned = 64} : i64

// CHECK: llvm.mlir.global private constant @aligned_global_const(42 : i64) {addr_space = 0 : i32, aligned = 32 : i64} : i64
llvm.mlir.global private constant @aligned_global_const(42 : i64) {aligned = 32} : i64

// CHECK: llvm.mlir.global internal constant @constant(3.700000e+01 : f64) {addr_space = 0 : i32} : f32
llvm.mlir.global internal constant @constant(37.0) : f32

// CHECK: llvm.mlir.global internal constant @".string"("foobar")
llvm.mlir.global internal constant @".string"("foobar") : !llvm.array<6 x i8>

// CHECK: llvm.mlir.global internal @string_notype("1234567")
llvm.mlir.global internal @string_notype("1234567")

// CHECK: llvm.mlir.global internal @global_undef()
llvm.mlir.global internal @global_undef() : i64

// CHECK: llvm.mlir.global internal @global_mega_initializer() {addr_space = 0 : i32} : i64 {
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
// CHECK: llvm.mlir.global external @has_thr_local(42 : i64) {addr_space = 0 : i32, thr_local} : i64
llvm.mlir.global external @has_thr_local(42 : i64) {thr_local} : i64
// CHECK: llvm.mlir.global external @has_dso_local(42 : i64) {addr_space = 0 : i32, dso_local} : i64
llvm.mlir.global external @has_dso_local(42 : i64) {dso_local} : i64
// CHECK: llvm.mlir.global external @has_addr_space(32 : i64) {addr_space = 3 : i32} : i64
llvm.mlir.global external @has_addr_space(32 : i64) {addr_space = 3: i32} : i64

// CHECK: llvm.comdat @__llvm_comdat
llvm.comdat @__llvm_comdat {
  // CHECK: llvm.comdat_selector @any any
  llvm.comdat_selector @any any
}
// CHECK: llvm.mlir.global external @any() comdat(@__llvm_comdat::@any) {addr_space = 1 : i32} : i64
llvm.mlir.global @any() comdat(@__llvm_comdat::@any) {addr_space = 1 : i32} : i64

// CHECK-LABEL: references
func.func @references() {
  // CHECK: llvm.mlir.addressof @".string" : !llvm.ptr
  %0 = llvm.mlir.addressof @".string" : !llvm.ptr

  // CHECK: llvm.mlir.addressof @global : !llvm.ptr
  %1 = llvm.mlir.addressof @global : !llvm.ptr

  // CHECK: llvm.mlir.addressof @has_addr_space : !llvm.ptr<3>
  %2 = llvm.mlir.addressof @has_addr_space : !llvm.ptr<3>

  llvm.return
}

// CHECK: llvm.mlir.global private local_unnamed_addr constant @local(42 : i64) {addr_space = 0 : i32} : i64
llvm.mlir.global private local_unnamed_addr constant @local(42 : i64) : i64

// CHECK: llvm.mlir.global private unnamed_addr constant @foo(42 : i64) {addr_space = 0 : i32} : i64
llvm.mlir.global private unnamed_addr constant @foo(42 : i64) : i64

// CHECK: llvm.mlir.global internal constant @sectionvar("teststring") {addr_space = 0 : i32, section = ".mysection"}
llvm.mlir.global internal constant @sectionvar("teststring")  {section = ".mysection"}: !llvm.array<10 x i8>

// CHECK: llvm.mlir.global internal thread_local constant @thread_local(42 : i32)
llvm.mlir.global internal thread_local constant @thread_local(42 : i32) : i32

// Visibility types.
// CHECK: llvm.mlir.global internal hidden constant @hidden(42 : i32)
llvm.mlir.global internal hidden constant @hidden(42 : i32) : i32

// CHECK: llvm.mlir.global internal protected unnamed_addr @protected(42 : i32)
llvm.mlir.global internal protected unnamed_addr @protected(42 : i32) : i32

// -----

// expected-error @+1 {{op requires attribute 'sym_name'}}
"llvm.mlir.global"() ({}) {linkage = #llvm.linkage<private>, type = i64, constant, global_type = i64, value = 42 : i64} : () -> ()

// -----

// expected-error @+1 {{requires attribute 'global_type'}}
"llvm.mlir.global"() ({}) {sym_name = "foo", constant, value = 42 : i64} : () -> ()

// -----

// expected-error @+1 {{expects type to be a valid element type for an LLVM global}}
llvm.mlir.global internal constant @constant(37.0) : !llvm.label

// -----
// expected-error @+1 {{'addr_space' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative}}
"llvm.mlir.global"() ({}) {sym_name = "foo", global_type = i64, value = 42 : i64, addr_space = -1 : i32, linkage = #llvm.linkage<private>} : () -> ()

// -----

// expected-error @+1 {{'addr_space' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative}}
"llvm.mlir.global"() ({}) {sym_name = "foo", global_type = i64, value = 42 : i64, addr_space = 1.0 : f32, linkage = #llvm.linkage<private>} : () -> ()

// -----

func.func @foo() {
  // expected-error @+1 {{op symbol's parent must have the SymbolTable trait}}
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
  llvm.mlir.addressof "foo" : i64 : !llvm.ptr
  llvm.return
}

// -----

func.func @foo() {
  // expected-error @+1 {{must reference a global defined by 'llvm.mlir.global'}}
  llvm.mlir.addressof @foo : !llvm.ptr
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
  llvm.mlir.addressof @g : !llvm.ptr
  llvm.return
}

// -----

llvm.mlir.global internal @g(32 : i64) {addr_space = 3: i32} : i64

func.func @mismatch_addr_space() {
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

// -----

// CHECK: llvm.mlir.global external @target_ext() {addr_space = 0 : i32} : !llvm.target<"spirv.Image", i32, 0>
llvm.mlir.global @target_ext() : !llvm.target<"spirv.Image", i32, 0>

// CHECK:       llvm.mlir.global external @target_ext_init() {addr_space = 0 : i32} : !llvm.target<"spirv.Image", i32, 0>
// CHECK-NEXT:    %0 = llvm.mlir.zero : !llvm.target<"spirv.Image", i32, 0>
// CHECK-NEXT:    llvm.return %0 : !llvm.target<"spirv.Image", i32, 0>
// CHECK-NEXT:  }
llvm.mlir.global @target_ext_init() : !llvm.target<"spirv.Image", i32, 0> {
  %0 = llvm.mlir.zero : !llvm.target<"spirv.Image", i32, 0>
  llvm.return %0 : !llvm.target<"spirv.Image", i32, 0>
}

// -----

// expected-error @+1 {{global with target extension type can only be initialized with zero-initializer}}
llvm.mlir.global @target_fail(0 : i64) : !llvm.target<"spirv.Image", i32, 0>

// -----

// CHECK-DAG: #[[TYPE:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "uint64_t", sizeInBits = 64, encoding = DW_ATE_unsigned>
// CHECK-DAG: #[[FILE:.*]] = #llvm.di_file<"not" in "existence">
// CHECK-DAG: #[[CU:.*]] = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #[[FILE]], producer = "MLIR", isOptimized = true, emissionKind = Full>
// CHECK-DAG: #[[GVAR0:.*]] = #llvm.di_global_variable<scope = #[[CU]], name = "global_with_expr_1", linkageName = "global_with_expr_1", file = #[[FILE]], line = 370, type = #[[TYPE]], isLocalToUnit = true, isDefined = true, alignInBits = 8>
// CHECK-DAG: #[[GVAR1:.*]] = #llvm.di_global_variable<scope = #[[CU]], name = "global_with_expr_2", linkageName = "global_with_expr_2", file = #[[FILE]], line = 371, type = #[[TYPE]], isLocalToUnit = true, isDefined = true, alignInBits = 8>
// CHECK-DAG: #[[GVAR2:.*]] = #llvm.di_global_variable<scope = #[[CU]], name = "global_with_expr_3", linkageName = "global_with_expr_3", file = #[[FILE]], line = 372, type = #[[TYPE]], isLocalToUnit = true, isDefined = true, alignInBits = 8>
// CHECK-DAG: #[[GVAR3:.*]] = #llvm.di_global_variable<scope = #[[CU]], name = "global_with_expr_4", linkageName = "global_with_expr_4", file = #[[FILE]], line = 373, type = #[[TYPE]], isLocalToUnit = true, isDefined = true, alignInBits = 8>
// CHECK-DAG: #[[EXPR0:.*]] = #llvm.di_global_variable_expression<var = #[[GVAR0]], expr = <>>
// CHECK-DAG: #[[EXPR1:.*]] = #llvm.di_global_variable_expression<var = #[[GVAR1]], expr = <[DW_OP_push_object_address, DW_OP_deref]>>
// CHECK-DAG: #[[EXPR2:.*]] = #llvm.di_global_variable_expression<var = #[[GVAR2]], expr = <[DW_OP_LLVM_arg(0), DW_OP_LLVM_arg(1), DW_OP_plus]>>
// CHECK-DAG: #[[EXPR3:.*]] = #llvm.di_global_variable_expression<var = #[[GVAR3]], expr = <[DW_OP_LLVM_convert(16, DW_ATE_signed)]>>
// CHECK-DAG:   llvm.mlir.global external @global_with_expr1() {addr_space = 0 : i32, dbg_expr = [#[[EXPR0]]]} : i64
// CHECK-DAG:   llvm.mlir.global external @global_with_expr2() {addr_space = 0 : i32, dbg_expr = [#[[EXPR1]]]} : i64
// CHECK-DAG:   llvm.mlir.global external @global_with_expr3() {addr_space = 0 : i32, dbg_expr = [#[[EXPR2]]]} : i64
// CHECK-DAG:   llvm.mlir.global external @global_with_expr4() {addr_space = 0 : i32, dbg_expr = [#[[EXPR3]]]} : i64

#di_file = #llvm.di_file<"not" in "existence">
#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #di_file, producer = "MLIR", isOptimized = true, emissionKind = Full>
#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "uint64_t", sizeInBits = 64, encoding = DW_ATE_unsigned>
llvm.mlir.global external @global_with_expr1() {addr_space = 0 : i32, dbg_expr = [#llvm.di_global_variable_expression<var = <scope = #di_compile_unit, name = "global_with_expr_1", linkageName = "global_with_expr_1", file = #di_file, line = 370, type = #di_basic_type, isLocalToUnit = true, isDefined = true, alignInBits = 8>, expr = <>>]} : i64
llvm.mlir.global external @global_with_expr2() {addr_space = 0 : i32, dbg_expr = [#llvm.di_global_variable_expression<var = <scope = #di_compile_unit, name = "global_with_expr_2", linkageName = "global_with_expr_2", file = #di_file, line = 371, type = #di_basic_type, isLocalToUnit = true, isDefined = true, alignInBits = 8>, expr = <[DW_OP_push_object_address, DW_OP_deref]>>]} : i64
llvm.mlir.global external @global_with_expr3() {addr_space = 0 : i32, dbg_expr = [#llvm.di_global_variable_expression<var = <scope = #di_compile_unit, name = "global_with_expr_3", linkageName = "global_with_expr_3", file = #di_file, line = 372, type = #di_basic_type, isLocalToUnit = true, isDefined = true, alignInBits = 8>, expr = <[DW_OP_LLVM_arg(0), DW_OP_LLVM_arg(1), DW_OP_plus]>>]} : i64
llvm.mlir.global external @global_with_expr4() {addr_space = 0 : i32, dbg_expr = [#llvm.di_global_variable_expression<var = <scope = #di_compile_unit, name = "global_with_expr_4", linkageName = "global_with_expr_4", file = #di_file, line = 373, type = #di_basic_type, isLocalToUnit = true, isDefined = true, alignInBits = 8>, expr = <[DW_OP_LLVM_convert(16, DW_ATE_signed)]>>]} : i64
