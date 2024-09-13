// Tablegen tests for the automatic generation of OpenMP clause operand
// structure definitions.

// Run tablegen to generate OmpCommon.td in temp directory first.
// RUN: mkdir -p %t/mlir/Dialect/OpenMP
// RUN: mlir-tblgen --gen-directive-decl --directives-dialect=OpenMP \
// RUN:   %S/../../../llvm/include/llvm/Frontend/OpenMP/OMP.td \
// RUN:   -I %S/../../../llvm/include > %t/mlir/Dialect/OpenMP/OmpCommon.td

// RUN: mlir-tblgen -gen-openmp-clause-ops -I %S/../../include -I %t %s 2>&1 | FileCheck %s

include "mlir/Dialect/OpenMP/OpenMPOpBase.td"


def OpenMP_MyFirstClause : OpenMP_Clause<
    /*isRequired=*/false, /*skipTraits=*/false, /*skipArguments=*/false,
    /*skipAssemblyFormat=*/false, /*skipDescription=*/false,
    /*skipExtraClassDeclaration=*/false> {
  let arguments = (ins
    // Simple attributes
    I32Attr:$int_attr,
    TypeAttr:$type_attr,
    DeclareTargetAttr:$omp_attr,

    // Array attributes
    F32ArrayAttr:$float_array_attr,
    StrArrayAttr:$str_array_attr,
    AnyIntElementsAttr:$anyint_elems_attr,
    RankedF32ElementsAttr<[3, 4, 5]>:$float_nd_elems_attr,

    // Optional attributes
    OptionalAttr<BoolAttr>:$opt_bool_attr,
    OptionalAttr<I64ArrayAttr>:$opt_int_array_attr,
    OptionalAttr<DenseI8ArrayAttr>:$opt_int_elems_attr,

    // Multi-level composition
    ConfinedAttr<OptionalAttr<I64Attr>, [IntMinValue<0>]>:$complex_opt_int_attr,

    // ElementsAttrBase-related edge cases.
    // CHECK: warning: could not infer array-like attribute element type for argument 'elements_attr', will use bare `storageType`
    ElementsAttr:$elements_attr,
    // CHECK: warning: could not infer array-like attribute element type for argument 'string_elements_attr', will use bare `storageType`
    StringElementsAttr:$string_elements_attr
  );
}
// CHECK:      struct MyFirstClauseOps {
// CHECK-NEXT:   ::mlir::IntegerAttr intAttr;
// CHECK-NEXT:   ::mlir::TypeAttr typeAttr;
// CHECK-NEXT:   ::mlir::omp::DeclareTargetAttr ompAttr;

// CHECK-NEXT:   ::llvm::SmallVector<::mlir::Attribute> floatArrayAttr;
// CHECK-NEXT:   ::llvm::SmallVector<::mlir::Attribute> strArrayAttr;
// CHECK-NEXT:   ::llvm::SmallVector<::llvm::APInt> anyintElemsAttr;
// CHECK-NEXT:   ::llvm::SmallVector<::llvm::APFloat> floatNdElemsAttr;

// CHECK-NEXT:   ::mlir::BoolAttr optBoolAttr;
// CHECK-NEXT:   ::llvm::SmallVector<::mlir::Attribute> optIntArrayAttr;
// CHECK-NEXT:   ::llvm::SmallVector<int8_t> optIntElemsAttr;

// CHECK-NEXT:   ::mlir::IntegerAttr complexOptIntAttr;

// CHECK-NEXT:   ::mlir::ElementsAttr elementsAttr;
// CHECK-NEXT:   ::mlir::DenseElementsAttr stringElementsAttr;
// CHECK-NEXT: }

def OpenMP_MySecondClause : OpenMP_Clause<
    /*isRequired=*/false, /*skipTraits=*/false, /*skipArguments=*/false,
    /*skipAssemblyFormat=*/false, /*skipDescription=*/false,
    /*skipExtraClassDeclaration=*/false> {
  let arguments = (ins
    I32:$int_val,
    Optional<AnyType>:$opt_any_val,
    Variadic<Index>:$variadic_index_val
  );
}
// CHECK:      struct MySecondClauseOps {
// CHECK-NEXT:   ::mlir::Value intVal;
// CHECK-NEXT:   ::mlir::Value optAnyVal;
// CHECK-NEXT:   ::llvm::SmallVector<::mlir::Value> variadicIndexVal;
// CHECK-NEXT: }

def OpenMP_MyFirstOp : OpenMP_Op<"op", clauses=[OpenMP_MyFirstClause]>;
// CHECK: using MyFirstOperands = detail::Clauses<MyFirstClauseOps>;

def OpenMP_MySecondOp : OpenMP_Op<"op", clauses=[OpenMP_MyFirstClause, OpenMP_MySecondClause]>;
// CHECK: using MySecondOperands = detail::Clauses<MyFirstClauseOps, MySecondClauseOps>;
