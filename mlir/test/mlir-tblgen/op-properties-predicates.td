// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s

include "mlir/IR/Constraints.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/Properties.td"

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "foobar";
}
class NS_Op<string mnemonic, list<Trait> traits = []> :
    Op<Test_Dialect, mnemonic, traits>;

def NonNegativeI64Prop : ConfinedProp<I64Prop,
  CPred<"$_self >= 0">, "non-negative int64_t">;

class NonEmptyArray<Property p> : ConfinedProp
    <ArrayProp<p>, Neg<CPred<"$_self.empty()">>,
    "non-empty array of " # p.summary>;

def OpWithPredicates : NS_Op<"op_with_predicates"> {
  let arguments = (ins
    NonNegativeI64Prop:$scalar,
    OptionalProp<NonNegativeI64Prop>:$optional,
    DefaultValuedProp<NonNegativeI64Prop, "0">:$defaulted,
    ConfinedProp<NonNegativeI64Prop,
      CPred<"$_self <= 5">, "between 0 and 5">:$moreConstrained,
    ArrayProp<NonNegativeI64Prop>:$array,
    NonEmptyArray<I64Prop>:$non_empty_unconstrained,
    NonEmptyArray<NonNegativeI64Prop>:$non_empty_constrained,
    // Test applying predicates when the fromStorage() on the optional<> isn't trivial.
    OptionalProp<NonEmptyArray<NonNegativeI64Prop>>:$non_empty_optional,
    I64Prop:$unconstrained
  );
}

// CHECK-LABEL: static ::llvm::LogicalResult __mlir_ods_local_prop_constraint_op2Dproperties2Dpredicates1
// CHECK-NEXT: int64_t prop
// CHECK-NEXT: if (!((prop >= 0)))
// CHECK: failed to satisfy constraint: non-negative int64_t

// CHECK-LABEL: static ::llvm::LogicalResult __mlir_ods_local_prop_constraint_op2Dproperties2Dpredicates2
// CHECK-NEXT: std::optional<int64_t> prop
// CHECK-NEXT: if (!(((!prop.has_value())) || (((*(prop)) >= 0))))
// CHECK: failed to satisfy constraint: optional non-negative int64_t

// CHECK-LABEL: static ::llvm::LogicalResult __mlir_ods_local_prop_constraint_op2Dproperties2Dpredicates3
// CHECK-NEXT: int64_t prop
// CHECK-NEXT: if (!(((prop >= 0)) && ((prop <= 5))))
// CHECK: failed to satisfy constraint: between 0 and 5

// CHECK-LABEL: static ::llvm::LogicalResult __mlir_ods_local_prop_constraint_op2Dproperties2Dpredicates4
// CHECK-NEXT: ::llvm::ArrayRef<int64_t> prop
// CHECK-NEXT: (!(::llvm::all_of(prop, [](const int64_t& baseStore) -> bool { return [](int64_t baseIface) -> bool { return ((baseIface >= 0)); }(baseStore); })))
// CHECK: failed to satisfy constraint: array of non-negative int64_t

// CHECK-LABEL: static ::llvm::LogicalResult __mlir_ods_local_prop_constraint_op2Dproperties2Dpredicates5
// CHECK-NEXT: ::llvm::ArrayRef<int64_t> prop
// CHECK-NEXT: if (!(!((prop.empty()))))
// CHECK: failed to satisfy constraint: non-empty array of int64_t

// CHECK-LABEL: static ::llvm::LogicalResult __mlir_ods_local_prop_constraint_op2Dproperties2Dpredicates6
// CHECK-NEXT: ::llvm::ArrayRef<int64_t> prop
// CHECX-NEXT: if (!((::llvm::all_of(prop, [](const int64_t& baseStore) -> bool { return [](int64_t baseIface) -> bool { return ((baseIface >= 0)); }(baseStore); })) && (!((prop.empty())))))
// CHECK: failed to satisfy constraint: non-empty array of non-negative int64_t

// CHECK-LABEL: static ::llvm::LogicalResult __mlir_ods_local_prop_constraint_op2Dproperties2Dpredicates7
// CHECK-NEXT: std::optional<::llvm::ArrayRef<int64_t>> prop
// CHECK-NEXT: if (!(((!prop.has_value())) || ((::llvm::all_of((*(prop)), [](const int64_t& baseStore) -> bool { return [](int64_t baseIface) -> bool { return ((baseIface >= 0)); }(baseStore); })) && (!(((*(prop)).empty()))))))
// CHECK: failed to satisfy constraint: optional non-empty array of non-negative int64_

// CHECK-LABEL: OpWithPredicates::verifyInvariantsImpl()
// Note: for test readability, we capture [[maybe_unused]] into the variable maybe_unused
// CHECK: [[maybe_unused:\[\[maybe_unused\]\]]] int64_t tblgen_scalar = this->getScalar();
// CHECK: if (::mlir::failed(__mlir_ods_local_prop_constraint_op2Dproperties2Dpredicates1(*this, tblgen_scalar, "scalar")))
// CHECK-NEXT:  return ::mlir::failure()

// CHECK: [[maybe_unused]] std::optional<int64_t> tblgen_optional = this->getOptional();
// CHECK: if (::mlir::failed(__mlir_ods_local_prop_constraint_op2Dproperties2Dpredicates2(*this, tblgen_optional, "optional")))

// COM: The predicates for "scalar" and "defaulted" are uniqued
// CHECK: [[maybe_unused]] int64_t tblgen_defaulted = this->getDefaulted();
// CHECK: if (::mlir::failed(__mlir_ods_local_prop_constraint_op2Dproperties2Dpredicates1(*this, tblgen_defaulted, "defaulted")))

// CHECK: [[maybe_unused]] int64_t tblgen_moreConstrained = this->getMoreConstrained();
// CHECK: if (::mlir::failed(__mlir_ods_local_prop_constraint_op2Dproperties2Dpredicates3(*this, tblgen_moreConstrained, "moreConstrained")))

// CHECK: [[maybe_unused]] ::llvm::ArrayRef<int64_t> tblgen_array = this->getArray();
// CHECK: if (::mlir::failed(__mlir_ods_local_prop_constraint_op2Dproperties2Dpredicates4(*this, tblgen_array, "array")))

// CHECK: [[maybe_unused]] ::llvm::ArrayRef<int64_t> tblgen_nonEmptyUnconstrained = this->getNonEmptyUnconstrained();
// CHECK: if (::mlir::failed(__mlir_ods_local_prop_constraint_op2Dproperties2Dpredicates5(*this, tblgen_nonEmptyUnconstrained, "non_empty_unconstrained")))

// CHECK: [[maybe_unused]] ::llvm::ArrayRef<int64_t> tblgen_nonEmptyConstrained = this->getNonEmptyConstrained();
// CHECK: if (::mlir::failed(__mlir_ods_local_prop_constraint_op2Dproperties2Dpredicates6(*this, tblgen_nonEmptyConstrained, "non_empty_constrained")))

// CHECK: [[maybe_unused]] std::optional<::llvm::ArrayRef<int64_t>> tblgen_nonEmptyOptional = this->getNonEmptyOptional();
// CHECK: if (::mlir::failed(__mlir_ods_local_prop_constraint_op2Dproperties2Dpredicates7(*this, tblgen_nonEmptyOptional, "non_empty_optional")))

// CHECK-NOT: int64_t tblgen_unconstrained
// CHECK: return ::mlir::success();
