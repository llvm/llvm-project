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

// CHECK-LABEL: OpWithPredicates::verifyInvariantsImpl()
// Note: for test readibility, we capture [[maybe_unused]] into the variable maybe_unused
// CHECK: [[maybe_unused:\[\[maybe_unused\]\]]] int64_t tblgen_scalar = this->getScalar();
// CHECK: if (!((tblgen_scalar >= 0)))
// CHECK-NEXT:  return emitOpError("property 'scalar' failed to satisfy constraint: non-negative int64_t");

// CHECK: [[maybe_unused]] std::optional<int64_t> tblgen_optional = this->getOptional();
// CHECK: if (!(((!tblgen_optional.has_value())) || (((*(tblgen_optional)) >= 0))))
// CHECK-NEXT:   return emitOpError("property 'optional' failed to satisfy constraint: optional non-negative int64_t");

// CHECK: [[maybe_unused]] int64_t tblgen_defaulted = this->getDefaulted();
// CHECK: if (!((tblgen_defaulted >= 0)))
// CHECK-NEXT:  return emitOpError("property 'defaulted' failed to satisfy constraint: non-negative int64_t");

// CHECK: [[maybe_unused]] int64_t tblgen_moreConstrained = this->getMoreConstrained();
// CHECK: if (!(((tblgen_moreConstrained >= 0)) && ((tblgen_moreConstrained <= 5))))
// CHECK-NEXT:  return emitOpError("property 'moreConstrained' failed to satisfy constraint: between 0 and 5");

// CHECK: [[maybe_unused]] ::llvm::ArrayRef<int64_t> tblgen_array = this->getArray();
// CHECK: if (!(::llvm::all_of(tblgen_array, [](const int64_t& baseStore) -> bool { return [](int64_t baseIface) -> bool { return ((baseIface >= 0)); }(baseStore); })))
// CHECK-NEXT: return emitOpError("property 'array' failed to satisfy constraint: array of non-negative int64_t");

// CHECK: [[maybe_unused]] ::llvm::ArrayRef<int64_t> tblgen_nonEmptyUnconstrained = this->getNonEmptyUnconstrained();
// CHECK: if (!(!((tblgen_nonEmptyUnconstrained.empty()))))
// CHECK-NEXT: return emitOpError("property 'non_empty_unconstrained' failed to satisfy constraint: non-empty array of int64_t");

// CHECK: [[maybe_unused]] ::llvm::ArrayRef<int64_t> tblgen_nonEmptyConstrained = this->getNonEmptyConstrained();
// CHECK: if (!((::llvm::all_of(tblgen_nonEmptyConstrained, [](const int64_t& baseStore) -> bool { return [](int64_t baseIface) -> bool { return ((baseIface >= 0)); }(baseStore); })) && (!((tblgen_nonEmptyConstrained.empty())))))
// CHECK-NEXT:   return emitOpError("property 'non_empty_constrained' failed to satisfy constraint: non-empty array of non-negative int64_t");

// CHECK: [[maybe_unused]] std::optional<::llvm::ArrayRef<int64_t>> tblgen_nonEmptyOptional = this->getNonEmptyOptional();
// CHECK: (!(((!tblgen_nonEmptyOptional.has_value())) || ((::llvm::all_of((*(tblgen_nonEmptyOptional)), [](const int64_t& baseStore) -> bool { return [](int64_t baseIface) -> bool { return ((baseIface >= 0)); }(baseStore); })) && (!(((*(tblgen_nonEmptyOptional)).empty()))))))
// CHECK-NEXT:   return emitOpError("property 'non_empty_optional' failed to satisfy constraint: optional non-empty array of non-negative int64_t");

// CHECK-NOT: int64_t tblgen_unconstrained
// CHECK: return ::mlir::success();
