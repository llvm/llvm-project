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

def NonNegativeI64Property : ConfinedProperty<I64Property,
  CPred<"$_self >= 0">, "non-negative int64_t">;

class NonEmptyArray<Property p> : ConfinedProperty
    <ArrayProperty<p>, Neg<CPred<"$_self.empty()">>,
    "non-empty array of " # p.summary>;

def OpWithPredicates : NS_Op<"op_with_predicates"> {
  let arguments = (ins
    NonNegativeI64Property:$scalar,
    OptionalProperty<NonNegativeI64Property>:$optional,
    DefaultValuedProperty<NonNegativeI64Property, "0">:$defaulted,
    ConfinedProperty<NonNegativeI64Property,
      CPred<"$_self <= 5">, "between 0 and 5">:$moreConstrained,
    ArrayProperty<NonNegativeI64Property>:$array,
    NonEmptyArray<I64Property>:$non_empty_unconstrained,
    NonEmptyArray<NonNegativeI64Property>:$non_empty_constrained,
    // Test applying predicates when the fromStorage() on the optional<> isn't trivial.
    OptionalProperty<NonEmptyArray<NonNegativeI64Property>>:$non_empty_optional,
    I64Property:$unconstrained
  );
}

// CHECK-LABEL: OpWithPredicates::verifyInvariantsImpl()
// CHECK: int64_t tblgen_scalar = this->getScalar(); (void)tblgen_scalar;
// CHECK: if (!((tblgen_scalar >= 0)))
// CHECK-NEXT:  return emitOpError("property 'scalar' failed to satiisfy constraint: non-negative int64_t");

// CHECK: std::optional<int64_t> tblgen_optional = this->getOptional(); (void)tblgen_optional;
// CHECK: if (!(((!tblgen_optional.has_value())) || (((*(tblgen_optional)) >= 0))))
// CHECK-NEXT:   return emitOpError("property 'optional' failed to satiisfy constraint: optional non-negative int64_t");

// CHECK: int64_t tblgen_defaulted = this->getDefaulted(); (void)tblgen_defaulted;
// CHECK: if (!((tblgen_defaulted >= 0)))
// CHECK-NEXT:  return emitOpError("property 'defaulted' failed to satiisfy constraint: non-negative int64_t");

// CHECK: int64_t tblgen_moreConstrained = this->getMoreConstrained(); (void)tblgen_moreConstrained;
// CHECK: if (!(((tblgen_moreConstrained >= 0)) && ((tblgen_moreConstrained <= 5))))
// CHECK-NEXT:  return emitOpError("property 'moreConstrained' failed to satiisfy constraint: between 0 and 5");

// CHECK: ::llvm::ArrayRef<int64_t> tblgen_array = this->getArray(); (void)tblgen_array;
// CHECK: if (!(::llvm::all_of(tblgen_array, [](const int64_t& baseStore) -> bool { return [](int64_t baseIface) -> bool { return ((baseIface >= 0)); }(baseStore); })))
// CHECK-NEXT: return emitOpError("property 'array' failed to satiisfy constraint: array of non-negative int64_t");

// CHECK: ::llvm::ArrayRef<int64_t> tblgen_nonEmptyUnconstrained = this->getNonEmptyUnconstrained(); (void)tblgen_nonEmptyUnconstrained;
// CHECK: if (!(!((tblgen_nonEmptyUnconstrained.empty()))))
// CHECK-NEXT: return emitOpError("property 'non_empty_unconstrained' failed to satiisfy constraint: non-empty array of int64_t");

// CHECK: ::llvm::ArrayRef<int64_t> tblgen_nonEmptyConstrained = this->getNonEmptyConstrained(); (void)tblgen_nonEmptyConstrained;
// CHECK: if (!((::llvm::all_of(tblgen_nonEmptyConstrained, [](const int64_t& baseStore) -> bool { return [](int64_t baseIface) -> bool { return ((baseIface >= 0)); }(baseStore); })) && (!((tblgen_nonEmptyConstrained.empty())))))
// CHECK-NEXT:   return emitOpError("property 'non_empty_constrained' failed to satiisfy constraint: non-empty array of non-negative int64_t");

// CHECK: std::optional<::llvm::ArrayRef<int64_t>> tblgen_nonEmptyOptional = this->getNonEmptyOptional(); (void)tblgen_nonEmptyOptional;
// CHECK: (!(((!tblgen_nonEmptyOptional.has_value())) || ((::llvm::all_of((*(tblgen_nonEmptyOptional)), [](const int64_t& baseStore) -> bool { return [](int64_t baseIface) -> bool { return ((baseIface >= 0)); }(baseStore); })) && (!(((*(tblgen_nonEmptyOptional)).empty()))))))
// CHECK-NEXT:   return emitOpError("property 'non_empty_optional' failed to satiisfy constraint: optional non-empty array of non-negative int64_t");

// CHECK-NOT: int64_t tblgen_unconstrained
// CHECK: return ::mlir::success();
