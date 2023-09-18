// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
  let usePropertiesForAttributes = 0;
}

class NS_Op<string mnemonic, list<Trait> traits = []> :
    Op<Test_Dialect, mnemonic, traits>;

/// Test unique'ing of type, attribute, successor, and region constraints.

def ATypePred : CPred<"typePred($_self, $_op)">;
def AType : Type<ATypePred, "a type">;
def OtherType : Type<ATypePred, "another type">;

def AnAttrPred : CPred<"attrPred($_self, $_op)">;
def AnAttr : Attr<AnAttrPred, "an attribute">;
def OtherAttr : Attr<AnAttrPred, "another attribute">;

def ASuccessorPred : CPred<"successorPred($_self, $_op)">;
def ASuccessor : Successor<ASuccessorPred, "a successor">;
def OtherSuccessor : Successor<ASuccessorPred, "another successor">;

def ARegionPred : CPred<"regionPred($_self, $_op)">;
def ARegion : Region<ARegionPred, "a region">;
def OtherRegion : Region<ARegionPred, "another region">;

// OpA and OpB have the same type, attribute, successor, and region constraints.

def OpA : NS_Op<"op_a"> {
  let arguments = (ins AType:$a, AnAttr:$b);
  let results = (outs AType:$ret);
  let successors = (successor ASuccessor:$c);
  let regions = (region ARegion:$d);
}

def OpB : NS_Op<"op_b"> {
  let arguments = (ins AType:$a, AnAttr:$b);
  let successors = (successor ASuccessor:$c);
  let regions = (region ARegion:$d);
}

// OpC has the same type, attribute, successor, and region predicates but has
// difference descriptions for them.

def OpC : NS_Op<"op_c"> {
  let arguments = (ins OtherType:$a, OtherAttr:$b);
  let results = (outs OtherType:$ret);
  let successors = (successor OtherSuccessor:$c);
  let regions = (region OtherRegion:$d);
}

/// Test that a type contraint was generated.
// CHECK:    static ::mlir::LogicalResult [[$A_TYPE_CONSTRAINT:__mlir_ods_local_type_constraint.*]](
// CHECK:      if (!((typePred(type, *op)))) {
// CHECK-NEXT:   return op->emitOpError(valueKind) << " #" << valueIndex
// CHECK-NEXT:       << " must be a type, but got " << type;

/// Test that duplicate type constraint was not generated.
// CHECK-NOT:        << " must be a type, but got " << type;

/// Test that a type constraint with a different description was generated.
// CHECK:    static ::mlir::LogicalResult [[$O_TYPE_CONSTRAINT:__mlir_ods_local_type_constraint.*]](
// CHECK:      if (!((typePred(type, *op)))) {
// CHECK-NEXT:   return op->emitOpError(valueKind) << " #" << valueIndex
// CHECK-NEXT:       << " must be another type, but got " << type;

/// Test that an attribute contraint was generated.
// CHECK:    static ::mlir::LogicalResult [[$A_ATTR_CONSTRAINT:__mlir_ods_local_attr_constraint.*]](
// CHECK:      if (attr && !((attrPred(attr, *op))))
// CHECK-NEXT:   return getDiag() << "attribute '" << attrName
// CHECK-NEXT:       << "' failed to satisfy constraint: an attribute";

/// Test that duplicate attribute constraint was not generated.
// CHECK-NOT:        << "' failed to satisfy constraint: an attribute";

/// Test that a attribute constraint with a different description was generated.
// CHECK:    static ::mlir::LogicalResult [[$O_ATTR_CONSTRAINT:__mlir_ods_local_attr_constraint.*]](
// CHECK:    static ::mlir::LogicalResult [[$O_ATTR_CONSTRAINT:__mlir_ods_local_attr_constraint.*]](
// CHECK:      if (attr && !((attrPred(attr, *op))))
// CHECK-NEXT:   return getDiag() << "attribute '" << attrName
// CHECK-NEXT:       << "' failed to satisfy constraint: another attribute";

/// Test that a successor contraint was generated.
// CHECK:    static ::mlir::LogicalResult [[$A_SUCCESSOR_CONSTRAINT:__mlir_ods_local_successor_constraint.*]](
// CHECK:      if (!((successorPred(successor, *op)))) {
// CHECK-NEXT:   return op->emitOpError("successor #") << successorIndex << " ('"
// CHECK-NEXT:       << successorName << ")' failed to verify constraint: a successor";

/// Test that duplicate successor constraint was not generated.
// CHECK-NOT:        << successorName << ")' failed to verify constraint: a successor";

/// Test that a successor constraint with a different description was generated.
// CHECK:    static ::mlir::LogicalResult [[$O_SUCCESSOR_CONSTRAINT:__mlir_ods_local_successor_constraint.*]](
// CHECK:      if (!((successorPred(successor, *op)))) {
// CHECK-NEXT:   return op->emitOpError("successor #") << successorIndex << " ('"
// CHECK-NEXT:       << successorName << ")' failed to verify constraint: another successor";

/// Test that a region contraint was generated.
// CHECK:    static ::mlir::LogicalResult [[$A_REGION_CONSTRAINT:__mlir_ods_local_region_constraint.*]](
// CHECK:      if (!((regionPred(region, *op)))) {
// CHECK-NEXT:   return op->emitOpError("region #") << regionIndex
// CHECK-NEXT:       << (regionName.empty() ? " " : " ('" + regionName + "') ")
// CHECK-NEXT:       << "failed to verify constraint: a region";

/// Test that duplicate region constraint was not generated.
// CHECK-NOT:        << "failed to verify constraint: a region";

/// Test that a region constraint with a different description was generated.
// CHECK:    static ::mlir::LogicalResult [[$O_REGION_CONSTRAINT:__mlir_ods_local_region_constraint.*]](
// CHECK:      if (!((regionPred(region, *op)))) {
// CHECK-NEXT:   return op->emitOpError("region #") << regionIndex
// CHECK-NEXT:       << (regionName.empty() ? " " : " ('" + regionName + "') ")
// CHECK-NEXT:       << "failed to verify constraint: another region";

/// Test that the uniqued constraints are being used.
// CHECK-LABEL: OpA::verify
// CHECK:         ::mlir::Attribute [[$B_ATTR:.*b]];
// CHECK:         if (::mlir::failed([[$A_ATTR_CONSTRAINT]](*this, [[$B_ATTR]], "b")))
// CHECK-NEXT:      return ::mlir::failure();
// CHECK:         auto [[$A_VALUE_GROUP:.*]] = getODSOperands(0);
// CHECK:         for (auto [[$A_VALUE:.*]] : [[$A_VALUE_GROUP]])
// CHECK-NEXT:      if (::mlir::failed([[$A_TYPE_CONSTRAINT]](*this, [[$A_VALUE]].getType(), "operand", index++)))
// CHECK-NEXT:        return ::mlir::failure();
// CHECK:         auto [[$RET_VALUE_GROUP:.*]] = getODSResults(0);
// CHECK:         for (auto [[$RET_VALUE:.*]] : [[$RET_VALUE_GROUP]])
// CHECK-NEXT:      if (::mlir::failed([[$A_TYPE_CONSTRAINT]](*this, [[$RET_VALUE]].getType(), "result", index++)))
// CHECK-NEXT:        return ::mlir::failure();
// CHECK:         for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
// CHECK-NEXT:      if (::mlir::failed([[$A_REGION_CONSTRAINT]](*this, region, "d", index++)))
// CHECK-NEXT:        return ::mlir::failure();
// CHECK:         for (auto *successor : ::llvm::MutableArrayRef(c()))
// CHECK-NEXT:      if (::mlir::failed([[$A_SUCCESSOR_CONSTRAINT]](*this, successor, "c", index++)))
// CHECK-NEXT:        return ::mlir::failure();

/// Test that the op with the same predicates but different with descriptions
/// uses the different constraints.
// CHECK-LABEL: OpC::verify
// CHECK:         ::mlir::Attribute [[$B_ATTR:.*b]];
// CHECK:         if (::mlir::failed([[$O_ATTR_CONSTRAINT]](*this, [[$B_ATTR]], "b")))
// CHECK-NEXT:      return ::mlir::failure();
// CHECK:         auto [[$A_VALUE_GROUP:.*]] = getODSOperands(0);
// CHECK:         for (auto [[$A_VALUE:.*]] : [[$A_VALUE_GROUP]])
// CHECK-NEXT:      if (::mlir::failed([[$O_TYPE_CONSTRAINT]](*this, [[$A_VALUE]].getType(), "operand", index++)))
// CHECK-NEXT:        return ::mlir::failure();
// CHECK:         auto [[$RET_VALUE_GROUP:.*]] = getODSResults(0);
// CHECK:         for (auto [[$RET_VALUE:.*]] : [[$RET_VALUE_GROUP]])
// CHECK-NEXT:      if (::mlir::failed([[$O_TYPE_CONSTRAINT]](*this, [[$RET_VALUE]].getType(), "result", index++)))
// CHECK-NEXT:        return ::mlir::failure();
// CHECK:         for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
// CHECK-NEXT:      if (::mlir::failed([[$O_REGION_CONSTRAINT]](*this, region, "d", index++)))
// CHECK-NEXT:        return ::mlir::failure();
// CHECK:         for (auto *successor : ::llvm::MutableArrayRef(c()))
// CHECK-NEXT:      if (::mlir::failed([[$O_SUCCESSOR_CONSTRAINT]](*this, successor, "c", index++)))
// CHECK-NEXT:        return ::mlir::failure();
