// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
}
class NS_Op<string mnemonic, list<Trait> traits> :
    Op<Test_Dialect, mnemonic, traits>;

def I32OrF32 : Type<CPred<"$_self.isInteger(32) || $_self.isF32()">,
                    "32-bit integer or floating-point type">;

def OpA : NS_Op<"op_for_CPred_containing_multiple_same_placeholder", []> {
  let arguments = (ins I32OrF32:$x);
  let results = (outs Variadic<I32OrF32>:$y);
}

// CHECK: static ::mlir::LogicalResult [[$INTEGER_FLOAT_CONSTRAINT:__mlir_ods_local_type_constraint.*]](
// CHECK:      if (!((type.isInteger(32) || type.isF32()))) {
// CHECK-NEXT:   return op->emitOpError(valueKind) << " #" << valueIndex
// CHECK-NEXT:       << " must be 32-bit integer or floating-point type, but got " << type;

// Check there is no verifier with same predicate generated.
// CHECK-NOT:  if (!((type.isInteger(32) || type.isF32()))) {
// CHECK-NOT:    return op->emitOpError(valueKind) << " #" << valueIndex
// CHECK-NOT.        << " must be 32-bit integer or floating-point type, but got " << type;

// CHECK: static ::mlir::LogicalResult [[$TENSOR_CONSTRAINT:__mlir_ods_local_type_constraint.*]](
// CHECK:       if (!(((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
// CHECK-NEXT:    return op->emitOpError(valueKind) << " #" << valueIndex
// CHECK-NEXT:        << " must be tensor of any type values, but got " << type;

// CHECK: static ::mlir::LogicalResult [[$TENSOR_INTEGER_FLOAT_CONSTRAINT:__mlir_ods_local_type_constraint.*]](
// CHECK:       if (!(((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isSignlessInteger(32))); }(type.cast<::mlir::ShapedType>().getElementType())))) {
// CHECK-NEXT:    return op->emitOpError(valueKind) << " #" << valueIndex
// CHECK-NEXT:        << " must be tensor of 32-bit float or 32-bit signless integer values, but got " << type;

// CHECK-LABEL: OpA::verify
// CHECK: auto valueGroup0 = getODSOperands(0);
// CHECK: for (auto v : valueGroup0) {
// CHECK:   if (::mlir::failed([[$INTEGER_FLOAT_CONSTRAINT]]

def OpB : NS_Op<"op_for_And_PredOpTrait", [
    PredOpTrait<"both first and second holds",
                And<[CPred<"first">, CPred<"second">]>>]> {
}

// CHECK-LABEL: OpB::verify
// CHECK: if (!(((first)) && ((second))))

def OpF : NS_Op<"op_for_int_min_val", []> {
  let arguments = (ins Confined<I32Attr, [IntMinValue<10>]>:$attr);
}

// CHECK-LABEL: OpFAdaptor::verify
// CHECK:       (tblgen_attr.cast<::mlir::IntegerAttr>().getInt() >= 10)
// CHECK-NEXT:  "attribute 'attr' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 10"

def OpFX : NS_Op<"op_for_int_max_val", []> {
  let arguments = (ins Confined<I32Attr, [IntMaxValue<10>]>:$attr);
}

// CHECK-LABEL: OpFXAdaptor::verify
// CHECK:       (tblgen_attr.cast<::mlir::IntegerAttr>().getInt() <= 10)
// CHECK-NEXT:  "attribute 'attr' failed to satisfy constraint: 32-bit signless integer attribute whose maximum value is 10"

def OpG : NS_Op<"op_for_arr_min_count", []> {
  let arguments = (ins Confined<ArrayAttr, [ArrayMinCount<8>]>:$attr);
}

// CHECK-LABEL: OpGAdaptor::verify
// CHECK:       (tblgen_attr.cast<::mlir::ArrayAttr>().size() >= 8)
// CHECK-NEXT:  "attribute 'attr' failed to satisfy constraint: array attribute with at least 8 elements"

def OpH : NS_Op<"op_for_arr_value_at_index", []> {
  let arguments = (ins Confined<ArrayAttr, [IntArrayNthElemEq<0, 8>]>:$attr);
}

// CHECK-LABEL: OpHAdaptor::verify
// CHECK: (((tblgen_attr.cast<::mlir::ArrayAttr>().size() > 0)) && ((tblgen_attr.cast<::mlir::ArrayAttr>()[0].cast<::mlir::IntegerAttr>().getInt() == 8)))))
// CHECK-NEXT:  "attribute 'attr' failed to satisfy constraint: array attribute whose 0-th element must be 8"

def OpI: NS_Op<"op_for_arr_min_value_at_index", []> {
  let arguments = (ins Confined<ArrayAttr, [IntArrayNthElemMinValue<0, 8>]>:$attr);
}

// CHECK-LABEL: OpIAdaptor::verify
// CHECK: (((tblgen_attr.cast<::mlir::ArrayAttr>().size() > 0)) && ((tblgen_attr.cast<::mlir::ArrayAttr>()[0].cast<::mlir::IntegerAttr>().getInt() >= 8)))))
// CHECK-NEXT: "attribute 'attr' failed to satisfy constraint: array attribute whose 0-th element must be at least 8"

def OpJ: NS_Op<"op_for_TCopVTEtAreSameAt", [
                PredOpTrait<"operands indexed at 0, 2, 3 should all have "
                 "the same type", TCopVTEtAreSameAt<[0, 2, 3]>>]> {
  let arguments = (ins
    AnyTensor:$a,
    AnyTensor:$b,
    AnyTensor:$c,
    AnyTensor:$d,
    AnyTensor:$e
  );
}

// CHECK-LABEL: OpJAdaptor::verify
// CHECK:      ::llvm::is_splat(::llvm::map_range(
// CHECK-SAME:   ::mlir::ArrayRef<unsigned>({0, 2, 3}),
// CHECK-SAME:   [this](unsigned i) { return getElementTypeOrSelf(this->getOperand(i)); }))
// CHECK: "failed to verify that operands indexed at 0, 2, 3 should all have the same type"

def OpK : NS_Op<"op_for_AnyTensorOf", []> {
  let arguments = (ins TensorOf<[F32, I32]>:$x);
}

// CHECK-LABEL: OpK::verify
// CHECK: auto valueGroup0 = getODSOperands(0);
// CHECK: for (auto v : valueGroup0) {
// CHECK: if (::mlir::failed([[$TENSOR_INTEGER_FLOAT_CONSTRAINT]]

def OpL : NS_Op<"op_for_StringEscaping", []> {
  let arguments = (ins
    StringBasedAttr<CPred<"$_self.cast<StringAttr>().getValue() == \"foo\"">,
                    "only value \"foo\" is allowed">:$s
  );
}

// CHECK-LABEL: OpLAdaptor::verify
// CHECK: getValue() == "foo"
// CHECK-NEXT: only value \"foo\" is allowed
