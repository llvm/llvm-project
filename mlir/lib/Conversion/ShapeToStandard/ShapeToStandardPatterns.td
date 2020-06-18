include "mlir/Dialect/Shape/IR/ShapeOps.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"

// Convert `from_extent_tensor` and `to_extent_tensor` to no-ops as shapes will
// be represented as extent tensors.
def FromExtentTensorOpConversion : Pat<
    (Shape_FromExtentTensorOp $input),
    (replaceWithValue $input)>;
def ToExtentTensorOpConversion : Pat<
    (Shape_ToExtentTensorOp $input),
    (replaceWithValue $input)>;

