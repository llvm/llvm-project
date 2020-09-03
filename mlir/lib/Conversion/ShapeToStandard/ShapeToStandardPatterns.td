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

// Convert `index_to_size` and `size_to_index` to no-ops as sizes will be
// represented as indices.
def IndexToSizeOpConversion : Pat<
    (Shape_IndexToSizeOp $arg),
    (replaceWithValue $arg)>;
def SizeToIndexOpConversion : Pat<
    (Shape_SizeToIndexOp $arg),
    (replaceWithValue $arg)>;
