# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import sparse_tensor as st

def run(f):
  print("\nTEST:", f.__name__)
  f()
  return f


# CHECK-LABEL: TEST: testEncodingAttr1D
@run
def testEncodingAttr1D():
  with Context() as ctx:
    parsed = Attribute.parse(
      '#sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], '
      'pointerBitWidth = 16, indexBitWidth = 32 }>')
    print(parsed)

    casted = st.EncodingAttr(parsed)
    # CHECK: equal: True
    print(f"equal: {casted == parsed}")

    # CHECK: dim_level_types: [<DimLevelType.compressed: 1>]
    print(f"dim_level_types: {casted.dim_level_types}")
    # CHECK: dim_ordering: None
    # Note that for 1D, the ordering is None, which exercises several special
    # cases.
    print(f"dim_ordering: {casted.dim_ordering}")
    # CHECK: pointer_bit_width: 16
    print(f"pointer_bit_width: {casted.pointer_bit_width}")
    # CHECK: index_bit_width: 32
    print(f"index_bit_width: {casted.index_bit_width}")

    created = st.EncodingAttr.get(casted.dim_level_types, None, 16, 32)
    print(created)
    # CHECK: created_equal: True
    print(f"created_equal: {created == casted}")

    # Verify that the factory creates an instance of the proper type.
    # CHECK: is_proper_instance: True
    print(f"is_proper_instance: {isinstance(created, st.EncodingAttr)}")
    # CHECK: created_pointer_bit_width: 16
    print(f"created_pointer_bit_width: {created.pointer_bit_width}")


# CHECK-LABEL: TEST: testEncodingAttr2D
@run
def testEncodingAttr2D():
  with Context() as ctx:
    parsed = Attribute.parse(
      '#sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], '
      'dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, '
      'pointerBitWidth = 16, indexBitWidth = 32 }>')
    print(parsed)

    casted = st.EncodingAttr(parsed)
    # CHECK: equal: True
    print(f"equal: {casted == parsed}")

    # CHECK: dim_level_types: [<DimLevelType.dense: 0>, <DimLevelType.compressed: 1>]
    print(f"dim_level_types: {casted.dim_level_types}")
    # CHECK: dim_ordering: (d0, d1) -> (d0, d1)
    print(f"dim_ordering: {casted.dim_ordering}")
    # CHECK: pointer_bit_width: 16
    print(f"pointer_bit_width: {casted.pointer_bit_width}")
    # CHECK: index_bit_width: 32
    print(f"index_bit_width: {casted.index_bit_width}")

    created = st.EncodingAttr.get(casted.dim_level_types, casted.dim_ordering,
        16, 32)
    print(created)
    # CHECK: created_equal: True
    print(f"created_equal: {created == casted}")


# CHECK-LABEL: TEST: testEncodingAttrOnTensor
@run
def testEncodingAttrOnTensor():
  with Context() as ctx, Location.unknown():
    encoding = st.EncodingAttr(Attribute.parse(
      '#sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], '
      'pointerBitWidth = 16, indexBitWidth = 32 }>'))
    tt = RankedTensorType.get((1024,), F32Type.get(), encoding=encoding)
    # CHECK: tensor<1024xf32, #sparse_tensor
    print(tt)
    # CHECK: #sparse_tensor.encoding
    print(tt.encoding)
    assert tt.encoding == encoding
