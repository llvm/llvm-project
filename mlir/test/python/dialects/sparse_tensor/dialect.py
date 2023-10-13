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
            "#sparse_tensor.encoding<{"
            "  map = (d0) -> (d0 : compressed),"
            "  posWidth = 16,"
            "  crdWidth = 32"
            "}>"
        )
        # CHECK: #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed), posWidth = 16, crdWidth = 32 }>
        print(parsed)

        casted = st.EncodingAttr(parsed)
        # CHECK: equal: True
        print(f"equal: {casted == parsed}")

        # CHECK: lvl_types: [<DimLevelType.compressed: 8>]
        print(f"lvl_types: {casted.lvl_types}")
        # CHECK: dim_to_lvl: None
        print(f"dim_to_lvl: {casted.dim_to_lvl}")
        # CHECK: pos_width: 16
        print(f"pos_width: {casted.pos_width}")
        # CHECK: crd_width: 32
        print(f"crd_width: {casted.crd_width}")

        created = st.EncodingAttr.get(casted.lvl_types, None, 0, 0)
        # CHECK: #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
        print(created)
        # CHECK: created_equal: False
        print(f"created_equal: {created == casted}")

        # Verify that the factory creates an instance of the proper type.
        # CHECK: is_proper_instance: True
        print(f"is_proper_instance: {isinstance(created, st.EncodingAttr)}")
        # CHECK: created_pos_width: 0
        print(f"created_pos_width: {created.pos_width}")


# CHECK-LABEL: TEST: testEncodingAttr2D
@run
def testEncodingAttr2D():
    with Context() as ctx:
        parsed = Attribute.parse(
            "#sparse_tensor.encoding<{"
            "  map = (d0, d1) -> (d1 : dense, d0 : compressed),"
            "  posWidth = 8,"
            "  crdWidth = 32"
            "}>"
        )
        # CHECK: #sparse_tensor.encoding<{ map = (d0, d1) -> (d1 : dense, d0 : compressed), posWidth = 8, crdWidth = 32 }>
        print(parsed)

        casted = st.EncodingAttr(parsed)
        # CHECK: equal: True
        print(f"equal: {casted == parsed}")

        # CHECK: lvl_types: [<DimLevelType.dense: 4>, <DimLevelType.compressed: 8>]
        print(f"lvl_types: {casted.lvl_types}")
        # CHECK: dim_to_lvl: (d0, d1) -> (d1, d0)
        print(f"dim_to_lvl: {casted.dim_to_lvl}")
        # CHECK: pos_width: 8
        print(f"pos_width: {casted.pos_width}")
        # CHECK: crd_width: 32
        print(f"crd_width: {casted.crd_width}")

        created = st.EncodingAttr.get(casted.lvl_types, casted.dim_to_lvl, 8, 32)
        # CHECK: #sparse_tensor.encoding<{ map = (d0, d1) -> (d1 : dense, d0 : compressed), posWidth = 8, crdWidth = 32 }>
        print(created)
        # CHECK: created_equal: True
        print(f"created_equal: {created == casted}")


# CHECK-LABEL: TEST: testEncodingAttrOnTensorType
@run
def testEncodingAttrOnTensorType():
    with Context() as ctx, Location.unknown():
        encoding = st.EncodingAttr(
            Attribute.parse(
                "#sparse_tensor.encoding<{"
                "  map = (d0) -> (d0 : compressed), "
                "  posWidth = 64,"
                "  crdWidth = 32"
                "}>"
            )
        )
        tt = RankedTensorType.get((1024,), F32Type.get(), encoding=encoding)
        # CHECK: tensor<1024xf32, #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed), posWidth = 64, crdWidth = 32 }>>
        print(tt)
        # CHECK: #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed), posWidth = 64, crdWidth = 32 }>
        print(tt.encoding)
        assert tt.encoding == encoding
