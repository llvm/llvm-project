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

        # CHECK: lvl_types: [262144]
        print(f"lvl_types: {casted.lvl_types}")
        # CHECK: dim_to_lvl: (d0) -> (d0)
        print(f"dim_to_lvl: {casted.dim_to_lvl}")
        # CHECK: lvl_to_dim: (d0) -> (d0)
        print(f"lvl_to_dim: {casted.lvl_to_dim}")
        # CHECK: pos_width: 16
        print(f"pos_width: {casted.pos_width}")
        # CHECK: crd_width: 32
        print(f"crd_width: {casted.crd_width}")

        created = st.EncodingAttr.get(casted.lvl_types, None, None, 0, 0)
        # CHECK: #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
        print(created)
        # CHECK: created_equal: False
        print(f"created_equal: {created == casted}")

        # Verify that the factory creates an instance of the proper type.
        # CHECK: is_proper_instance: True
        print(f"is_proper_instance: {isinstance(created, st.EncodingAttr)}")
        # CHECK: created_pos_width: 0
        print(f"created_pos_width: {created.pos_width}")


# CHECK-LABEL: TEST: testEncodingAttrStructure
@run
def testEncodingAttrStructure():
    with Context() as ctx:
        parsed = Attribute.parse(
            "#sparse_tensor.encoding<{"
            "  map = (d0, d1) -> (d0 : dense, d1 floordiv 4 : dense,"
            "  d1 mod 4 : structured[2, 4]),"
            "  posWidth = 16,"
            "  crdWidth = 32"
            "}>"
        )
        # CHECK: #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 floordiv 4 : dense, d1 mod 4 : structured[2, 4]), posWidth = 16, crdWidth = 32 }>
        print(parsed)

        casted = st.EncodingAttr(parsed)
        # CHECK: equal: True
        print(f"equal: {casted == parsed}")

        # CHECK: lvl_types: [65536, 65536, 4406638542848]
        print(f"lvl_types: {casted.lvl_types}")
        # CHECK: lvl_formats_enum: [<LevelFormat.dense: 65536>, <LevelFormat.dense: 65536>, <LevelFormat.n_out_of_m: 2097152>]
        print(f"lvl_formats_enum: {casted.lvl_formats_enum}")
        # CHECK: structured_n: 2
        print(f"structured_n: {casted.structured_n}")
        # CHECK: structured_m: 4
        print(f"structured_m: {casted.structured_m}")
        # CHECK: dim_to_lvl: (d0, d1) -> (d0, d1 floordiv 4, d1 mod 4)
        print(f"dim_to_lvl: {casted.dim_to_lvl}")
        # CHECK: lvl_to_dim: (d0, d1, d2) -> (d0, d1 * 4 + d2)
        print(f"lvl_to_dim: {casted.lvl_to_dim}")
        # CHECK: pos_width: 16
        print(f"pos_width: {casted.pos_width}")
        # CHECK: crd_width: 32
        print(f"crd_width: {casted.crd_width}")

        created = st.EncodingAttr.get(
            casted.lvl_types, casted.dim_to_lvl, casted.lvl_to_dim, 0, 0
        )
        # CHECK: #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 floordiv 4 : dense, d1 mod 4 : structured[2, 4]) }>
        print(created)
        # CHECK: created_equal: False
        print(f"created_equal: {created == casted}")

        built_2_4 = st.EncodingAttr.build_level_type(
            st.LevelFormat.n_out_of_m, [], 2, 4
        )
        built_dense = st.EncodingAttr.build_level_type(st.LevelFormat.dense)
        dim_to_lvl = AffineMap.get(
            2,
            0,
            [
                AffineExpr.get_dim(0),
                AffineExpr.get_floor_div(AffineExpr.get_dim(1), 4),
                AffineExpr.get_mod(AffineExpr.get_dim(1), 4),
            ],
        )
        lvl_to_dim = AffineMap.get(
            3,
            0,
            [
                AffineExpr.get_dim(0),
                AffineExpr.get_add(
                    AffineExpr.get_mul(AffineExpr.get_dim(1), 4),
                    AffineExpr.get_dim(2),
                ),
            ],
        )
        built = st.EncodingAttr.get(
            [built_dense, built_dense, built_2_4],
            dim_to_lvl,
            lvl_to_dim,
            0,
            0,
        )
        # CHECK: #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 floordiv 4 : dense, d1 mod 4 : structured[2, 4]) }>
        print(built)
        # CHECK: built_equal: True
        print(f"built_equal: {built == created}")

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

        # CHECK: lvl_types: [65536, 262144]
        print(f"lvl_types: {casted.lvl_types}")
        # CHECK: dim_to_lvl: (d0, d1) -> (d1, d0)
        print(f"dim_to_lvl: {casted.dim_to_lvl}")
        # CHECK: lvl_to_dim: (d0, d1) -> (d1, d0)
        print(f"lvl_to_dim: {casted.lvl_to_dim}")
        # CHECK: pos_width: 8
        print(f"pos_width: {casted.pos_width}")
        # CHECK: crd_width: 32
        print(f"crd_width: {casted.crd_width}")

        created = st.EncodingAttr.get(
            casted.lvl_types,
            casted.dim_to_lvl,
            casted.lvl_to_dim,
            8,
            32,
        )
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
