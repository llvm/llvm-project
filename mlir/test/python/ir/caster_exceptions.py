# RUN: %PYTHON %s | FileCheck %s

# Python exceptions raised by user-registered type/attribute casters while
# converting a C API object (MlirType, MlirAttribute, ...) returned from C++ to
# Python must propagate to the caller instead of aborting the process.

import gc
from mlir.ir import *
from mlir.dialects import llvm


def run(f):
    print("\nTEST:", f.__name__)
    f()
    gc.collect()
    assert Context._get_live_count() == 0
    return f


class CasterError(Exception):
    pass


def raising_caster(obj):
    raise CasterError(f"cannot cast {obj}")


# CHECK-LABEL: TEST: testTypeCasterRaisesInList
@run
def testTypeCasterRaisesInList():
    with Context(), Location.unknown():
        i32 = IntegerType.get_signless(32)
        ft = FunctionType.get(inputs=[i32], results=[])
        register_type_caster(IntegerType.static_typeid, replace=True)(raising_caster)
        try:
            # `inputs` appends MlirType values to an nb::list.
            ft.inputs
        except CasterError as e:
            # CHECK: CasterError: cannot cast i32
            print("CasterError:", e)
        finally:
            register_type_caster(IntegerType.static_typeid, replace=True)(
                lambda t: IntegerType(t)
            )
        # CHECK: [IntegerType(i32)]
        print(ft.inputs)


# CHECK-LABEL: TEST: testTypeCasterRaisesInReturn
@run
def testTypeCasterRaisesInReturn():
    with Context(), Location.unknown():
        arr = llvm.ArrayType.get(IntegerType.get_signless(32), 4)
        register_type_caster(IntegerType.static_typeid, replace=True)(raising_caster)
        try:
            # `element_type` returns an MlirType directly.
            arr.element_type
        except CasterError as e:
            # CHECK: CasterError: cannot cast i32
            print("CasterError:", e)
        finally:
            register_type_caster(IntegerType.static_typeid, replace=True)(
                lambda t: IntegerType(t)
            )
        # CHECK: IntegerType(i32)
        print(repr(arr.element_type))


# CHECK-LABEL: TEST: testAttributeCasterRaisesInReturn
@run
def testAttributeCasterRaisesInReturn():
    with Context(), Location.unknown():
        md_const = llvm.MDConstantAttr.get(
            IntegerAttr.get(IntegerType.get_signless(32), 42)
        )
        register_type_caster(IntegerAttr.static_typeid, replace=True)(raising_caster)
        try:
            # `value` returns an MlirAttribute directly.
            md_const.value
        except CasterError as e:
            # CHECK: CasterError: cannot cast 42 : i32
            print("CasterError:", e)
        finally:
            register_type_caster(IntegerAttr.static_typeid, replace=True)(
                lambda a: IntegerAttr(a)
            )
        # CHECK: IntegerAttr(42 : i32)
        print(repr(md_const.value))
