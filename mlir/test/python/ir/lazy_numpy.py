# RUN: %PYTHON %s | FileCheck %s
# RUN: %PYTHON %s --without-numpy | FileCheck %s --check-prefix=NO-NUMPY
# RUN: %PYTHON %s --missing-dependency | FileCheck %s --check-prefix=DEPENDENCY
# RUN: %PYTHON %s --broken-numpy | FileCheck %s --check-prefix=BROKEN

# Importing mlir.ir must not import NumPy. The dense elements attribute builders
# import it when first called, and report a missing NumPy at that point.

import sys

import_error = None
if "--without-numpy" in sys.argv:
    sys.modules["numpy"] = None
elif "--missing-dependency" in sys.argv:
    import_error = ModuleNotFoundError(
        "No module named 'numpy_dependency'", name="numpy_dependency"
    )
elif "--broken-numpy" in sys.argv:
    import_error = ImportError("NumPy initialization failed")

if import_error is not None:

    class BrokenNumpyFinder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "numpy":
                raise import_error

    sys.meta_path.insert(0, BrokenNumpyFinder())

from mlir.ir import *

# CHECK: numpy imported: False
# NO-NUMPY: numpy imported: False
# DEPENDENCY: numpy imported: False
# BROKEN: numpy imported: False
print("numpy imported:", sys.modules.get("numpy") is not None)

# CHECK: F64ElementsAttr: dense<[1.500000e+00, 2.500000e+00]> : tensor<2xf64>
# CHECK: I32ElementsAttr: dense<[1, 2, 3]> : tensor<3xi32>
# CHECK: I64ElementsAttr: dense<[4, 5]> : tensor<2xi64>
# CHECK: IndexElementsAttr: dense<6> : tensor<1xindex>
# NO-NUMPY: F64ElementsAttr: ModuleNotFoundError: {{.*}}requires NumPy
# NO-NUMPY: I32ElementsAttr: ModuleNotFoundError: {{.*}}requires NumPy
# NO-NUMPY: I64ElementsAttr: ModuleNotFoundError: {{.*}}requires NumPy
# NO-NUMPY: IndexElementsAttr: ModuleNotFoundError: {{.*}}requires NumPy
# DEPENDENCY: F64ElementsAttr: ModuleNotFoundError: No module named 'numpy_dependency'
# DEPENDENCY: I32ElementsAttr: ModuleNotFoundError: No module named 'numpy_dependency'
# DEPENDENCY: I64ElementsAttr: ModuleNotFoundError: No module named 'numpy_dependency'
# DEPENDENCY: IndexElementsAttr: ModuleNotFoundError: No module named 'numpy_dependency'
# BROKEN: F64ElementsAttr: ImportError: NumPy initialization failed
# BROKEN: I32ElementsAttr: ImportError: NumPy initialization failed
# BROKEN: I64ElementsAttr: ImportError: NumPy initialization failed
# BROKEN: IndexElementsAttr: ImportError: NumPy initialization failed
with Context() as ctx:
    for kind, values in (
        ("F64ElementsAttr", [1.5, 2.5]),
        ("I32ElementsAttr", [1, 2, 3]),
        ("I64ElementsAttr", [4, 5]),
        ("IndexElementsAttr", [6]),
    ):
        try:
            print(f"{kind}:", AttrBuilder.get(kind)(values, context=ctx))
        except ImportError as e:
            if import_error is not None:
                assert e is import_error
            else:
                assert "--without-numpy" in sys.argv
                assert isinstance(e, ModuleNotFoundError) and e.name == "numpy"
            print(f"{kind}: {type(e).__name__}: {e}")

# CHECK: numpy imported: True
# NO-NUMPY: numpy imported: False
# DEPENDENCY: numpy imported: False
# BROKEN: numpy imported: False
print("numpy imported:", sys.modules.get("numpy") is not None)
