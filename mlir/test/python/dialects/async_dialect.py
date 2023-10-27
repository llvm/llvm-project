# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
import mlir.dialects.async_dialect
import mlir.dialects.async_dialect.passes
from mlir.passmanager import *


def run(f):
    print("\nTEST:", f.__name__)
    f()


def testAsyncPass():
    with Context() as context:
        PassManager.parse("any(async-to-async-runtime)")
    print("SUCCESS")


# CHECK-LABEL: testAsyncPass
#       CHECK: SUCCESS
run(testAsyncPass)
