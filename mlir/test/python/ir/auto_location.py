# RUN: %PYTHON %s | FileCheck %s

import gc
from contextlib import contextmanager

from mlir.ir import *
from mlir.dialects._ods_common import _cext
from mlir.dialects import arith, _arith_ops_gen


def run(f):
    print("\nTEST:", f.__name__)
    f()
    gc.collect()
    assert Context._get_live_count() == 0


@contextmanager
def with_infer_location():
    _cext.globals.set_loc_tracebacks_enabled(True)
    yield
    _cext.globals.set_loc_tracebacks_enabled(False)


# CHECK-LABEL: TEST: testInferLocations
@run
def testInferLocations():
    with Context() as ctx, Location.unknown(), with_infer_location():
        ctx.allow_unregistered_dialects = True

        op = Operation.create("custom.op1")
        one = arith.constant(IndexType.get(), 1)
        _cext.globals.register_traceback_file_exclusion(arith.__file__)
        two = arith.constant(IndexType.get(), 2)

        # fmt: off
        # CHECK: loc(callsite("testInferLocations"("{{.*}}/test/python/ir/auto_location.py":31:13 to :43) at callsite("run"("{{.*}}/test/python/ir/auto_location.py":13:4 to :7) at "<module>"("{{.*}}/test/python/ir/auto_location.py":26:1 to :4))))
        # fmt: on
        print(op.location)

        # fmt: off
        # CHECK: loc(callsite("ConstantOp.__init__"("{{.*}}/mlir/dialects/arith.py":65:12 to :76) at callsite("constant"("{{.*}}/mlir/dialects/arith.py":110:40 to :81) at callsite("testInferLocations"("{{.*}}/test/python/ir/auto_location.py":32:14 to :48) at callsite("run"("{{.*}}/test/python/ir/auto_location.py":13:4 to :7) at "<module>"("{{.*}}/test/python/ir/auto_location.py":26:1 to :4))))))
        # fmt: on
        print(one.location)

        # fmt: off
        # CHECK: loc(callsite("testInferLocations"("{{.*}}/test/python/ir/auto_location.py":34:14 to :48) at callsite("run"("{{.*}}/test/python/ir/auto_location.py":13:4 to :7) at "<module>"("{{.*}}/test/python/ir/auto_location.py":26:1 to :4))))
        # fmt: on
        print(two.location)

        _cext.globals.register_traceback_file_inclusion(_arith_ops_gen.__file__)
        three = arith.constant(IndexType.get(), 3)
        # fmt: off
        # CHECK: loc(callsite("ConstantOp.__init__"("{{.*}}/mlir/dialects/_arith_ops_gen.py":405:4 to :218) at callsite("testInferLocations"("{{.*}}/test/python/ir/auto_location.py":52:16 to :50) at callsite("run"("{{.*}}/test/python/ir/auto_location.py":13:4 to :7) at "<module>"("{{.*}}/test/python/ir/auto_location.py":26:1 to :4)))))
        # fmt: on
        print(three.location)

        def foo():
            four = arith.constant(IndexType.get(), 4)
            print(four.location)

        # fmt: off
        # CHECK: loc(callsite("ConstantOp.__init__"("{{.*}}/mlir/dialects/_arith_ops_gen.py":405:4 to :218) at callsite("testInferLocations.<locals>.foo"("{{.*}}/test/python/ir/auto_location.py":59:19 to :53) at callsite("testInferLocations"("{{.*}}/test/python/ir/auto_location.py":65:8 to :13) at callsite("run"("{{.*}}/test/python/ir/auto_location.py":13:4 to :7) at "<module>"("{{.*}}/test/python/ir/auto_location.py":26:1 to :4))))))
        # fmt: on
        foo()

        _cext.globals.register_traceback_file_exclusion(__file__)

        # fmt: off
        # CHECK: loc("ConstantOp.__init__"("{{.*}}/mlir/dialects/_arith_ops_gen.py":405:4 to :218))
        # fmt: on
        foo()
