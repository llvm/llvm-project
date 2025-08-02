# RUN: %PYTHON %s | FileCheck %s
import gc
import traceback

from mlir import source_info_util
from mlir.source_info_util import _traceback_to_location
from mlir import traceback_util
from mlir.ir import Context

# CHECK: hello
print("hello")


# traceback_util.register_exclusion(__file__)


def run(f):
    print("\nTEST:", f.__name__)
    with Context() as ctx:
        f()
    gc.collect()
    # assert Context._get_live_count() == 0
    return f


@run
def foo():
    def bar():
        curr = source_info_util.current()
        print(curr.name_stack)
        print(curr.traceback)
        traceback.print_tb(
            traceback_util.filter_traceback(curr.traceback.as_python_traceback())
        )

        loc = _traceback_to_location(curr.traceback)
        print(loc)

    bar()
