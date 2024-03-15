# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects.transform import interpreter as interp


def test_in_context(f):
    with ir.Context(), ir.Location.unknown():
        f()
    return f


print_root_module = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.print %root { name = \"from interpreter\" }: !transform.any_op
    transform.yield
  }
}"""


@test_in_context
def print_self():
    m = ir.Module.parse(print_root_module.replace("from interpreter", "print_self"))
    interp.apply_named_sequence(m, m.body.operations[0], m)


# CHECK-LABEL: print_self
# CHECK: transform.named_sequence @__transform_main
# CHECK: transform.print
# CHECK: transform.yield


@test_in_context
def print_other():
    transform = ir.Module.parse(
        print_root_module.replace("from interpreter", "print_other")
    )
    payload = ir.Module.parse("module attributes { this.is.payload } {}")
    interp.apply_named_sequence(payload, transform.body.operations[0], transform)


# CHECK-LABEL: print_other
# CHECK-NOT: transform
# CHECK: this.is.payload


@test_in_context
def failed():
    payload = ir.Module.parse("module attributes { this.is.payload } {}")
    try:
        interp.apply_named_sequence(payload, payload, payload)
    except ValueError as e:
        assert (
            "must implement TransformOpInterface to be used as transform root" in str(e)
        )
