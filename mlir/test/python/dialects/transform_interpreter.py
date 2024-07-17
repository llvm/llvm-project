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
def transform_options():
    options = interp.TransformOptions()
    options.expensive_checks = False
    options.enforce_single_top_level_transform_op = True
    m = ir.Module.parse(
        print_root_module.replace("from interpreter", "transform_options")
    )
    payload = ir.Module.parse("module attributes { this.is.payload } {}")
    interp.apply_named_sequence(payload, m.body.operations[0], m, options)


# CHECK-LABEL: transform_options


@test_in_context
def failed():
    payload = ir.Module.parse("module attributes { this.is.payload } {}")
    try:
        interp.apply_named_sequence(payload, payload, payload)
    except ValueError as e:
        assert (
            "must implement TransformOpInterface to be used as transform root" in str(e)
        )


print_root_via_include_module = """
module @print_root_via_include_module attributes {transform.with_named_sequence} {
  transform.named_sequence private @callee1(%root: !transform.any_op {transform.readonly})
  transform.named_sequence private @callee2(%root: !transform.any_op {transform.readonly})
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.include @callee2 failures(propagate)
        (%root) : (!transform.any_op) -> ()
    transform.yield
  }
}"""

callee2_definition = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence private @callee1(%root: !transform.any_op {transform.readonly})
  transform.named_sequence @callee2(%root: !transform.any_op {transform.readonly}) {
    transform.include @callee1 failures(propagate)
        (%root) : (!transform.any_op) -> ()
    transform.yield
  }
}
"""

callee1_definition = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @callee1(%root: !transform.any_op {transform.readonly}) {
    transform.print %root { name = \"from interpreter\" }: !transform.any_op
    transform.yield
  }
}
"""


@test_in_context
def include():
    main = ir.Module.parse(print_root_via_include_module)
    callee1 = ir.Module.parse(callee1_definition)
    callee2 = ir.Module.parse(callee2_definition)
    interp.copy_symbols_and_merge_into(main, callee1)
    interp.copy_symbols_and_merge_into(main, callee2)

    # CHECK: @print_root_via_include_module
    # CHECK: transform.named_sequence @__transform_main
    # CHECK: transform.include @callee2
    #
    # CHECK: transform.named_sequence @callee1
    # CHECK: transform.print
    #
    # CHECK: transform.named_sequence @callee2
    # CHECK: transform.include @callee1
    interp.apply_named_sequence(main, main.body.operations[0], main)


@test_in_context
def partial_include():
    main = ir.Module.parse(print_root_via_include_module)
    callee2 = ir.Module.parse(callee2_definition)
    interp.copy_symbols_and_merge_into(main, callee2)

    try:
        interp.apply_named_sequence(main, main.body.operations[0], main)
    except ValueError as e:
        assert "Failed to apply" in str(e)


@test_in_context
def repeated_include():
    main = ir.Module.parse(print_root_via_include_module)
    callee2 = ir.Module.parse(callee2_definition)
    interp.copy_symbols_and_merge_into(main, callee2)

    try:
        interp.copy_symbols_and_merge_into(main, callee2)
    except ValueError as e:
        assert "doubly defined symbol @callee2" in str(e)
