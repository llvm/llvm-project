// RUN: mlir-opt -split-input-file -convert-pdl-to-pdl-interp %s | FileCheck %s

// Ensuse that the dependency between add & less
// causes them to be in the correct order.
// CHECK-LABEL: matcher
// CHECK: apply_constraint "return_attr_constraint"
// CHECK: apply_constraint "use_attr_constraint"

module {
  pdl.pattern : benefit(1) {
    %0 = attribute
    %1 = types
    %2 = operation "tosa.mul" {"shift" = %0} -> (%1 : !pdl.range<type>)
    %3 = attribute = 0 : i32
    %4 = attribute = 1 : i32
    %5 = apply_native_constraint "return_attr_constraint"(%3, %4 : !pdl.attribute, !pdl.attribute) : !pdl.attribute
    apply_native_constraint "use_attr_constraint"(%0, %5 : !pdl.attribute, !pdl.attribute)
    rewrite %2 with "rewriter"
  }
}

// -----

// CHECK-LABEL: matcher
// CHECK: %[[ATTR:.*]] = pdl_interp.get_attribute "attr" of
// CHECK: %[[CONSTRAINT:.*]] = pdl_interp.apply_constraint "return_attr_constraint"
// CHECK: pdl_interp.are_equal %[[ATTR:.*]], %[[CONSTRAINT:.*]]

pdl.pattern : benefit(1) {
    %inputOp = operation
    %result = result 0 of %inputOp
    %attr = pdl.apply_native_constraint "return_attr_constraint"(%inputOp : !pdl.operation) : !pdl.attribute
    %root = operation(%result : !pdl.value) {"attr" = %attr}
    rewrite %root with "rewriter"(%attr : !pdl.attribute)
}

// -----

// CHECK-LABEL: matcher
// CHECK: %[[CONSTRAINT:.*]] = pdl_interp.apply_constraint "return_value_constr"
// CHECK: %[[VALUE:.*]] = pdl_interp.get_operand 0
// CHECK: pdl_interp.are_equal %[[VALUE:.*]], %[[CONSTRAINT:.*]]

pdl.pattern : benefit(1) {
    %attr = attribute = 10
    %value = pdl.apply_native_constraint "return_value_constr"(%attr: !pdl.attribute) : !pdl.value
    %root = operation(%value : !pdl.value)
    rewrite %root with "rewriter"
}

// -----

// CHECK-LABEL: matcher
// CHECK: %[[CONSTRAINT:.*]] = pdl_interp.apply_constraint "return_type_constr"
// CHECK: %[[TYPE:.*]] = pdl_interp.get_value_type of
// CHECK: pdl_interp.are_equal %[[TYPE:.*]], %[[CONSTRAINT:.*]]

pdl.pattern : benefit(1) {
    %attr = attribute = 10
    %type = pdl.apply_native_constraint "return_type_constr"(%attr: !pdl.attribute) : !pdl.type
    %root = operation -> (%type : !pdl.type)
    rewrite %root with "rewriter"
}

// -----

// CHECK-LABEL: matcher
// CHECK: %[[CONSTRAINT:.*]] = pdl_interp.apply_constraint "return_type_range_constr"
// CHECK: %[[TYPE:.*]] = pdl_interp.get_value_type of
// CHECK: pdl_interp.are_equal %[[TYPE:.*]], %[[CONSTRAINT:.*]]

pdl.pattern : benefit(1) {
    %attr = attribute = 10
    %types = pdl.apply_native_constraint "return_type_range_constr"(%attr: !pdl.attribute) : !pdl.range<type>
    %root = operation -> (%types : !pdl.range<type>)
    rewrite %root with "rewriter"
}
