// RUN: mlir-opt -split-input-file %s | mlir-opt
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt

// -----

func.func @operations(%attribute: !pdl.attribute,
                 %input: !pdl.value,
                 %type: !pdl.type) {
  // attributes, operands, and results
  %op0 = pdl_interp.create_operation "foo.op"(%input : !pdl.value) {"attr" = %attribute} -> (%type : !pdl.type)

  // attributes, and results
  %op1 = pdl_interp.create_operation "foo.op" {"attr" = %attribute} -> (%type : !pdl.type)

  // attributes
  %op2 = pdl_interp.create_operation "foo.op" {"attr" = %attribute, "attr1" = %attribute}

  // operands, and results
  %op3 = pdl_interp.create_operation "foo.op"(%input : !pdl.value) -> (%type : !pdl.type)

  pdl_interp.finalize
}

// -----

func.func @extract(%attrs : !pdl.range<attribute>, %ops : !pdl.range<operation>, %types : !pdl.range<type>, %vals: !pdl.range<value>) {
  // attribute at index 0
  %attr = pdl_interp.extract 0 of %attrs : !pdl.attribute

  // operation at index 1
  %op = pdl_interp.extract 1 of %ops : !pdl.operation

  // type at index 2
  %type = pdl_interp.extract 2 of %types : !pdl.type

  // value at index 3
  %val = pdl_interp.extract 3 of %vals : !pdl.value

  pdl_interp.finalize
}

// -----

func.func @foreach(%ops: !pdl.range<operation>) {
  // iterate over a range of operations
  pdl_interp.foreach %op : !pdl.operation in %ops {
    %val = pdl_interp.get_result 0 of %op
    pdl_interp.continue
  } -> ^end

  ^end:
    pdl_interp.finalize
}

// -----

func.func @users(%value: !pdl.value, %values: !pdl.range<value>) {
  // all the users of a single value
  %ops1 = pdl_interp.get_users of %value : !pdl.value

  // all the users of all the values in a range
  %ops2 = pdl_interp.get_users of %values : !pdl.range<value>

  pdl_interp.finalize
}
