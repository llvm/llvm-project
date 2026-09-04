// RUN: mlir-opt -split-input-file %s | mlir-opt
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt

// -----

// Unused operation to force loading the `arithmetic` dialect for the
// test of type inferrence.
arith.constant true

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

  // inferred results
  %op4 = pdl_interp.create_operation "arith.constant" -> <inferred>

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

// -----

// Region/Block navigation operations
func.func @region_block_navigation(%op : !pdl.operation) {
  // get region at index 0
  %region = pdl_interp.get_region %op at 0

  // get block at index 0
  %block = pdl_interp.get_block %region at 0

  // get block argument at index 0
  %arg = pdl_interp.get_block_argument %block at 0

  // get block arguments starting from index 1 as a range
  %args = pdl_interp.get_block_arguments %block at 1 : !pdl.range<value>

  pdl_interp.finalize
}

// -----

// Region/Block predicate operations
module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.check_region_count of %root is 1 -> ^bb1, ^end

  ^bb1:
    %region = pdl_interp.get_region %root at 0
    pdl_interp.check_block_count of %region is 1 -> ^bb2, ^end

  ^bb2:
    %block = pdl_interp.get_block %region at 0
    pdl_interp.check_block_arg_count of %block is 2 -> ^bb3, ^end

  ^bb3:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// -----

// GetBlockOps + ForEach pattern for iterating over operations in a block
module @patterns_foreach {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    %region = pdl_interp.get_region %root at 0
    %block = pdl_interp.get_block %region at 0
    %ops = pdl_interp.get_block_ops of %block
    pdl_interp.foreach %op : !pdl.operation in %ops {
      pdl_interp.continue
    } -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      pdl_interp.finalize
    }
  }
}

// -----

// TakeRegion and MoveBlock operations
module @patterns_move {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.check_region_count of %root is 2 -> ^bb1, ^end

  ^bb1:
    pdl_interp.record_match @rewriters::@move_region(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @move_region(%root : !pdl.operation) {
      %region0 = pdl_interp.get_region %root at 0
      %region1 = pdl_interp.get_region %root at 1

      // Transfer all blocks from region0 into region1.
      pdl_interp.take_region %region0 before %region1

      // Get blocks from region1 and reorder.
      %block0 = pdl_interp.get_block %region1 at 0
      %block1 = pdl_interp.get_block %region1 at 1

      pdl_interp.move_block %block1 before %block0

      pdl_interp.finalize
    }
  }
}

