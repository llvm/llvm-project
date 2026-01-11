# RUN: %PYTHON -m ty check --output-format concise %s | FileCheck %s

from mlir.ir import (
    Module,
    Context,
    Region,
    Block,
    OpView,
    Attribute,
    OpOperandList,
    RegionSequence,
    OpResultList,
    OpResult,
    BlockList,
    BlockArgumentList,
    BlockPredecessors,
    BlockSuccessors,
    OperationList,
    OpAttributeMap,
    NamedAttribute,
)
from typing import reveal_type

# This file is not a valid python program. It is only used to test type inference.

module = Module.create()
# CHECK: Revealed type: `Module`
reveal_type(module)

if True:  # Tests for Module
    # CHECK: Revealed type: `Block`
    reveal_type(module.body)

    # CHECK: Revealed type: `Operation`
    reveal_type(module.operation)

if True:  # Tests for Block
    block: Block = module.body

    for block_iter in block:
        # CHECK: Revealed type: `OpView`
        reveal_type(block_iter)

    # CHECK: Revealed type: `BlockArgumentList`
    reveal_type(block.arguments)

    if True:  # Tests for BlockArgumentList
        block_arguments_list: BlockArgumentList = block.arguments

        # CHECK: Revealed type: `BlockArgument`
        reveal_type(block.arguments[0])
        for block_arguments_iter in block.arguments:
            # CHECK: Revealed type: `BlockArgument`
            reveal_type(block_arguments_iter)

    # CHECK: Revealed type: `OperationList`
    reveal_type(block.operations)
    if True:  # Tests for OperationList
        operation_list: OperationList = block.operations

        # CHECK: Revealed type: `OpView`
        reveal_type(operation_list[0])
        for operation_list_iter in operation_list:
            # CHECK: Revealed type: `OpView`
            reveal_type(operation_list_iter)

    # CHECK: Revealed type: `BlockPredecessors`
    reveal_type(block.predecessors)
    if True:  # Tests for BlockPredecessors
        block_predecessors: BlockPredecessors = block.predecessors

        # CHECK: Revealed type: `Block`
        reveal_type(block_predecessors[0])
        for block_predecessors_iter in block_predecessors:
            # CHECK: Revealed type: `Block`
            reveal_type(block_predecessors_iter)

    # CHECK: Revealed type: `BlockSuccessors`
    reveal_type(block.successors)
    if True:  # Tests for BlockSuccessors
        block_successors: BlockSuccessors = block.successors

        # CHECK: Revealed type: `Block`
        reveal_type(block_successors[0])
        for block_successors_iter in block_successors:
            # CHECK: Revealed type: `Block`
            reveal_type(block_successors_iter)

if True:  # Tests for OpView
    opview: OpView = module.body.operations[0]

    # CHECK: Revealed type: `OpAttributeMap`
    reveal_type(opview.attributes)
    if True:  # Tests for OpAttributeMap
        attribue_map: OpAttributeMap = opview.attributes

        # CHECK: Revealed type: `NamedAttribute`
        reveal_type(attribue_map[0])

        # CHECK: Revealed type: `Attribute`
        reveal_type(attribue_map["str"])

        # This type hint is a lie, because `get` will also return any other default argument
        # CHECK: Revealed type: `Attribute | None`
        reveal_type(attribue_map.get("str"))

        # CHECK: Revealed type: `list[tuple[str, Attribute]]`
        reveal_type(attribue_map.items())

        # CHECK: Revealed type: `list[str]`
        reveal_type(attribue_map.keys())

        # CHECK: Revealed type: `list[Attribute]`
        reveal_type(attribue_map.values())

    # CHECK: Revealed type: `OpOperandList`
    reveal_type(opview.operands)
    if True:  # Tests for OpOperandList
        op_operands_list: OpOperandList = opview.operands

        # CHECK: Revealed type: `Value`
        reveal_type(op_operands_list[0])
        for op_operands_list_iter in op_operands_list:
            # CHECK: Revealed type: `Value`
            reveal_type(op_operands_list_iter)

    # CHECK: Revealed type: `RegionSequence`
    reveal_type(opview.regions)
    if True:  # Tests for RegionSequence
        region_sequence: RegionSequence = opview.regions

        # CHECK: Revealed type: `Region`
        reveal_type(region_sequence[0])
        for regions_sequence_iter in region_sequence:
            # CHECK: Revealed type: `Region`
            reveal_type(regions_sequence_iter)

    # CHECK: Revealed type: `OpResultList`
    reveal_type(opview.results)
    if True:  # Tests for OpResultList
        result_list: OpResultList = opview.results

        # CHECK: Revealed type: `OpResult`
        reveal_type(result_list[0])
        for result_list_iter in result_list:
            # CHECK: Revealed type: `OpResult`
            reveal_type(result_list_iter)

    # CHECK: Revealed type: `OpResult`
    reveal_type(opview.result)

if True:  # Tests for Region
    region: Region = module.body.operations[0].regions[0]

    for region_iter in region:
        # CHECK: Revealed type: `Block`
        reveal_type(region_iter)

    # CHECK: Revealed type: `BlockList`
    reveal_type(region.blocks)
    if True:  # Tests for BlockList
        blocklist: BlockList = region.blocks

        # CHECK: Revealed type: `Block`
        reveal_type(blocklist[0])
        for blocklist_iter in blocklist:
            # CHECK: Revealed type: `Block`
            reveal_type(blocklist_iter)
