from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, TypeVar

try:
    from .... import ir
    from ....dialects import transform
    from ....dialects.transform import structured
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e


@dataclass
class Value(abc.ABC):
    """Wrapper around a transform value handle with methods to chain further transforms."""

    _mlir_value: ir.Value
    children: list[Value] = field(default_factory=list)
    parent: Optional[Value] = None

    @property
    def mlir_value(self) -> ir.Value:
        return self._mlir_value


@dataclass
class Param(Value):
    """Wrapper around a transform Param with methods to chain further transforms."""


@dataclass
class OpHandle(Value):
    """Wrapper around a transform OpHandle with methods to chain further transforms."""

    def match_ops(
        self,
        ops: str
        | ir.OpView
        | structured.MatchInterfaceEnum
        | Sequence[str | ir.OpView],
    ) -> OpHandle:
        """Returns a handle to ops that match the given names, types, or interface.

        If only a single type is given, the value wrapped by the resulting
        handle is populated with the respective type.
        """
        # Handle interface.
        if isinstance(ops, structured.MatchInterfaceEnum) or (
            isinstance(ops, str) and ops in structured.MatchInterfaceEnum.__members__
        ):
            if isinstance(ops, str):
                ops = structured.MatchInterfaceEnum[ops]
            match_op = structured.MatchOp(
                transform.AnyOpType.get(),
                self.mlir_value,
                interface=ops,
            )

        # Handle op name(s), either given directly as string or given as op.
        else:
            if isinstance(ops, str):
                op_type = transform.OperationType.get(ops)
                op_names = [ops]
            elif isinstance(ops, Sequence):
                op_type = transform.AnyOpType.get()
                op_names = [
                    op if isinstance(op, str) else op.OPERATION_NAME for op in ops
                ]
            else:
                op_type = transform.OperationType.get(ops.OPERATION_NAME)
                op_names = [ops.OPERATION_NAME]
            match_op = structured.MatchOp.match_op_names(
                op_type,
                self.mlir_value,
                op_names,
            )

        handle = OpHandle(match_op.results_, parent=self)
        self.children.append(handle)
        return handle


ValueT = TypeVar("ValueT", bound=Value)


def insert_transform_script(
    module: ir.Module,
    script: Callable[[ValueT], None],
    dump_script: bool = False,
) -> None:
    """Inserts the transform script of the schedule into the module.

    Args:
        module: Existing module into which the script should be inserted.
        script: The transform script to apply at.
        dump_script: Whether to dump the script after creation.
    """
    # Insert the script into the IR
    with module.context, ir.Location.unknown(module.context):
        with ir.InsertionPoint.at_block_begin(module.body):
            sequence_op = transform.SequenceOp(
                transform.FailurePropagationMode.Propagate,
                (),
                transform.AnyOpType.get(),
            )
        with ir.InsertionPoint(sequence_op.body):
            script(OpHandle(sequence_op.bodyTarget))
            transform.YieldOp([])

    if dump_script:
        print(sequence_op)
