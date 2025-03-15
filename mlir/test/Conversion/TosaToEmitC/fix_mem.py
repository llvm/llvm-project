import argparse
import dataclasses
import re
import pathlib
import typing


@dataclasses.dataclass(frozen=True)
class Pattern:
    match: str
    substitution: str | typing.Callable[[str], str]
    name: str

    def substitute(self, input: str) -> str:
        return re.sub(self.match, self.substitution, input, 0, re.MULTILINE)


SUBSTITUTIONS = [
    # Insert additional constant with 0
    Pattern(
        r"func.func(.*)\{",
        "func.func\\1{\\n    %zzz = arith.constant 0 : index",
        "const_0",
    ),
    # Convert all 0D memrefs 1D memrefs
    Pattern(r"memref<(\D\S*)>", "memref<1x\\1>", "memref-1d"),
    # memref.load
    Pattern(
        r"memref.load %(.*)\[()\] : memref<(.*)>",
        "memref.load %\\1[%zzz] : memref<\\3>",
        "memref-load",
    ),
    # memref.store
    Pattern(
        r"memref.store %(.*), %(.*)\[()\] : memref<(.*)>",
        "memref.store %\\1, %\\2[%zzz] : memref<\\4>",
        "memref-store",
    ),
    # memref.alloca alignment
    Pattern(
        r"%(.*) \= memref.alloca\(\) \{alignment \= .* : .*\} : memref\<(.*)\>",
        "%\\1 = memref.alloca() : memref<\\2>",
        "memref-alloca",
    ),
    # memref.copy
    Pattern(
        r"memref.copy %(.*), %(.*) : memref<(.*)> to memref<(.*)>",
        "linalg.copy ins(%\\1 : memref<\\3>) outs(%\\2 : memref<\\4>)",
        "memref-copy",
    ),
    # arith.extf
    Pattern(
        r"%(.*) = arith.extf %(.*) : (.*) to (.*)",
        "%\\1 = emitc.cast %\\2 : \\3 to \\4",
        "arith-extf",
    ),
    # arith.truncf
    Pattern(
        r"%(.*) = arith.truncf %(.*) : (.*) to (.*)",
        "%\\1 = emitc.cast %\\2 : \\3 to \\4",
        "arith-truncf",
    ),
    # arith.index_cast
    Pattern(
        r"%(.*) = arith.index_cast %(.*) : index to (.*)",
        "%\\1 = emitc.cast %\\2 : index to \\3",
        "arith-index-cast",
    ),
    # arith.cmpf ogt
    Pattern(
        r"%(.*) = arith.cmpf ogt, %(.*), %(.*) : (.*)",
        "%\\1 = emitc.cmp gt , %\\2, %\\3 : (\\4, \\4) -> i1",
        "arith-cmpf",
    ),
    # arith.cmpf ugt
    Pattern(
        r"%(.*) = arith.cmpf ugt, %(.*), %(.*) : (.*)",
        "%\\1 = emitc.cmp gt , %\\2, %\\3 : (\\4, \\4) -> i1",
        "arith-cmpf",
    ),
    # arith.cmpf ult
    Pattern(
        r"%(.*) = arith.cmpf ult, %(.*), %(.*) : (.*)",
        "%\\1 = emitc.cmp lt , %\\2, %\\3 : (\\4, \\4) -> i1",
        "arith-cmpf",
    ),
    # arith.cmpf uno
    Pattern(
        r"%(.*) = arith.cmpf uno, %(.*), %(.*) : (.*)",
        "%\\1 = emitc.cmp ne , %\\2, %\\3 : (\\4, \\4) -> i1",
        "arith-cmpf",
    ),
    # args
    Pattern(
        r"func.func @forward\(%(.*): memref<(.*)>, %(.*): memref<(.*)>\) \{",
        'func.func @forward(%xxx: !emitc.array<\\2>, %yyy: !emitc.array<\\4>) {\\n    %\\1 = "builtin.unrealized_conversion_cast"(%xxx) : (!emitc.array<\\2>) -> memref<\\2>\\n    %\\3 = "builtin.unrealized_conversion_cast"(%yyy) : (!emitc.array<\\4>) -> memref<\\4>',
        "func-args",
    ),
    # TODO: replace tensor.concat with insert_slice
    # TODO: add output_shape to tensor.expand_shape
    # TODO: arith: truncf, extf, cmpf
    # # tensor.concat
    # Pattern(
    #     r"%(.*) = tensor.concat.*\n",
    #     lambda m: m,
    #     "tensor-concat",
    # ),
]


def substitute(input: str, names: list[str] | None) -> str:
    for pattern in SUBSTITUTIONS:
        if names is None or pattern.name in names:
            input = pattern.substitute(input)
    return input


def run(input_path: str, output_path: str, patterns: list[str] | None):
    input = pathlib.Path(input_path)
    if not input.exists():
        raise ValueError(f"File not found: {input}")

    input_str = input.read_text()
    output_str = substitute(input_str, patterns)

    output = pathlib.Path(output_path)
    output.write_text(output_str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path", metavar="input-path", help="Path to input mlir file"
    )
    parser.add_argument(
        "output_path", metavar="output-path", help="Path to output mlir file"
    )
    parser.add_argument("-p", "--pattern", action="append")
    args = parser.parse_args()

    run(args.input_path, args.output_path, args.pattern)


if __name__ == "__main__":
    main()
