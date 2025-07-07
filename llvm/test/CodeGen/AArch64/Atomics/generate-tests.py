#!/usr/bin/env python3
import textwrap
import enum
import os
import re

"""
Generate the tests in llvm/test/CodeGen/AArch64/Atomics. Run from top level llvm-project.
"""

TRIPLES = [
    "aarch64",
    "aarch64_be",
]


class ByteSizes:
    def __init__(self, pairs):
        if not isinstance(pairs, list):
            raise ValueError("Must init with a list of key-value pairs")

        self._data = pairs[:]

    def __iter__(self):
        return iter(self._data)


# fmt: off
Type = ByteSizes([
   ("i8",   1),
   ("i16",  2),
   ("i32",  4),
   ("i64",  8),
   ("i128", 16)])

FPType = ByteSizes([
   ("half",   2),
   ("bfloat", 2),
   ("float",  4),
   ("double", 8)])
# fmt: on


# Is this an aligned or unaligned access?
class Aligned(enum.Enum):
    aligned = True
    unaligned = False

    def __str__(self) -> str:
        return self.name

    def __bool__(self) -> bool:
        return self.value


class AtomicOrder(enum.Enum):
    notatomic = 0
    unordered = 1
    monotonic = 2
    acquire = 3
    release = 4
    acq_rel = 5
    seq_cst = 6

    def __str__(self) -> str:
        return self.name


ATOMICRMW_ORDERS = [
    AtomicOrder.monotonic,
    AtomicOrder.acquire,
    AtomicOrder.release,
    AtomicOrder.acq_rel,
    AtomicOrder.seq_cst,
]

ATOMIC_LOAD_ORDERS = [
    AtomicOrder.unordered,
    AtomicOrder.monotonic,
    AtomicOrder.acquire,
    AtomicOrder.seq_cst,
]

ATOMIC_STORE_ORDERS = [
    AtomicOrder.unordered,
    AtomicOrder.monotonic,
    AtomicOrder.release,
    AtomicOrder.seq_cst,
]

ATOMIC_FENCE_ORDERS = [
    AtomicOrder.acquire,
    AtomicOrder.release,
    AtomicOrder.acq_rel,
    AtomicOrder.seq_cst,
]

CMPXCHG_SUCCESS_ORDERS = [
    AtomicOrder.monotonic,
    AtomicOrder.acquire,
    AtomicOrder.release,
    AtomicOrder.acq_rel,
    AtomicOrder.seq_cst,
]

CMPXCHG_FAILURE_ORDERS = [
    AtomicOrder.monotonic,
    AtomicOrder.acquire,
    AtomicOrder.seq_cst,
]

FENCE_ORDERS = [
    AtomicOrder.acquire,
    AtomicOrder.release,
    AtomicOrder.acq_rel,
    AtomicOrder.seq_cst,
]


class Feature(enum.Flag):
    # Feature names in filenames are determined by the spelling here:
    v8a = enum.auto()
    v8_1a = enum.auto()  # -mattr=+v8.1a, mandatory FEAT_LOR, FEAT_LSE
    rcpc = enum.auto()  # FEAT_LRCPC
    lse2 = enum.auto()  # FEAT_LSE2
    outline_atomics = enum.auto()  # -moutline-atomics
    rcpc3 = enum.auto()  # FEAT_LSE2 + FEAT_LRCPC3
    lse2_lse128 = enum.auto()  # FEAT_LSE2 + FEAT_LSE128

    def test_scope():
        return "all"

    @property
    def mattr(self):
        if self == Feature.outline_atomics:
            return "+outline-atomics"
        if self == Feature.v8_1a:
            return "+v8.1a"
        if self == Feature.rcpc3:
            return "+lse2,+rcpc3"
        if self == Feature.lse2_lse128:
            return "+lse2,+lse128"
        return "+" + self.name


class FPFeature(enum.Flag):
    # Feature names in filenames are determined by the spelling here:
    v8a_fp = enum.auto()
    lsfe = enum.auto()  # FEAT_LSFE

    def test_scope():
        return "atomicrmw"

    @property
    def mattr(self):
        if self == FPFeature.v8a_fp:
            return "+v8a"
        return "+" + self.name


ATOMICRMW_OPS = [
    "xchg",
    "add",
    "sub",
    "and",
    "nand",
    "or",
    "xor",
    "max",
    "min",
    "umax",
    "umin",
]

FP_ATOMICRMW_OPS = [
    "fadd",
    "fsub",
    "fmax",
    "fmin",
    "fmaximum",
    "fminimum",
]


def relpath():
    # __file__ changed to return absolute path in Python 3.9. Print only
    # up to llvm-project (6 levels higher), to avoid unnecessary diffs and
    # revealing directory structure of people running this script
    top = "../" * 6
    fp = os.path.relpath(__file__, os.path.abspath(os.path.join(__file__, top)))
    return fp


def generate_unused_res_test(featname, ordering, op, alignval):
    if featname != "lsfe" or op == "fsub" or alignval == 1:
        return False
    if ordering not in [AtomicOrder.monotonic, AtomicOrder.release]:
        return False
    return True


def align(val, aligned: bool) -> int:
    return val if aligned else 1


def all_atomicrmw(f, datatype, atomicrmw_ops, featname):
    instr = "atomicrmw"
    generate_unused = False
    tests = []
    for op in atomicrmw_ops:
        for aligned in Aligned:
            for ty, val in datatype:
                alignval = align(val, aligned)
                for ordering in ATOMICRMW_ORDERS:
                    name = f"atomicrmw_{op}_{ty}_{aligned}_{ordering}"
                    tests.append(
                        textwrap.dedent(
                            f"""
                        define dso_local {ty} @{name}(ptr %ptr, {ty} %value) {{
                            %r = {instr} {op} ptr %ptr, {ty} %value {ordering}, align {alignval}
                            ret {ty} %r
                        }}
                    """
                        )
                    )
                    if generate_unused_res_test(featname, ordering, op, alignval):
                        generate_unused = True
                        name = f"atomicrmw_{op}_{ty}_{aligned}_{ordering}_unused"
                        tests.append(
                            textwrap.dedent(
                                f"""
                           define dso_local void @{name}(ptr %ptr, {ty} %value) {{
                               %r = {instr} {op} ptr %ptr, {ty} %value {ordering}, align {alignval}
                               ret void
                           }}
                        """
                            )
                        )

    if generate_unused:
        f.write(
            "\n; NOTE: '_unused' tests are added to ensure we do not lower to "
            "ST[F]ADD when the destination register is WZR/XZR.\n"
            "; See discussion on https://github.com/llvm/llvm-project/pull/131174\n"
        )

    for test in tests:
        f.write(test)


def all_load(f):
    for aligned in Aligned:
        for ty, val in Type:
            alignval = align(val, aligned)
            for ordering in ATOMIC_LOAD_ORDERS:
                for const in [False, True]:
                    name = f"load_atomic_{ty}_{aligned}_{ordering}"
                    instr = "load atomic"
                    if const:
                        name += "_const"
                    arg = "ptr readonly %ptr" if const else "ptr %ptr"
                    f.write(
                        textwrap.dedent(
                            f"""
                        define dso_local {ty} @{name}({arg}) {{
                            %r = {instr} {ty}, ptr %ptr {ordering}, align {alignval}
                            ret {ty} %r
                        }}
                    """
                        )
                    )


def all_store(f):
    for aligned in Aligned:
        for ty, val in Type:
            alignval = align(val, aligned)
            for ordering in ATOMIC_STORE_ORDERS:  # FIXME stores
                name = f"store_atomic_{ty}_{aligned}_{ordering}"
                instr = "store atomic"
                f.write(
                    textwrap.dedent(
                        f"""
                    define dso_local void @{name}({ty} %value, ptr %ptr) {{
                        {instr} {ty} %value, ptr %ptr {ordering}, align {alignval}
                        ret void
                    }}
                """
                    )
                )


def all_cmpxchg(f):
    for aligned in Aligned:
        for ty, val in Type:
            alignval = align(val, aligned)
            for success_ordering in CMPXCHG_SUCCESS_ORDERS:
                for failure_ordering in CMPXCHG_FAILURE_ORDERS:
                    for weak in [False, True]:
                        name = f"cmpxchg_{ty}_{aligned}_{success_ordering}_{failure_ordering}"
                        instr = "cmpxchg"
                        if weak:
                            name += "_weak"
                            instr += " weak"
                        f.write(
                            textwrap.dedent(
                                f"""
                            define dso_local {ty} @{name}({ty} %expected, {ty} %new, ptr %ptr) {{
                                %pair = {instr} ptr %ptr, {ty} %expected, {ty} %new {success_ordering} {failure_ordering}, align {alignval}
                                %r = extractvalue {{ {ty}, i1 }} %pair, 0
                                ret {ty} %r
                            }}
                        """
                            )
                        )


def all_fence(f):
    for ordering in FENCE_ORDERS:
        name = f"fence_{ordering}"
        f.write(
            textwrap.dedent(
                f"""
            define dso_local void @{name}() {{
                fence {ordering}
                ret void
            }}
        """
            )
        )


def header(f, triple, features, filter_args: str):
    f.write(
        "; NOTE: Assertions have been autogenerated by "
        "utils/update_llc_test_checks.py UTC_ARGS: "
    )
    f.write(filter_args)
    f.write("\n")
    f.write(f"; The base test file was generated by ./{relpath()}\n")

    for feat in features:
        for OptFlag in ["-O0", "-O1"]:
            f.write(
                " ".join(
                    [
                        ";",
                        "RUN:",
                        "llc",
                        "%s",
                        "-o",
                        "-",
                        "-verify-machineinstrs",
                        f"-mtriple={triple}",
                        f"-mattr={feat.mattr}",
                        OptFlag,
                        "|",
                        "FileCheck",
                        "%s",
                        f"--check-prefixes=CHECK,{OptFlag}\n",
                    ]
                )
            )


def write_lit_tests(feature, datatypes, ops):
    for triple in TRIPLES:
        # Feature has no effect on fence, so keep it to one file.
        with open(f"{triple}-fence.ll", "w") as f:
            filter_args = r'--filter "^\s*(dmb)"'
            header(f, triple, Feature, filter_args)
            all_fence(f)

        for feat in feature:
            with open(f"{triple}-atomicrmw-{feat.name}.ll", "w") as f:
                filter_args = r'--filter-out "\b(sp)\b" --filter "^\s*(ld[^r]|st[^r]|swp|cas|bl|add|and|eor|orn|orr|sub|mvn|sxt|cmp|ccmp|csel|dmb)"'
                header(f, triple, [feat], filter_args)
                all_atomicrmw(f, datatypes, ops, feat.name)

            # Floating point atomics only supported for atomicrmw currently
            if feature.test_scope() == "atomicrmw":
                continue

            with open(f"{triple}-cmpxchg-{feat.name}.ll", "w") as f:
                filter_args = r'--filter-out "\b(sp)\b" --filter "^\s*(ld[^r]|st[^r]|swp|cas|bl|add|and|eor|orn|orr|sub|mvn|sxt|cmp|ccmp|csel|dmb)"'
                header(f, triple, [feat], filter_args)
                all_cmpxchg(f)

            with open(f"{triple}-atomic-load-{feat.name}.ll", "w") as f:
                filter_args = r'--filter-out "\b(sp)\b" --filter "^\s*(ld|st[^r]|swp|cas|bl|add|and|eor|orn|orr|sub|mvn|sxt|cmp|ccmp|csel|dmb)"'
                header(f, triple, [feat], filter_args)
                all_load(f)

            with open(f"{triple}-atomic-store-{feat.name}.ll", "w") as f:
                filter_args = r'--filter-out "\b(sp)\b" --filter "^\s*(ld[^r]|st|swp|cas|bl|add|and|eor|orn|orr|sub|mvn|sxt|cmp|ccmp|csel|dmb)"'
                header(f, triple, [feat], filter_args)
                all_store(f)


if __name__ == "__main__":
    os.chdir("llvm/test/CodeGen/AArch64/Atomics/")
    write_lit_tests(Feature, Type, ATOMICRMW_OPS)
    write_lit_tests(FPFeature, FPType, FP_ATOMICRMW_OPS)

    print(
        textwrap.dedent(
            """
        Testcases written. To update checks run:
            $ ./llvm/utils/update_llc_test_checks.py -u llvm/test/CodeGen/AArch64/Atomics/*.ll

        Or in parallel:
            $ parallel ./llvm/utils/update_llc_test_checks.py -u ::: llvm/test/CodeGen/AArch64/Atomics/*.ll
    """
        )
    )
