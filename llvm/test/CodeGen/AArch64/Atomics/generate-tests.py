#!/usr/bin/env python3
import textwrap
import enum
import os
"""
Generate the tests in llvm/test/CodeGen/AArch64/Atomics. Run from top level llvm-project.
"""

TRIPLES = [
    'aarch64',
    'aarch64_be',
]


# Type name size
class Type(enum.Enum):
    # Value is the size in bytes
    i8 = 1
    i16 = 2
    i32 = 4
    i64 = 8
    i128 = 16

    def align(self, aligned: bool) -> int:
        return self.value if aligned else 1

    def __str__(self) -> str:
        return self.name


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
    lse128 = enum.auto()  # FEAT_LSE128

    @property
    def mattr(self):
        if self == Feature.outline_atomics:
            return '+outline-atomics'
        if self == Feature.v8_1a:
            return '+v8.1a'
        if self == Feature.rcpc3:
            return '+lse2,+rcpc3'
        return '+' + self.name


ATOMICRMW_OPS = [
    'xchg',
    'add',
    'sub',
    'and',
    'nand',
    'or',
    'xor',
    'max',
    'min',
    'umax',
    'umin',
]


def all_atomicrmw(f):
    for op in ATOMICRMW_OPS:
        for aligned in Aligned:
            for ty in Type:
                for ordering in ATOMICRMW_ORDERS:
                    name = f'atomicrmw_{op}_{ty}_{aligned}_{ordering}'
                    instr = 'atomicrmw'
                    f.write(
                        textwrap.dedent(f'''
                        define dso_local {ty} @{name}(ptr %ptr, {ty} %value) {{
                            %r = {instr} {op} ptr %ptr, {ty} %value {ordering}, align {ty.align(aligned)}
                            ret {ty} %r
                        }}
                    '''))


def all_load(f):
    for aligned in Aligned:
        for ty in Type:
            for ordering in ATOMIC_LOAD_ORDERS:
                for const in [False, True]:
                    name = f'load_atomic_{ty}_{aligned}_{ordering}'
                    instr = 'load atomic'
                    if const:
                        name += '_const'
                    arg = 'ptr readonly %ptr' if const else 'ptr %ptr'
                    f.write(
                        textwrap.dedent(f'''
                        define dso_local {ty} @{name}({arg}) {{
                            %r = {instr} {ty}, ptr %ptr {ordering}, align {ty.align(aligned)}
                            ret {ty} %r
                        }}
                    '''))


def all_store(f):
    for aligned in Aligned:
        for ty in Type:
            for ordering in ATOMIC_STORE_ORDERS:  # FIXME stores
                name = f'store_atomic_{ty}_{aligned}_{ordering}'
                instr = 'store atomic'
                f.write(
                    textwrap.dedent(f'''
                    define dso_local void @{name}({ty} %value, ptr %ptr) {{
                        {instr} {ty} %value, ptr %ptr {ordering}, align {ty.align(aligned)}
                        ret void
                    }}
                '''))


def all_cmpxchg(f):
    for aligned in Aligned:
        for ty in Type:
            for success_ordering in CMPXCHG_SUCCESS_ORDERS:
                for failure_ordering in CMPXCHG_FAILURE_ORDERS:
                    for weak in [False, True]:
                        name = f'cmpxchg_{ty}_{aligned}_{success_ordering}_{failure_ordering}'
                        instr = 'cmpxchg'
                        if weak:
                            name += '_weak'
                            instr += ' weak'
                        f.write(
                            textwrap.dedent(f'''
                            define dso_local {ty} @{name}({ty} %expected, {ty} %new, ptr %ptr) {{
                                %pair = {instr} ptr %ptr, {ty} %expected, {ty} %new {success_ordering} {failure_ordering}, align {ty.align(aligned)}
                                %r = extractvalue {{ {ty}, i1 }} %pair, 0
                                ret {ty} %r
                            }}
                        '''))


def all_fence(f):
    for ordering in FENCE_ORDERS:
        name = f'fence_{ordering}'
        f.write(
            textwrap.dedent(f'''
            define dso_local void @{name}() {{
                fence {ordering}
                ret void
            }}
        '''))


def header(f, triple, features, filter_args: str):
    f.write('; NOTE: Assertions have been autogenerated by '
            'utils/update_llc_test_checks.py UTC_ARGS: ')
    f.write(filter_args)
    f.write('\n')
    f.write(f'; The base test file was generated by {__file__}\n')
    for feat in features:
        for OptFlag in ['-O0', '-O1']:
            f.write(' '.join([
                ';', 'RUN:', 'llc', '%s', '-o', '-', '-verify-machineinstrs',
                f'-mtriple={triple}', f'-mattr={feat.mattr}', OptFlag, '|',
                'FileCheck', '%s', f'--check-prefixes=CHECK,{OptFlag}\n'
            ]))


def write_lit_tests():
    os.chdir('llvm/test/CodeGen/AArch64/Atomics/')
    for triple in TRIPLES:
        # Feature has no effect on fence, so keep it to one file.
        with open(f'{triple}-fence.ll', 'w') as f:
            filter_args = r'--filter "^\s*(dmb)"'
            header(f, triple, Feature, filter_args)
            all_fence(f)

        for feat in Feature:
            with open(f'{triple}-atomicrmw-{feat.name}.ll', 'w') as f:
                filter_args = r'--filter-out "\b(sp)\b" --filter "^\s*(ld[^r]|st[^r]|swp|cas|bl|add|and|eor|orn|orr|sub|mvn|sxt|cmp|ccmp|csel|dmb)"'
                header(f, triple, [feat], filter_args)
                all_atomicrmw(f)

            with open(f'{triple}-cmpxchg-{feat.name}.ll', 'w') as f:
                filter_args = r'--filter-out "\b(sp)\b" --filter "^\s*(ld[^r]|st[^r]|swp|cas|bl|add|and|eor|orn|orr|sub|mvn|sxt|cmp|ccmp|csel|dmb)"'
                header(f, triple, [feat], filter_args)
                all_cmpxchg(f)

            with open(f'{triple}-atomic-load-{feat.name}.ll', 'w') as f:
                filter_args = r'--filter-out "\b(sp)\b" --filter "^\s*(ld|st[^r]|swp|cas|bl|add|and|eor|orn|orr|sub|mvn|sxt|cmp|ccmp|csel|dmb)"'
                header(f, triple, [feat], filter_args)
                all_load(f)

            with open(f'{triple}-atomic-store-{feat.name}.ll', 'w') as f:
                filter_args = r'--filter-out "\b(sp)\b" --filter "^\s*(ld[^r]|st|swp|cas|bl|add|and|eor|orn|orr|sub|mvn|sxt|cmp|ccmp|csel|dmb)"'
                header(f, triple, [feat], filter_args)
                all_store(f)

if __name__ == '__main__':
    write_lit_tests()

    print(textwrap.dedent('''
        Testcases written. To update checks run:
            $ ./llvm/utils/update_llc_test_checks.py -u llvm/test/CodeGen/AArch64/Atomics/*.ll

        Or in parallel:
            $ parallel ./llvm/utils/update_llc_test_checks.py -u ::: llvm/test/CodeGen/AArch64/Atomics/*.ll
    '''))
