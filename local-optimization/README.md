# Local Optimization Pass for LLVM

Five local optimizations, implemented as one LLVM function pass named
`local-opt`. Each optimization is a separate function; `run()` executes them.

| # | Optimization | Function |
|---|---|---|
| 1 | Constant propagation (and copy propagation) | `constantPropagation()` |
| 2 | Instruction combining: constant folding, algebraic identities, power of 2 as a shift | `instCombine()` |
| 3 | Dead code elimination (and redundant assignment elimination) | `deadCodeElimination()` |
| 4 | Strength reduction | `strengthReduction()` |
| 5 | Common subexpression elimination | `commonSubexpressionElimination()` |

"Local" means each optimization looks at one basic block at a time.

## Directory layout

```
local-optimization/
├── README.md
├── CMakeLists.txt              builds the pass as an opt plugin
├── src/
│   ├── LocalOptimization.h     the pass class: run() + the five functions
│   └── LocalOptimization.cpp   the implementation and plugin registration
└── testcases/
    ├── *.c                     C testcases, expected results at the bottom
    └── run_tests.sh            compiles, optimizes and checks every testcase
```

## Building

### 1. LLVM

You need an LLVM build containing `clang`, `opt` and `FileCheck`. The
expected location is `build/` at the root of this repository (`../build`
from this directory). To build it, from the repository root:

```sh
cmake -S llvm -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_PROJECTS=clang \
      -DLLVM_TARGETS_TO_BUILD=X86
ninja -C build clang opt FileCheck
```

Plugins are enabled by default on Linux (`LLVM_ENABLE_PLUGINS=ON`).

### 2. This pass

```sh
cd local-optimization
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_DIR=$PWD/../build/lib/cmake/llvm
cmake --build build
```

This produces `build/LocalOptimization.so`. It is loaded into `opt` at run
time, so LLVM does not have to be modified or rebuilt.

## Running

```sh
LLVM=../build/bin

# C -> LLVM IR
$LLVM/clang -S -emit-llvm -O0 -Xclang -disable-O0-optnone \
            -fno-discard-value-names testcases/cse.c -o cse.ll

# LLVM IR -> optimized LLVM IR
$LLVM/opt -load-pass-plugin=build/LocalOptimization.so \
          -passes=local-opt -S cse.ll -o cse.opt.ll
```

The clang flags:
- `-Xclang -disable-O0-optnone` is required. At `-O0` clang marks every
  function `optnone`, and `opt` then skips all optimization passes on it.
- `-fno-discard-value-names` is optional. It keeps C variable names (`%a`,
  `%b`) in the IR, which makes the output easier to read.

### Pass options

| Pipeline | Runs |
|---|---|
| `-passes=local-opt` | all five optimizations |
| `-passes='local-opt<strength-reduction>'` | only strength reduction |
| `-passes='local-opt<constprop;instcombine>'` | only the ones listed |
| `-passes='local-opt<verbose>'` | all five, printing every change to stderr |

The option names are `constprop`, `instcombine`, `dce`,
`strength-reduction`, `cse` and `verbose`. Example of the `verbose` output:

```
[instcombine] %div = sdiv i32 %a, %a   -->   i32 1
[instcombine] %sub = sub nsw i32 %b, %b   -->   i32 0
[dce] store i32 3, ptr %a, align 4   -->   removed (redundant assignment)
```

## Testcases

```sh
cmake --build build --target check     # or: ./testcases/run_tests.sh
```

Each testcase is compiled with clang, optimized twice, and the output is
checked with LLVM's `FileCheck` against the expected results written at the
bottom of the testcase:

- **`ONLY`** checks the result with just the optimization(s) the test is
  about (named on the `TEST-OPTS:` line).
- **`ALL`** checks the result with all five optimizations.

The generated IR is written to `build/testcases/`: `<name>.ll` (before),
`<name>.only.ll` and `<name>.all.ll`.

| Testcase | Tests | Result with all five |
|---|---|---|
| `redundant_assignment.c` | DCE: redundant assignment `a=3; a=3;` | `return 3` |
| `algebraic_identity.c` | InstCombine: `a/a`, `b-b`, `r/r`, `r-r`, `+ 23` | `return 23` |
| `copy_propagation.c` | Constant propagation: `c=d; e=c+b` | `return d + 4` |
| `constant_folding.c` | Constant propagation + folding | `return 22` |
| `strength_reduction.c` | Strength reduction: `2*c`, `f*8` | `return 3` (`d`, `e` are dead) |
| `power_of_two.c` | Powers of 2: `*16`, unsigned `/8` and `%4` | shift, shift, and; signed `/` kept |
| `dead_code.c` | DCE keeps `printf`, removes the unused computation | `printf(...); return 10` |
| `cse.c` | CSE: `b+c`, `b+c`, `c+b` | one addition, reused |

```
PASS  algebraic_identity     local-opt<constprop;instcombine>, local-opt
PASS  constant_folding       local-opt<constprop;instcombine>, local-opt
PASS  copy_propagation       local-opt<constprop>, local-opt
PASS  cse                    local-opt<constprop;cse>, local-opt
PASS  dead_code              local-opt<dce>, local-opt
PASS  power_of_two           local-opt<instcombine>, local-opt
PASS  redundant_assignment   local-opt<dce>, local-opt
PASS  strength_reduction     local-opt<strength-reduction>, local-opt

8 passed, 0 failed.
```

## The optimizations

At `-O0`, clang puts every C variable in memory: `int a = 2;` becomes an
`alloca` (the variable), a `store` (assignment) and a `load` (each read). The
pass works directly on that form.

Optimizations 1 and 3 only touch a variable whose address is never taken.
Once `&a` is passed somewhere (`scanf("%d", &a)`), a function call could
change `a` behind the pass's back, so its loads and stores are left alone.

### 1. Constant propagation — `constantPropagation()`

The pass walks each basic block and remembers the value each variable holds,
replacing every load of the variable with that value.

- If the value is a constant, this is constant propagation.
- If the value is another variable's value, it is copy propagation.

`copy_propagation.c`, `c = d; e = c + b;`:

```llvm
; before                                  ; after
store i32 4, ptr %b                       store i32 4, ptr %b
%0 = load i32, ptr %d                     %0 = load i32, ptr %d
store i32 %0, ptr %c                      store i32 %0, ptr %c
%1 = load i32, ptr %c                     %add = add nsw i32 %0, 4
%2 = load i32, ptr %b                     ret i32 %add
%add = add nsw i32 %1, %2
...
```

The knowledge starts empty in every block, because what a variable holds on
entry depends on which block ran before.

### 2. Instruction combining — `instCombine()`

Every integer arithmetic instruction is tried against three groups of rules,
in this order:

**a. Constant folding.** Both operands are constants: `2 + 3 → 5`,
`5 * 9 → 45`, `45 / 2 → 22`.

Operations whose result is undefined are *not* folded: division by zero,
`INT_MIN / -1`, and shifts by the bit width or more.

**b. Algebraic identities.**

| Rule | Rule | Rule |
|---|---|---|
| `x + 0 → x` | `x * 1 → x` | `x / 1 → x` |
| `x - 0 → x` | `x * 0 → 0` | `x / x → 1` |
| `x - x → 0` | `x % 1 → 0` | `x % x → 0` |
| `x & x → x` | `x \| 0 → x` | `x ^ x → 0` |
| `x << 0 → x` | `x & 0 → 0` | `x ^ 0 → x` |

`x / x → 1` looks wrong for `x = 0`, but dividing by zero is undefined
behaviour in C and in LLVM IR. The compiler may therefore assume `x ≠ 0`.

In `algebraic_identity.c`, constant propagation and these rules reduce the
whole function to `return 23`:

```
a/a → 1    b/b → 1    1*1 → 1    b-b → 0    1+0 → 1
r/r → 1    r-r → 0    0+23 → 23
```

**c. Power of 2 as a shift.** `x * 2^N → x << N`, for example
`x * 16 → x << 4`.

### 3. Dead code elimination — `deadCodeElimination()`

The basic rule: an instruction that is **unused and has no side effects** is
removed. An unused instruction **with** side effects is kept: the result of
`printf("Hello\n")` is never used, but it prints, so it stays.

Removing an instruction can make its operands unused, so a worklist checks
them again.

At `-O0` the dead code is usually a `store`. A store counts as a side
effect, so two extra rules handle it:

- **Redundant assignment.** A store that writes the value the variable
  already holds is removed (`int a = 3; a = 3;`).
- **Variable never read.** If a variable is never loaded, its stores cannot
  be observed. The stores and the variable are removed, which can leave the
  computed value dead too (`c = a + b` with `c` unused).

`dead_code.c` after all five optimizations:

```llvm
%call = call i32 (ptr, ...) @printf(ptr noundef @.str)
ret i32 10
```

### 4. Strength reduction — `strengthReduction()`

Expensive operations by a power of two are replaced by cheap ones:

| Before | After |
|---|---|
| `x * 2^N` | `x << N` |
| `x / 2^N` (unsigned) | `x >> N` |
| `x % 2^N` (unsigned) | `x & (2^N - 1)` |

`strength_reduction.c`, with `local-opt<strength-reduction>`:

```llvm
%mul = mul nsw i32 2, %0        →    %mul = shl nsw i32 %0, 1
%mul1 = mul nsw i32 %1, 8       →    %mul1 = shl nsw i32 %1, 3
```

Signed division is not changed. In C, `-7 / 2` is `-3` (rounds toward zero),
but `-7 >> 1` is `-4` (rounds down).

The `nsw` ("no signed overflow") flag is kept, except when `2^N` is the sign
bit (`x * 2^31` for `int`). As a signed number that constant is negative, so
the flag would no longer be correct.

The assignment lists multiplication by a power of 2 under both InstCombine
and strength reduction, so both functions use the same helper,
`mulByPowerOf2ToShift()`. That way each function produces the shift even
when it runs alone. When all five run, InstCombine runs first and does it.

### 5. Common subexpression elimination — `commonSubexpressionElimination()`

The pass walks each basic block keeping a list of the expressions computed so
far. If an identical expression was already computed, its result is reused
and the new copy is deleted.

`cse.c`, `a = b + c; d = b + c; e = c + b; return a * d * e;`:

```llvm
%add = add nsw i32 %b, %c
%mul = mul nsw i32 %add, %add
%mul3 = mul nsw i32 %mul, %add
ret i32 %mul3
```

- Commutative operations match in either order, so `c + b` is the same as
  `b + c`.
- This is safe because LLVM IR is in SSA form: `%b` and `%c` can never change
  after they are defined.
- Only pure expressions are considered: arithmetic, comparisons, casts and
  address calculations. Loads and calls are not, because memory can change in
  between.

At `-O0`, the two `b + c` are computed from separate loads of `b` and `c`.
They become identical once constant propagation has replaced those loads with
the arguments `%b` and `%c`.

## run()

```
run()
 └─ repeat until nothing changes:
      constantPropagation()
      instCombine()
      deadCodeElimination()
      strengthReduction()
      commonSubexpressionElimination()
```

The optimizations create work for each other:
- Propagation exposes constants for folding.
- Folding makes the original loads dead.
- Removing a dead read can make a variable "never read".

So the sequence is repeated until a whole round changes nothing. This always
ends: every transformation either deletes an instruction or replaces a
multiply/divide/remainder with a shift or mask, and neither can go on
forever.

## Limitations

- **Local only.** Values are not tracked from one basic block into the next.
  Code with `if` or loops is optimized block by block.
- **Integers only** for folding, identities and strength reduction.
- **Signed division by a power of 2 is not reduced** (see section 4).
- **Uninitialized variables are left alone.** Reading one, like `d` in
  `copy_propagation.c`, stays a load; the pass does not exploit the
  undefined value.
