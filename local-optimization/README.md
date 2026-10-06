# Local Optimization Pass

Five local optimizations in one LLVM pass, `local-opt`. Each optimization is
a separate function in [src/LocalOptimization.cpp](src/LocalOptimization.cpp),
and `run()` executes them. "Local" means one basic block at a time.

| # | Function | What it does | Example |
|---|---|---|---|
| 1 | `constantPropagation` | Replace a variable read with the value last stored to it (constant or copy) | `a = 2; c = 4 + a` → `c = 4 + 2` |
| 2 | `instCombine` | Constant folding | `4 + 2 + 3` → `9` |
|   |   | Algebraic identities | `x/x` → `1`, `x-x` → `0`, `x*1` → `x`, `x+0` → `x` |
|   |   | Power of 2 as a shift | `x * 8` → `x << 3` |
| 3 | `deadCodeElimination` | Remove unused instructions with no side effects | unused `a + b` removed; unused `printf()` kept |
|   |   | Remove redundant assignments | `a = 3; a = 3;` → `a = 3;` |
| 4 | `strengthReduction` | Multiply by a power of 2 → left shift | `2*c` → `c << 1`, `f*8` → `f << 3` |
| 5 | `commonSubexpressionElimination` | Reuse an identical earlier expression | `a = b+c; d = b+c` → `d = a` |

`run()` repeats the five until nothing changes, because each one creates work
for the others. For example, propagation exposes constants that can then be
folded.

`x * 2^N → x << N` is listed under both InstCombine and strength reduction in
the assignment, so both use the same helper, `mulToShift()`.

## Layout

```
local-optimization/
├── CMakeLists.txt
├── README.md
├── src/LocalOptimization.cpp
└── testcases/
    ├── *.c             testcases, expected result in CHECK lines at the bottom
    └── run_tests.sh
```

## Build

You need LLVM built with clang in `build/` at the root of this repository:

```sh
cmake -S llvm -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS=clang
ninja -C build clang opt FileCheck
```

Then build the pass. This produces `build/LocalOptimization.so`:

```sh
cd local-optimization
cmake -S . -B build -G Ninja -DLLVM_DIR=$PWD/../build/lib/cmake/llvm
cmake --build build
```

## Run

```sh
../build/bin/clang -S -emit-llvm -O0 -Xclang -disable-O0-optnone testcases/cse.c -o cse.ll
../build/bin/opt -load-pass-plugin=build/LocalOptimization.so -passes=local-opt -S cse.ll
```

`-Xclang -disable-O0-optnone` is needed because at `-O0` clang marks every
function `optnone`, and `opt` would skip it.

## Test

```sh
./testcases/run_tests.sh
```

Each testcase is compiled, optimized, and compared with the `CHECK` lines at
its end. The generated IR is written to `build/testcases/`.

| Testcase | Result |
|---|---|
| `redundant_assignment.c` | second `a=3`, `b=4` removed; `a+b` → `7`; `return 3` |
| `algebraic_identity.c` | all arithmetic removed; `return 23` |
| `copy_propagation.c` | `e = c + b` → `e = d + 4` |
| `constant_folding.c` | `c = 9`, `result = 45`, `return 22` |
| `strength_reduction.c` | `2*c` → `c << 1`, `f*8` → `f << 3` |
| `cse.c` | `b + c` computed once; `return a * a` |
