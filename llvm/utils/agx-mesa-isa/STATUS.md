# Strict mode (`agx-strict`): status, 2026-10-21

Strict mode makes the AGX toolchain emit only instruction forms that Mesa's asahi compiler emits for M1 (G13, G13X)
and M2 (G14, G14D): the 133 forms `mesa_isa.py` reads off Mesa's packer (`src/asahi/compiler/agx_pack.c`) and the
operand values its compiler creates. Code built this way relies only on encodings a production driver uses.

## How to turn it on

| Layer | Switch | Repository, branch `claude/bold-davinci-46xi9q` |
| --- | --- | --- |
| LLVM IR | `"target-features"="+agx-strict"`, `-mattr=+agx-strict`, or module flag `!{i32 1, !"agx-strict", i32 1}` | Jianwen-Ding/llvm-project, `llvm/docs/AGXUsage.rst` "Strict mode" |
| MLIR | `#agxir.target<chip = "...", features = "+agx-strict">`; `agx-expand-ops features=+agx-strict` | Jianwen-Ding/llvm-project |
| Fat binaries | `CHIP+agx-strict` entries from `agx-build-fatbin.py` (default; `--no-agx-strict` leaves them out) | Jianwen-Ding/llvm-project, `mlir/utils/agx-kernels` |
| Simulator | `--agx-strict GPU` on `agx_run`, `agx_dispatch`, `agx_cost`, or `AGX_STRICT=GPU` | Jianwen-Ding/AGX-Performance-Simulator (PR #8) |
| ZLUDA | `zoc --agx-strict`; `ZLUDA_AGX_STRICT=1` for `zluda_agx` (loads the `+agx-strict` entries) | Jianwen-Ding/ZLUDA, `docs/src/agx.md` "Strict instruction mode" |
| Sail | `sail/agx/gen/mesacheck.py` | Jianwen-Ding/PTX-formalization |

Only G13 and G14 processors are accepted; any other is a fatal error. Every encoded instruction (objects, assembly
output and `llvm-mc`) is checked, and one Mesa would not emit is an error naming the closest Mesa form.

## Evidence (`tools/check-all.sh`, all nine checks pass)

| Check | Result |
| --- | --- |
| tables | the backend's `AGXMesaISA.inc`, `mesa-isa.json` and the simulator's `mesa_table.rs` are what `mesa_isa.py` generates |
| fuzz | Mesa's own packer: 32,700 instructions over 109 opcodes accepted; the Python, C++ and Rust engines agree on 21,800 mutated instructions; random bytes accepted 0.33% |
| lit | 59 AGX CodeGen / MC / disassembler tests |
| codegen | every CodeGen/AGX test compiles with `+agx-strict` for agx-g13g and agx-g14c |
| mlir | the MLIR AGX, AGX-to-AGXIR and GPU-to-AGXIR tests and the ChipInfo unit tests |
| fatbin | ZLUDA's sgemm builds with `agx-g13g`, `agx-g14c` and both `+agx-strict` entries; all 8 library kernel sources build in strict mode |
| simulator | `cargo test --release` (the strict check and its corpus tests) |
| zluda | 388 of 388 PTX test kernels compile with `--agx-strict` |
| sail | the Sail model decodes every non-graphics instruction of Mesa's packed corpus: g14 440 of 440 and g13 405 of 405 (85 opcodes each) |

Also measured: GEMM kernels correct on the simulator for G13, G14 and both strict entries; ZLUDA conformance on
agx-g14c and agx-g13g identical in strict and default mode (strict FP: 282 programs, all 1,728 checks). Cost on
agx-bench: code +6.0%, cycles +0.4%.

## M2 measurements integrated

AGX-Performance-Simulator `tasks/ptx-re-blockers.md`, "Where the answers went", lists each one: scratch memory with
Mesa's helper (B1), launch limits (B22), co-residency (B5, B30), fault detection from zero timestamps (B20), the G14
block barrier (B16), saturating converts (B27), no jump over short ALU bodies (B31, with its cost in the simulator),
host-coherent accesses (B7), and the documented behaviours B4, B21 and B26.

## Not done

- **Not run on hardware:** scratch (B1), fault detection (B20) and the launch limits (B22) follow the M2's
  measurements and recipe, but the M2 has been offline since.
- **B33 (vectorizing memory accesses):** the backend has no legal vector types to do it with.
- **The Sail model's transcendentals** are correctly rounded rather than the hardware's tables (about one ulp
  apart), and the packed device-memory formats decode but do not execute (unmeasured). See PTX-formalization
  `sail/README.md`.
- **Error traces:** hardware faults report almost nothing to the process (soft faults hide out-of-bounds accesses).
  A compiler-instrumented debug mode (`--agx-checked`) was proposed, not built.
