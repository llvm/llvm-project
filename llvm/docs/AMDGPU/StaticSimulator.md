(amdgpu-static-simulator)=

# AMDGPU Static Performance Simulator

## Introduction

AMDGPUSim is a standalone AMDGPU static performance simulator library. It
provides cycle-accurate simulation of instruction execution, tracking stalls
from various sources including functional unit contention, memory latencies,
and WMMA co-execution rules. Each function is simulated in a single pass over
its blocks in layout order.

### Enabling the Static Simulator

The `AMDGPUStaticSimulator` pass is off by default and can be enabled with
`-amdgpu-enable-static-simulator`. It currently supports only gfx1250.

```bash
# llc
llc -mtriple=amdgpu12.50 -amdgpu-enable-static-simulator kernel.ll -o kernel.s

# clang
clang ... -mllvm -amdgpu-enable-static-simulator

# Per-instruction trace (requires an assertions build)
llc -mtriple=amdgpu12.50 -amdgpu-enable-static-simulator \
    -amdgpu-static-sim-verbose -debug-only=amdgpu-static-simulator \
    kernel.ll -o kernel.s

# Run only the pass on MIR (the enable flag is not needed)
llc -mtriple=amdgpu12.50 -run-pass=amdgpu-static-simulator \
    -filetype=null kernel.mir
```

The simulator models expert scheduling mode. It emits a warning for functions
where expert scheduling mode is not enabled, because its estimates may be
overly optimistic.

### Expected Output

For each analyzed function, the pass prints a report to stderr in assembly
comment style:

- **Raw Metrics**: instruction, byte, cycle, and stall totals, the stall
  breakdown by source, wait counts, and WMMA co-execution utilization
- **Instruction Breakdown**: counts per instruction class
- **Derived Metrics**: compute IPC and stall ratio
- **Block Metrics**: the same counts and stalls for each block in layout order

For example, a kernel that loads two bf16 matrices and chains two WMMAs
reports:

```text
; ============================================================
; wmma_kernel - STATIC PERFORMANCE ESTIMATE (gfx1250)
; ============================================================
;
; === Raw Metrics (single layout-order traversal) ===
;   Instructions: 24
;   Bytes:        176
;   Cycles:       383
;   Stall:        359
;   FU:7 | DelayAlu:4 | Wait:311 | VaVdst:37
;   Waitcnts:     4
;   WMMA occupancy cycles: 16 | Co-executed: 1 (6%) | Blocked: 0
...
```

With the verbose options, each instruction also prints its class, unit,
latency, stall breakdown, and WMMA window state. See Verbose Logging below.

## Architecture

```text
+--------------------------------------------------------------------+
|                         User Code (Pass/Tool)                      |
+--------------------------------------------------------------------+
|                                                                    |
|                 +------------------+                               |
|                 | MachineInstrInfo |                               |
|                 |  (MIR Adapter)   |                               |
|                 +--------+---------+                               |
|                          |                                         |
|                          |  implements SimInstInfo                 |
|                          v                                         |
|              +-----------------------+                             |
|              |     SimInstInfo       | <-- Abstract Interface      |
|              |  (property queries)   |                             |
|              +-----------+-----------+                             |
|                          |                                         |
|                          v                                         |
|  +--------------------------------------------------------------+  |
|  |                     Simulator Class                          |  |
|  |  +---------+  +----------+  +-----------+  +--------------+  |  |
|  |  | SimInst |  | HWModel  |  |GPUSimState|  | simulateInst |  |  |
|  |  +---------+  +----------+  +-----------+  +--------------+  |  |
|  |                    owns GPUSimState                          |  |
|  +--------------------------------------------------------------+  |
|                                                                    |
+--------------------------------------------------------------------+
```

## Core Library

The core library, `LLVMAMDGPUSim`, contains the simulator logic with no
dependencies on LLVM CodeGen or MC layers.

### `SimInst`

`SimInst.h` defines a lightweight instruction wrapper holding:
- `void *InstPtr`: pointer to the source instruction (`MachineInstr *`)
- `InstClass Class`: cached instruction classification
- `unsigned Latency`: cached latency
- `FunctionalUnit Unit`: cached functional unit
- `unsigned InstIndex`: caller assigned sequence number used for logging

Also defines enums used throughout:
- `InstClass`: instruction classification (VALU, SALU, TRANS, WMMA, DS_READ,
  etc.)
- `FunctionalUnit`: hardware units (XDL, VALU, SALU, TRANS, LDS, VMEM, SMEM,
  BRANCH)
- `WaitType`: wait instruction types (DS, VMEMLoad, VMEMStore, SMEM, Tensor,
  XCnt, DepCtr)
- `WaitRequirement`: counter type and threshold imposed by a wait instruction

`AMDGPU::CoExecInfo` and `AMDGPU::CoExecStageType` are shared with the backend
through `AMDGPUCoExecInfo.h`.

### `SimInstInfo`

`SimInstInfo.h` defines an abstract interface for querying instruction
properties. Concrete implementations provide the actual property extraction
from the underlying instruction type. The simulator queries:

- **VALU and TRANS issue**: repeat rate (`getRepeatRate`), whether the
  instruction is a long-latency VALU (`isLOLVALU`), and resource cycles
  (`getResourceCycles`)
- **Dependencies and waits**: the `s_delay_alu` immediate (`getDelayAluImm`),
  every counter threshold imposed by a wait instruction (`getWaitInfo`), and
  the VA_VDST target (`getVaVdstTarget`). Combined waits such as
  `S_WAIT_LOADCNT_DSCNT` return two thresholds. The VA_VDST target is 15 when
  no wait applies.
- **WMMA**: the co-execution window (`getCoExecInfo`) and whether the WMMA is
  scaled (`hasWMMAScaling`)
- **Registers**: whether a VALU or WMMA reads SGPRs (`hasSGPROperands`), used
  for VA_SSRC stalls on a following SALU
- **Size**: instruction size in bytes (`getInstBytes`)

### `HWModel`

`HWModel.h` defines hardware configuration parameters including:
- Memory FIFO depths
- VA_VDST multiplier

`HWModel` is an abstract base declaring the hardware parameter fields, which
default to `~0u` (a deliberately invalid "unset" sentinel). Concrete subclasses
set target-specific values in their constructor. The only subclass today is
`GFX1250HWModel`. Obtain one via the factory `createHWModel(GPUTarget)`, which
returns `std::unique_ptr<HWModel>`.

Note that per-class default latencies (`getLatencyForClass`) and the latency
and capacity constants in the `GFX1250` namespace are gfx1250 values baked in
as `constexpr`, not part of the subclass. WMMA co-execution windows come from
the shared `AMDGPUCoExecInfo.h` model. Adding a target today requires a
`GPUTarget` enum value, a new `HWModel` subclass, and a factory case.

### `GPUSimState`

`SimState.h` defines the simulation state, including:
- Current cycle counter
- Functional unit busy times
- Pending memory operations (DS, VMEM, SMEM, TDM)
- Recent VALU/TRANS tracking for delay_alu
- Pending VA_VDST writes and VA_SSRC busy time
- Active WMMA window state

The `GPUSimState` is owned by the `Simulator` class and is not designed to be
subclassed by users. Users can inspect it via `Simulator::getState()` which
returns a `const GPUSimState &`.

### `InstrSimInfo`

`InstrInfo.h` defines the per-instruction simulation result:
- `StallCycles`: total stall cycles
- `Reason`: dominant stall reason (enum `StallReason`)
- `Breakdown`: detailed per-source stall breakdown (`StallBreakdown` struct)
- WMMA window info (stage, co-execution status)
- MSB_SET fusion, exposure, and masking flags

`StallBreakdown` is populated by the simulator core and used by the pass for
verbose logging and `StaticSimulatorBlockMetrics` attribution. It holds one
counter per stall source in the Stall Reasons table, plus `EffectiveStall`,
the actual stall applied. Sources can overlap, so `total()` returns
`EffectiveStall` rather than the sum of the counters.

### `Simulator`

The `Simulator` class in `Simulator.h` wraps the core simulation logic and
owns the `GPUSimState`, `SimInstInfo` reference, `HWModel` reference, and
configuration.

The main entry points are `simulateInst()`, which simulates one instruction
and returns its `InstrSimInfo`, and `getState()`, which exposes the current
`GPUSimState`. `SimulatorConfig` enables verbose logging and sets its output
stream.

By wrapping state ownership and configuration in the `Simulator` class,
`simulateInst()` only needs the instruction itself and an optional lookahead
buffer. The lookahead supplies the next instruction, which decides whether an
exposed `s_set_vgpr_msb` is masked by a co-execution stall.

`Simulator.cpp` implements the core simulation logic. For each instruction it:
1. computes every stall source (unit busy, co-exec, delay_alu, va_vdst,
   va_ssrc, wait, memory FIFO, etc.)
2. fills `InstrSimInfo` and `StallBreakdown` from those sources
3. records the instruction's effects on state (unit busy, WMMA window, pending
   ops, VA_VDST)

Scaled WMMAs are timed as one VALU `ld_scale` cycle followed by the XDL matrix
cycle.

## MIR Adapter

`MachineInstrInfo`, in `MIRAdapter.h` and `MIRAdapter.cpp`, is the concrete
`SimInstInfo` implementation for `MachineInstr`. It depends on AMDGPU CodeGen,
so it is compiled into `AMDGPUCodeGen` rather than `LLVMAMDGPUSim`.
- Uses `SIInstrInfo` and `SIRegisterInfo` for property extraction
- Full access to scheduling model for accurate latencies and resource cycles
- WMMA coexecution windows through `AMDGPU::getCoExecInfo()` and scaling
  through `AMDGPU::getWMMAInstInfoHelper()`

## Static Simulator Pass

`AMDGPUStaticSimulator`, in `AMDGPUStaticSimulator.h` and
`AMDGPUStaticSimulator.cpp`, is a legacy MachineFunction pass added after
branch relaxation in `GCNPassConfig::addPreEmitPass()` when
`-amdgpu-enable-static-simulator` is set. For each function it:
1. skips subtargets without `hasGFX1250Insts()`
2. warns when expert scheduling mode is not enabled
3. creates a `MachineInstrInfo` adapter, a GFX1250 `HWModel` from
   `createHWModel()`, and one `Simulator`, then walks the blocks once in
   layout order. Bundle headers and meta instructions are skipped. Each
   remaining instruction is passed to `simulateInst()`, with the next
   instruction as lookahead for `s_set_vgpr_msb`
4. aggregates `StaticSimulatorBlockMetrics` per block and for the function
5. prints the `StaticSimulatorReport` to stderr

The pass does not modify the function.

## Stall Reasons

The simulator tracks these stall sources:

| Reason | Description |
|--------|-------------|
| `FU_BUSY` | Functional unit or VALU resource not available |
| `COEXEC_BLOCKED` | WMMA co-execution window or scaled WMMA scale-read slot blocks issue |
| `LONG_LAT_VALU` | Long-latency VALU waits for the active WMMA window to end |
| `LOLVALU_TRANS_HAZARD` | TRANS and long-latency VALU must issue two cycles apart |
| `VA_SSRC_STALL` | VALU or WMMA with SGPR operands blocks a following SALU |
| `VA_VDST_WAIT` | Waiting for VA_VDST counter |
| `WAITCNT` | Explicit wait instruction |
| `DELAY_ALU` | delay_alu dependency |
| `MEM_FIFO` | DS, VMEM, or TDM FIFO full |
| `MSB_SET_EXPOSED` | s_set_vgpr_msb neither fused nor masked |

## Verbose Logging

With `-amdgpu-static-sim-verbose -debug-only=amdgpu-static-simulator` in an
assertions build, the simulator core prints:

- **Stall result**: `AMDGPU static simulator instruction 18 stall 7`
- **Unit state**: `-> LastVALUCycle = 2`

The pass adds its own logging for:
- Instruction header (cycle, opcode, class, unit, latency):
  `Class: VALU | Unit: VALU | Latency: 5 | ResourceCycles: 1 | Size: 4 bytes`
- Stall breakdown: `Stalls: FU=7 -> Total: 7`
- WMMA occupancy: `Class: WMMA | Unit: XDL | Occupancy: 8 | Window: 9`
- WMMA pattern: `-> WMMA pattern: 0EIIEEIIV`
- WMMA window stage: `-> WMMA[1/9] E OK`
- Active WMMA: `-> ActiveWMMA: cycles 346-355 [back-to-back]`
- Cycle advance: `-> Advancing cycle: 339 -> 346`
- MSB_SET fusion and exposure
- Block and function summaries

## Build Configuration

The simulator is built if the AMDGPU backend is enabled with
`LLVM_TARGETS_TO_BUILD`. A detailed report containing cycle information prints
by default, but the per-instruction trace (`-amdgpu-static-sim-verbose`)
requires an assertions build (`LLVM_ENABLE_ASSERTIONS`).

## Design Benefits

1. **Clean Separation**: core simulator doesn't know about MachineInstr
2. **Lazy Evaluation**: properties computed on demand, not upfront
3. **Extensibility**: easy to add new instruction sources (e.g., external
   tools)
4. **Small SimInst**: only caches frequently used fields (Class, Latency, Unit)
5. **Minimal Dependencies**: core library only depends on LLVM Support
6. **Encapsulated State**: `Simulator` class owns `GPUSimState`, and users
   inspect it via const ref
7. **Configurable Logging**: simulator verbose output is controlled by
   `SimulatorConfig`, and the pass adds its own per-instruction and summary
   logging

## Future Work

- MC layer support through an `MCInstInfo` adapter
- Per-target latencies: move the target-independent defaults among the
  `GFX1250` constants into a base class, with per-target subclasses such as
  `GFX1250Latency` overriding the values that differ. `getLatencyForClass`
  then becomes a member accessor on that hierarchy.
