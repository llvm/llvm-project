# AMDGPUSim Library Design

## Overview

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

# Per-instruction verbose output (requires an assertions build)
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

```
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

```
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
|  |  | SimInst |  | HWModel  |  | SimState  |  | simulateInst |  |  |
|  |  +---------+  +----------+  +-----------+  +--------------+  |  |
|  |                    owns GPUSimState                          |  |
|  +--------------------------------------------------------------+  |
|                                                                    |
+--------------------------------------------------------------------+
```

## Components

### Core Library (`LLVMAMDGPUSim`)

The core library contains the simulator logic with no dependencies on LLVM
CodeGen or MC layers.

#### `SimInst` (SimInst.h)

Lightweight instruction wrapper holding:
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

#### `SimInstInfo` (SimInstInfo.h)

Abstract interface for querying instruction properties. Concrete
implementations provide the actual property extraction from the underlying
instruction type.

```cpp
class SimInstInfo {
public:
  // VALU/TRANS
  virtual unsigned getRepeatRate(const SimInst &SI) const = 0;
  virtual bool isLOLVALU(const SimInst &SI) const = 0;
  virtual unsigned getResourceCycles(const SimInst &SI) const = 0;

  // delay_alu / Wait
  virtual unsigned getDelayAluImm(const SimInst &SI) const = 0;
  virtual WaitRequirements getWaitInfo(const SimInst &SI) const = 0;
  virtual unsigned getVaVdstTarget(const SimInst &SI) const = 0;

  // WMMA
  virtual std::optional<AMDGPU::CoExecInfo>
  getCoExecInfo(const SimInst &SI) const = 0;
  virtual bool hasWMMAScaling(const SimInst &SI) const = 0;

  // Registers
  virtual bool hasSGPROperands(const SimInst &SI) const = 0;

  // Counting
  virtual unsigned getInstBytes(const SimInst &SI) const = 0;
};
```

Key notes:
- `getWaitInfo()` returns every counter threshold imposed by one instruction.
  Combined waits such as `S_WAIT_LOADCNT_DSCNT` return two requirements.
- `getVaVdstTarget()` returns 15 when no VA_VDST wait applies.
- `hasSGPROperands()` indicates VALU or WMMA instructions that read SGPRs, used
  for VA_SSRC stalls on following SALU instructions.

#### `HWModel` (HWModel.h)

Hardware configuration parameters including:
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

#### `GPUSimState` (SimState.h)

Simulation state including:
- Current cycle counter
- Functional unit busy times
- Pending memory operations (DS, VMEM, SMEM, TDM)
- Recent VALU/TRANS tracking for delay_alu
- Pending VA_VDST writes and VA_SSRC busy time
- Active WMMA window state

The `GPUSimState` is owned by the `Simulator` class and is not designed to be
subclassed by users. Users can inspect it via `Simulator::getState()` which
returns a `const GPUSimState &`.

#### `InstrSimInfo` (InstrInfo.h)

Per-instruction simulation result:
- `StallCycles`: total stall cycles
- `Reason`: dominant stall reason (enum `StallReason`)
- `Breakdown`: detailed per-source stall breakdown (`StallBreakdown` struct)
- WMMA window info (stage, co-execution status)
- MSB_SET fusion, exposure, and masking flags

##### `StallBreakdown`

Detailed stall breakdown populated by the simulator core, used by the pass for
verbose logging and `StaticSimulatorBlockMetrics` attribution:

```cpp
struct StallBreakdown {
  unsigned EffectiveStall;
  unsigned FU, VALUSlot, CoExec, CoExecFromEffective;
  unsigned DelayAlu, WaitCnt, MemFIFO;
  unsigned LongLatVALU, LOLVALUTRANSHazard, SSRC, VaVdst;

  unsigned total() const;    // EffectiveStall, since sources can overlap
};
```

#### `Simulator` Class (Simulator.h)

The `Simulator` class wraps the core simulation logic and owns the
`GPUSimState`, `SimInstInfo` reference, `HWModel` reference, and configuration.

```cpp
struct SimulatorConfig {
  bool Verbose = false;          // Enable verbose logging
  raw_ostream *Log = nullptr;    // Output stream for verbose logging
};

class Simulator {
  class Impl;
  std::unique_ptr<Impl> PImpl;

public:
  Simulator(const SimInstInfo &II, const HWModel &M, SimulatorConfig C = {});

  InstrSimInfo simulateInst(const SimInst &Inst, ArrayRef<SimInst> Lookahead = {});
  void advanceCycles(unsigned N);
  const GPUSimState &getState() const;
  const SimulatorConfig &getConfig() const;
  const HWModel &getModel() const;
};
```

By wrapping state ownership and configuration in the `Simulator` class,
`simulateInst()` only needs the instruction itself and an optional lookahead
buffer. The lookahead supplies the next instruction, which decides whether an
exposed `s_set_vgpr_msb` is masked by a co-execution stall.

#### `Simulator.cpp`

Core simulation logic implementing:
1. `computeStallSources()`: computes all stall sources (unit busy, co-exec,
   delay_alu, va_vdst, va_ssrc, wait, memory FIFO, etc.)
2. `populateInfo()`: fills `InstrSimInfo` and `StallBreakdown` from stall
   sources
3. `recordInstruction()`: records instruction effects on state (unit busy, WMMA
   window, pending ops, VA_VDST)
4. Scaled WMMA timing: one VALU `ld_scale` cycle followed by the XDL matrix
   cycle

### MIR Adapter

#### `MachineInstrInfo` (MIRAdapter.h/.cpp)

Concrete `SimInstInfo` implementation for `MachineInstr`. It depends on AMDGPU
CodeGen, so it is compiled into `AMDGPUCodeGen` rather than `LLVMAMDGPUSim`.
- Uses `SIInstrInfo` and `SIRegisterInfo` for property extraction
- Full access to scheduling model for accurate latencies and resource cycles
- WMMA coexecution windows through `AMDGPU::getCoExecInfo()` and scaling
  through `AMDGPU::getWMMAInstInfoHelper()`

### Static Simulator Pass

#### `AMDGPUStaticSimulator` (AMDGPUStaticSimulator.h/.cpp)

A legacy MachineFunction pass added after branch relaxation in
`GCNPassConfig::addPreEmitPass()` when `-amdgpu-enable-static-simulator` is
set. For each function it:
1. skips subtargets without `hasGFX1250Insts()`
2. warns when expert scheduling mode is not enabled
3. walks the blocks once in layout order with one `Simulator`, skipping bundle
   headers and meta instructions, and passes the next instruction as lookahead
   for `s_set_vgpr_msb`
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

When `SimulatorConfig::Verbose` is true and `Log` is non-null, the simulator
outputs:

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

## Usage Example

```cpp
#include "AMDGPUSim/AMDGPUSim.h"
#include "AMDGPUSim/MIRAdapter.h"

using namespace llvm::AMDGPUSim;

void analyzeFunction(MachineFunction &MF) {
  const SIInstrInfo &TII = *MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  MachineInstrInfo InstInfo(TII, TII.getRegisterInfo());
  std::unique_ptr<HWModel> Model = createHWModel(GPUTarget::GFX1250);

  SimulatorConfig Cfg;
  Cfg.Verbose = true;
  Cfg.Log = &dbgs();
  Simulator Sim(InstInfo, *Model, Cfg);

  unsigned TotalStalls = 0;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB.instrs()) {
      // Bundle headers and meta instructions do not issue.
      if (MI.isBundle() || MI.isMetaInstruction())
        continue;

      InstrSimInfo Result = Sim.simulateInst(InstInfo.createSimInst(MI));
      TotalStalls += Result.StallCycles;

      if (Result.Reason != StallReason::NONE) {
        // Attribute stalls via Result.Breakdown
      }
    }
  }

  // Total cycles include both issue cycles and stall cycles.
  unsigned TotalCycles = Sim.getState().CurrentCycle;
}
```

## Build Configuration

The core library is built in `AMDGPUSim/CMakeLists.txt` and links only LLVM
Support:

```cmake
add_llvm_component_library(LLVMAMDGPUSim
  HWModel.cpp
  Simulator.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_COMPONENTS
  Support

  ADD_TO_COMPONENT
  AMDGPU
  )
```

`MIRAdapter.cpp` and `AMDGPUStaticSimulator.cpp` are compiled into
`AMDGPUCodeGen`, which links `AMDGPUSim`.

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

## File Structure

```
AMDGPUSim/
|-- AMDGPUSim.h      # Main public header (includes core headers)
|-- SimInst.h        # Lightweight instruction wrapper + enums
|-- SimInstInfo.h    # Abstract property query interface
|-- HWModel.h        # Hardware model configuration
|-- HWModel.cpp      # GFX1250 hardware model
|-- SimState.h       # Simulation state (GPUSimState)
|-- InstrInfo.h      # Per-instruction result (InstrSimInfo, StallBreakdown, StallReason)
|-- Simulator.h      # Simulator class and SimulatorConfig
|-- Simulator.cpp    # Core simulation logic
|-- MIRAdapter.h     # MachineInstrInfo declaration
|-- MIRAdapter.cpp   # MachineInstrInfo implementation
|-- CMakeLists.txt   # Build configuration
`-- doc/
    `-- Design.md    # This document
```
