# PPE42 Backend Implementation for LLVM

## Overview

This document describes the PPE42 target support implementation within the LLVM PowerPC backend. PPE42 is a 32-bit embedded processor architecture based on a subset of the Power ISA v2.07, featuring an in-order, non-speculative core with limited 64-bit support through Virtual Doubleword Registers (VDR).

## Architecture Summary

- **Base ISA**: Power ISA v2.07 (subset)
- **Word Size**: 32-bit
- **Execution Model**: In-order, non-speculative
- **Branch Prediction**: None
- **Register File**: 16 GPRs (subset of PowerPC's 32)
- **64-bit Support**: Via VDR (Virtual Doubleword Registers)
- **Endianness**: Big-endian only

### Variants
- **PPE42**: Base variant
- **PPE42X**: Adds limited 64-bit features
- **PPE42XM**: Adds multiply high word instructions

## Key Architectural Differences from PPC32

| Feature | PPC32 | PPE42 |
|---------|-------|-------|
| GPR Count | 32 | 16 |
| CR Fields | 8 | 1 (CR0 only) |
| Privilege Levels | Yes | No |
| Speculation | Yes | No |
| Branch Prediction | Yes | No |
| Load/Store Model | Out-of-order capable | Strict in-order |
| 64-bit Operations | No | Partial (via VDR) |
| Stack Operations | Standard | + lsku/stsku |

## Register Model

### General Purpose Registers (GPR)
Available registers: **R0, R1, R2, R3-R10, R13, R28-R31** (16 total)

Missing from standard PowerPC: R11, R12, R14-R27

### Virtual Doubleword Registers (VDR)
VDRs are constructed from pairs of consecutive GPRs for 64-bit operations.

**Format**: `VDR(n) = {GPR(n), GPR(n+1 mod 32)}`

**Available VDRs**:
- VD0 (R0:R1)
- VD2-VD9 (R2:R3 through R9:R10)
- VD28-VD31 (R28:R29 through R31:R0)

**Note**: VD31 wraps around, combining R31 (high 32 bits) with R0 (low 32 bits)

**Assembly Syntax**: `d0`, `d2`-`d9`, `d28`-`d31`

### Condition Register
Only **CR0** exists (4 bits: LT, GT, EQ, SO)

### VDR Implementation Strategy

PPE42's VDR (Virtual Doubleword Register) support is implemented through a combination of register class definitions, pseudo-instructions, and custom lowering.

#### Core Approach

**VDR registers are register pairs** that can only be used for:
1. 64-bit load/store operations (LVD/STVD)
2. 64-bit rotate-insert operations (RLDIMI_VDR)
3. Loading 64-bit immediates (LI8_VDR pseudo-instruction)

**VDR registers CANNOT** perform general arithmetic operations. For 64-bit arithmetic, LLVM uses separate 32-bit GPR operations.

#### Key Implementation Components

##### 1. Register Class Definition
```cpp
if (Subtarget.isPPE42()) {
  addRegisterClass(MVT::i64, &PPC::VDRCRegClass);
  // Tells LLVM: "I have i64 registers (VDR)"
}
```

##### 2. LI8_VDR Pseudo-Instruction
Handles loading 64-bit immediate values into VDR pairs:

```tablegen
let isPseudo = 1, isReMaterializable = 1 in {
def LI8_VDR : PPEVDPseudo<(outs vdrc:$dst), (ins i64imm:$imm),
                          "#LI8_VDR", []>;
}
```

**Expansion** (in PPCInstrInfo.cpp):
```cpp
// Expands to 4 instructions that load immediate into GPR pair:
// LI   rLo, imm_low_16
// LIS  rHi, imm_high_16
// ORI  rHi, rHi, imm_mid_16
// ORIS rLo, rLo, imm_upper_16
```

##### 3. RLDIMI_VDR Instruction
64-bit rotate-left-double-immediate-then-mask-insert for VDR:

```tablegen
let Constraints = "$rA = $rS" in {
def RLDIMI_VDR : MDForm_1<30, 3,
                          (outs vdrc:$rA),
                          (ins vdrc:$rS, vdrc:$rB, u6imm:$SH, u6imm:$MBE),
                          "rldimi $rA, $rB, $SH, $MBE", IIC_IntRotateDI,
                          []>,
                 isPPC64, Requires<[IsPPE42]>;
}
```

**Key Feature**: The `$rA = $rS` constraint implements read-modify-write semantics:
- Input: existing value in `$rS` (same register as output `$rA`)
- Modifies: inserts rotated bits from `$rB` into `$rS`
- Output: result in `$rA` (same physical register)

This constraint ensures the register allocator:
1. Allocates the same physical register for both `$rS` and `$rA`
2. Preserves the input value for the read-modify-write operation
3. Avoids unnecessary COPY instructions

##### 4. Pattern Matching for RLDIMI_VDR

The `tryAsSingleRLDIMI()` function in PPCISelDAGToDAG.cpp detects patterns that can use RLDIMI_VDR:

```cpp
if (Subtarget->isPPE42() && VT == MVT::i64) {
  // For PPE42, use RLDIMI_VDR with 4 operands
  SDValue Ops[] = {Val, Res, getI32Imm(SH, dl), getI32Imm(MB, dl)};
  return CurDAG->getMachineNode(PPC::RLDIMI_VDR, dl, MVT::i64, Ops);
}
```

This matches patterns like:
```c
value = (value & ~mask) | ((source << shift) & mask);
```

And generates:
```assembly
rldimi r4, r6, 63, 0  # Insert rotated r6 into r4
```

#### Why No BUILD_PAIR Support?

**BUILD_PAIR** is an LLVM operation that combines two i32 values into one i64 value. We intentionally **do not** set `setOperationAction(ISD::BUILD_PAIR, MVT::i64, Expand)` because:

1. **Not needed for current use cases**: Our test code operates on complete i64 values (load, modify, store), not combining separate i32 values
2. **Simpler codebase**: Avoiding unnecessary complexity
3. **Learn by doing**: Will add BUILD_PAIR support when we encounter code that actually needs it

**When BUILD_PAIR would be needed**:
```c
// This would require BUILD_PAIR:
uint64_t combine(uint32_t hi, uint32_t lo) {
    return ((uint64_t)hi << 32) | lo;
}
```

When we encounter such code, LLVM will fail with:
```
LLVM ERROR: Cannot select: t6: i64 = BUILD_PAIR t4, t5
```

At that point, we can add the appropriate lowering.

#### Complete Flow Example

**C Code**:
```c
volatile uint64_t *ptr = (uint64_t*)0x50000;
uint64_t value = *ptr;
value |= 0x8000000000000000ULL;  // Set bit 63
*ptr = value;
```

**Generated Assembly**:
```assembly
lis r3, 5                    # Load address high
lvd r4, 0(r3)                # Load 64-bit value into VD4 (R4:R5)
li r7, -1                    # Load immediate -1 into R7
lis r6, -1                   # Load immediate -1 into R6
ori r6, r6, 65535            # Complete 0xFFFFFFFF in R6
oris r7, r7, 65535           # Complete 0xFFFFFFFF in R7
rldimi r4, r6, 63, 0         # Insert bit 63 from R6:R7 into R4:R5
stvd r4, 0(r3)               # Store 64-bit value from VD4
```

**Key Points**:
1. LVD/STVD use VDR register pairs (R4:R5)
2. Immediate loading uses LI8_VDR pseudo (expands to 4 instructions)
3. RLDIMI_VDR performs the bit manipulation on the VDR pair
4. Register constraint ensures R4:R5 is preserved across RLDIMI_VDR

## Implementation Details

### Files Modified/Added

#### New Files
- **PPCInstrPPEVD.td**: VDR instruction definitions (LVD, STVD)

#### Modified Files
- **PPCRegisterInfo.td**: VDR register class definitions
- **PPC.td**: PPE42 processor and feature definitions
- **PPCAsmParser.cpp**: VDR register parsing support
- **PPCDisassembler.cpp**: VDR register decoding support
- **PPCMCTargetDesc.h**: VDR register class mapping
- **PPCTargetParser.def**: PPE42 CPU definition

### Register Class Definitions

#### VDRC (Virtual Doubleword Register Class)
```tablegen
def VDRC : RegisterClass<"PPC", [i64], 64, (add
  VD0, VD2, VD3, VD4, VD5, VD6, VD7, VD8, VD9,
  VD28, VD29, VD30
)>;
```

**Properties**:
- Type: i64 (64-bit integers)
- Alignment: 64-bit
- Used for: 64-bit load/store operations

### Instruction Set

#### VDR Load/Store Instructions

##### LVD - Load Virtual Doubleword
```assembly
lvd dN, offset(rA)
```
- **Opcode**: 5
- **Format**: D-Form
- **Operation**: Loads 64 bits from memory into VDR
- **Encoding**: `000101 | RST | RA | D`

**Example**:
```assembly
lvd d4, 0(r3)      # Load 64 bits from [R3+0] into VD4 (R4:R5)
lvd d28, 100(r1)   # Load 64 bits from [R1+100] into VD28 (R28:R29)
```

##### STVD - Store Virtual Doubleword
```assembly
stvd dN, offset(rA)
```
- **Opcode**: 6
- **Format**: D-Form
- **Operation**: Stores 64 bits from VDR to memory
- **Encoding**: `000110 | RST | RA | D`

**Example**:
```assembly
stvd d4, 0(r3)     # Store VD4 (R4:R5) to [R3+0]
stvd d28, 100(r1)  # Store VD28 (R28:R29) to [R1+100]
```

### Decoder Namespace

PPE42 instructions use a separate decoder namespace to avoid conflicts with other PowerPC variants (particularly Power10's LXVP/STXVP which share opcode 6).

```tablegen
let DecoderNamespace = "PPE42";
```

This creates an isolated instruction decoding table that is checked first when the PPE42 feature is enabled.

## Building and Using

### Target Triple
```
powerpc-unknown-elf
```

### CPU Selection
```bash
clang -target powerpc-unknown-elf -mcpu=ppe42 -c test.c
```

### Feature Flags
```bash
clang -target powerpc-unknown-elf -mattr=+ppe42 -c test.c
```

### Assembly Example
```assembly
# PPE42 assembly with VDR instructions
.text
.globl _start
_start:
    # Load 64-bit value
    lvd d4, 0(r3)
    
    # Store 64-bit value
    stvd d4, 8(r3)
    
    # Standard 32-bit operations
    lwz r5, 0(r3)
    stw r5, 4(r3)
    
    blr
```

## Calling Convention

Based on PowerPC EABI (32-bit) with adjustments for reduced register set:

- **Stack Pointer**: R1
- **Argument Registers**: R3-R10 (8 registers)
- **Return Value**: R3
- **Callee-Saved**: R28-R31, R1, R2
- **Caller-Saved**: R3-R10

**Note**: Reduced register count increases register pressure compared to standard PowerPC.

## Limitations and Restrictions

### Not Implemented
The following PowerPC features are **not available** in PPE42:

- System call instruction (`sc`)
- Privilege model instructions
- Floating-point operations
- Vector/SIMD instructions
- String/multiple load/store
- Divide instructions
- CR logical instructions
- `isync`, `eieio` (use `sync` instead)
- Instruction cache operations

### Memory Model
- 32-bit flat address space
- Big-endian only
- Byte-addressable
- Alignment enforced by memory system
- No MMU at architecture level

### Execution Model
- Strict in-order execution
- No speculation
- No branch prediction
- Loads are always blocking and precise
- Stores can be imprecise (controlled by MSR[IPE])

## Testing

### Assembly Test
```bash
# Assemble PPE42 code
llvm-mc -triple=powerpc-unknown-elf -mcpu=ppe42 \
        -filetype=obj test.s -o test.o

# Disassemble
llvm-objdump -d --mcpu=ppe42 test.o
```

### Compiler Test
```bash
# Compile C code for PPE42
clang -target powerpc-unknown-elf -mcpu=ppe42 \
      -c test.c -o test.o

# View assembly
clang -target powerpc-unknown-elf -mcpu=ppe42 \
      -S test.c -o test.s
```

### Example C Code
```c
// test.c - PPE42 64-bit operations via VDR
#include <stdint.h>

void test_vdr(uint64_t *ptr) {
    // Compiler should generate LVD/STVD for 64-bit operations
    uint64_t value = *ptr;
    value += 1;
    *ptr = value;
}
```

**Note**: Direct VDR usage typically requires inline assembly. The compiler may not automatically generate LVD/STVD for all 64-bit operations.

## Inline Assembly

### Using VDR Registers
```c
void inline_asm_example(void) {
    uint64_t value;
    
    // Load 64 bits using LVD
    __asm__ volatile (
        "lvd d4, 0(%1)\n\t"
        "stvd d4, 0(%0)"
        : "=r" (&value)
        : "r" (some_address)
        : "d4"
    );
}
```

### Register Constraints
- `"r"`: GPR registers (R0-R31, but only 16 available)
- VDR registers must be specified explicitly (e.g., `"d4"`)

## Debugging

### Disassembly
```bash
# Disassemble with PPE42 decoder
llvm-objdump -d --mcpu=ppe42 binary.o

# Expected output for LVD:
# 0: 14 83 00 00   lvd d4, 0(r3)
```

### Verification
```bash
# Verify instruction encoding
echo "lvd d4, 0(r3)" | llvm-mc -triple=powerpc-unknown-elf \
     -mcpu=ppe42 -show-encoding

# Expected: encoding: [0x14,0x83,0x00,0x00]
```

## Performance Considerations

### Register Pressure
With only 16 GPRs available, register allocation is more constrained than standard PowerPC:
- More spills to stack likely
- Careful register usage in hot paths
- Consider using VDRs for 64-bit data to reduce pressure

### Pipeline Characteristics
| Operation | Latency |
|-----------|---------|
| ALU | 1 cycle (pipelined) |
| Branch | 2 cycles |
| Fused compare-branch | 3 cycles |
| Load/Store | 2 + memory latency |

### Optimization Tips
1. **Minimize 64-bit operations**: Use VDR only when necessary
2. **Reduce register pressure**: Careful variable scoping
3. **Avoid branches**: In-order execution makes branches expensive
4. **Use fused compare-branch**: When available (PPE42 specific)

## Code Generation Improvements

This section tracks code-generation changes made to the LLVM PPE42 backend as
part of ongoing firmware size optimisation work.  The baseline for comparisons
is the GCC PPE42 toolchain (`powerpc-eabi-gcc`) used in previous firmware
builds.

---

### [Done] Sibling-Call Optimisation (SCO) for 32-bit SVR4 / PPE42

**Files**: `llvm/lib/Target/PowerPC/PPCISelLowering.cpp`
**Functions**: `IsEligibleForTailCallOptimization`, `LowerCall_32SVR4`
**Test**: `llvm/test/CodeGen/PowerPC/ppe42-tailcall-sco.ll`

Two separate gaps in the 32-bit SVR4 lowering were fixed, both required to
produce clean frameless SCO code matching GCC output.

---

#### Part 1 — SCO eligibility gate (`IsEligibleForTailCallOptimization`)

**Problem**

The 32-bit SVR4 code path returned `false` from
`IsEligibleForTailCallOptimization` unless `-tailcallopt` /
`GuaranteedTailCallOpt` was explicitly set.  This meant every tail call —
including trivially optimisable wrapper functions like:

```cpp
sbeReturnCode* getCfamRegister(...) {
    return sbeReturnCode::errorl(SBE_PRI_USER_ERROR,
                                 SBE_SEC_FUNCTIONALITY_NOT_SUPPORTED);
}
```

was compiled as a **full non-leaf call** with `mflr`/`stwu`/`bl`/`lwz`/
`mtlr`/`blr`, adding 32 bytes per stub instead of the 4-byte `b` GCC emits.

**Root cause**

The 64-bit SVR4 path (`IsEligibleForTailCallOptimization_64SVR4`) has a full
SCO implementation; the 32-bit path had none — a historical gap.  The
TOC-sharing constraints that make 64-bit SCO complicated do not apply to PPE42
(PPE42 has no TOC register).

**Fix**

Extended `IsEligibleForTailCallOptimization` to add an SCO path for the 32-bit
SVR4 / PPE42 case:

- Respects `-disable-ppc-sco` (existing flag, now shared with the 32-bit path).
- Allows SCO for `CallingConv::C` and `CallingConv::Fast` when caller and callee
  use the same CC.
- Guards on: no varargs, no byval parameters, outgoing args that would require
  a new stack slot (GPRsUsed > 8) block SCO.
- Blocks SCO for indirect calls (function pointers) and unresolved external
  symbols — see **Known limitation** below.
- The existing `GuaranteedTailCallOpt` + `fastcc` path is preserved unchanged.
- Added `isCalleeExternalSymbol` parameter (matching the 64-bit interface) and
  threaded it through `isEligibleForTCO`.

---

#### Part 2 — Frameless emission (`LowerCall_32SVR4`)

**Problem**

Even after Part 1 allowed SCO, `LowerCall_32SVR4` still emitted a spurious
stack frame for every SCO call:

```asm
stwu  1, -16(1)       ; ❌ unnecessary stack frame
lwz   3, 20(1)        ; ❌ reload from caller's incoming stack slot
li    4, 9
li    5, -1
stw   3, 28(1)        ; ❌ spill to outgoing arg stack slot
li    3, 3
addi  1, 1, 24        ; ❌ partial SP restore
b     errorl          ; ✅ correct tail branch
```

**Root cause**

`LowerCall_32SVR4` was missing the `IsSibCall` flag that `LowerCall_64SVR4`
uses to bypass `CALLSEQ_START` for sibling calls.  Without it, the 8-byte SVR4
linkage-area reservation always set `MFI.adjustsStack() = true`, preventing
the zero-frame path and forcing `stwu` / `lwz` / `stw` / `addi`.  The ABI is
correct — the bug was purely a missing code path in the 32-bit lowering.

**Fix**

Added `IsSibCall` to `LowerCall_32SVR4`, mirroring the 64-bit path:

```cpp
bool IsSibCall =
    IsTailCall && !getTargetMachine().Options.GuaranteedTailCallOpt;

int SPDiff = 0;
if (!IsSibCall)
    SPDiff = CalculateTailCallSPDiff(DAG, IsTailCall, NumBytes);

if (!IsSibCall)
    Chain = DAG.getCALLSEQ_START(Chain, NumBytes, 0, dl);
...
if (IsTailCall && !IsSibCall)
    PrepareTailCall(...);
```

With `IsSibCall = true`, `CALLSEQ_START` is never issued, `MFI.adjustsStack()`
stays `false`, and the prologue/epilog pass produces a frameless function.
For callers forwarding stack-spilled arguments, `SPDiff = 0` means
`CalculateTailCallArgDest` places the `FixedObject` at the same incoming offset
— no copy needed.  Callers that produce new values for stack-spilled args are
correctly rejected by Part 1 (GPRsUsed > 8).

---

#### Known limitation — indirect tail calls not SCO'd

PPE42 has PC-relative branch instructions (`b` / `bl`) but the 32-bit SVR4
lowering path does not implement indirect tail calls via CTR (`bctr`).  The
64-bit path supports indirect SCO by loading the function pointer into the
Count Register and emitting `bctr`; that infrastructure does not exist in
`LowerCall_32SVR4`.

> **Note on terminology**: LLVM's `isUsingPCRelativeCalls()` predicate refers
> specifically to the ELFv2 PC-relative GOT-free addressing mode (introduced
> for POWER10, enabled with `-mpcrel`), **not** to the `b`/`bl` instructions
> themselves.  PPE42's `b` and `bl` are PC-relative branches, but PPE42 does
> not use the ELFv2 PCRel mode.

As a result, tail calls through function pointers are not SCO'd and fall back
to a normal `bl` + `blr` sequence.  This is correct and safe; it is a
missed-optimisation rather than a correctness issue.  A future enhancement
could implement CTR-based indirect tail calls for the 32-bit path.

---

#### Combined outcome

`getCfamRegister` / `putCfamRegister` now compile to:

```asm
li  r3, 3
li  r4, 9
li  r5, -1
b   errorl
```

No `stwu`, no `stw`/`lwz` spills, no `addi` SP restore — matching GCC exactly.
Overall LLVM vs GCC `.text` size overhead improved from **+81.4% → +36.4%**
across the SBE firmware object files.

---

### [TODO] Fix over-wide `.text` section alignment for PPE42

**File**: `llvm/lib/Target/PowerPC/PPCAsmPrinter.cpp`
**Function**: `PPCAsmPrinter::runOnMachineFunction`
**Priority**: Low — padding exists only in `.o` files; no runtime impact

#### Problem

PPE42 `.text.*` function sections are emitted with `2**3` (8-byte) alignment.
PPE42 is an in-order 32-bit core — there is no instruction-fetch alignment
benefit beyond 4 bytes.  The extra alignment produces up to 4 bytes of
`00 00 00 00` padding after the final instruction in short functions,
visible in `objdump` output of `.o` files (e.g. `plat_hw_access.o`).

> **Note — `.o` file only, not a runtime problem.**  The `00 00 00 00` word
> is ELF section-alignment padding inserted by the object writer to satisfy
> the section's `sh_addralign` field.  It is **not** a generated instruction
> and is **not present in the final linked firmware image**: the linker strips
> inter-section padding when it merges `.text.*` sections during final layout.
> This issue has zero impact on runtime code size or behaviour.

#### Root Cause

`VDRC` (Virtual Doubleword Register class) is correctly defined with 64-bit
register alignment in `PPCRegisterInfo.td` (a VDR is a 64-bit value spanning
two 32-bit GPRs).  When any VDR spill slot is created, `MachineFrameInfo`
calls `ensureMaxAlignment(Align(8))`, setting `MFI.MaxAlign = 8`.
`AsmPrinter::emitFunctionHeader` then emits `.align 3` which bumps the ELF
section `sh_addralign` to 8 — even for functions that never actually spill a
VDR.  Changing the `VDRC` class alignment would be architecturally wrong.

#### Fix needed

In `PPCAsmPrinter::runOnMachineFunction`, cap `MF.setAlignment()` to
`Align(4)` for PPE42 **before** the parent `AsmPrinter::runOnMachineFunction`
runs `emitFunctionHeader`:

```cpp
if (Subtarget->isPPE42())
  MF.setAlignment(std::min(MF.getAlignment(), Align(4)));
```

This caps only the `.align` directive emitted for the function entry; it does
not affect actual VDR stack slot alignment, which remains 8-byte for
correctness.

**Expected outcome**: `sh_addralign` set to 4 on all PPE42 function sections;
no trailing `00 00 00 00` padding in `.o` objdump output; minor reduction in
`.o` file sizes.  No change to the final linked image.

---

## Future Enhancements

### Potential Additions
- [ ] Fused compare-branch instruction support
- [ ] Stack frame instructions (lsku/stsku)
- [ ] PPE42X 64-bit rotate/shift instructions
- [ ] PPE42XM multiply high word support
- [ ] Instruction scheduling model
- [ ] Compiler optimization passes for reduced register set

### Known Issues
- VDR usage in compiler-generated code is limited
- May require explicit inline assembly for optimal VDR usage
- Register allocator not yet optimized for 16-register constraint

## References

- **PPE42 ISA Specification**: See `PPE42_ISA.md` in project root
- **PowerPC EABI**: Embedded Application Binary Interface
- **LLVM PowerPC Backend**: `llvm/lib/Target/PowerPC/`

## Contact and Support

For issues or questions regarding the PPE42 backend implementation:
- Check existing LLVM PowerPC backend documentation
- Review PPE42 ISA specification
- Examine test cases in `appsource/` directory

## Version History

- **v1.2** (2026-05-31): Refined VDR implementation
  - Removed BUILD_PAIR support (not needed for current use cases)
  - Added RLDIMI_VDR instruction with register constraints
  - Implemented LI8_VDR pseudo-instruction for 64-bit immediate loading
  - Added read-modify-write semantics via register constraints
  - Optimized register allocation for VDR operations

- **v1.1** (2026-05-16): Enhanced VDR support
  - Added RLDIMI_VDR for 64-bit rotate-insert operations
  - Implemented pattern matching for bit manipulation
  - Added support for 64-bit immediate values in VDR

- **v1.0** (2026-05): Initial PPE42 backend implementation
  - VDR register class support
  - LVD/STVD instruction support
  - Assembly parser and disassembler support
  - Decoder namespace isolation

---

**Last Updated**: May 31, 2026
**LLVM Version**: 19.0.0 (development)
**Status**: Functional - VDR load/store, rotate-insert, and immediate loading supported
