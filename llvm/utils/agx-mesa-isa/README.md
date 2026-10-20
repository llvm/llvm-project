# Mesa's AGX instruction forms

`mesa_isa.py` describes every instruction form Mesa's asahi compiler emits for
G13 (M1) and G14 (M2): its packer, `src/asahi/compiler/agx_pack.c`, together
with the operand values its compiler creates. It is the reference for the AGX
backend's `mesa-isa` feature (llvm/docs/AGXUsage.rst, "Mesa ISA mode") and for
the AGX Performance Simulator's `--mesa-isa` option.

```
mesa_isa.py --cpp ../../lib/Target/AGX/MCTargetDesc/AGXMesaISA.inc   # backend table
mesa_isa.py --rust .../AGX-Performance-Simulator/rust/src/isa/mesa_table.rs
mesa_isa.py --json mesa-isa.json                                     # the forms, readable
mesa_isa.py --gen g14 --check 3800 0500000d00c01200                  # check instructions (hex bytes)
tools/fuzz.sh MESA_SOURCE_DIR [COUNT [SEED [CORPUS_OUT]]]            # compare with Mesa's packer
```

A form is one opcode at one length. Its fields carry operands and immediates,
each checked by a rule that decodes the operand as Mesa packs it (ALU sources,
memory registers, threadgroup bases ...); every other bit must equal the form's
fixed value. A long form is accepted only when the short one cannot hold the
instruction, since Mesa always picks the short length when it can. Generations
are `g13`, `g13x` (multi-cluster M1), `g14` and `g14d` (M2 Ultra): Mesa's output
differs between them only in bit 45 of device atomics.

`tools/fuzz.c` builds random instructions the way Mesa's compiler builds them
(operand sizes, masks, formats, atomic operations, modes), packs them with Mesa's
own `agx_pack.c` and checks that the reference engine accepts each one as the
right opcode. The engines in the backend (`AGXMesaISA.cpp`) and the simulator
(`mesa.rs`) follow the reference engine; the simulator's tests compare them on
mutated instructions.

Mesa is used read-only, as documentation of what a production driver emits;
nothing here is copied from it into the backend.
