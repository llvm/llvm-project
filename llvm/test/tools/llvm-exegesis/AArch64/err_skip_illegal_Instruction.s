REQUIRES: aarch64-registered-target

## Check for skipping of illegal instruction errors (AUT and LDGM)
RUN: llvm-exegesis -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%d --min-instructions=1 --opcode-name=AUTIA --benchmark-phase=assemble-measured-code 2>&1
CHECK: AUTIA: Unsupported opcode: isPointerAuth/isUncheckedAccess

RUN: llvm-exegesis -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=%d --min-instructions=1 --opcode-name=LDGM --benchmark-phase=assemble-measured-code 2>&1
CHECK: LDGM: Unsupported opcode: isPointerAuth/isUncheckedAccess