# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s

# Instructions specific to the PowerPC 6xx family:

# CHECK-BE: mfibatu 12, 0                    # encoding: [0x7d,0x90,0x82,0xa6]
# CHECK-LE: mfibatu 12, 0                    # encoding: [0xa6,0x82,0x90,0x7d]
mfibatu %r12, 0
# CHECK-BE: mfibatl 12, 0                    # encoding: [0x7d,0x91,0x82,0xa6]
# CHECK-LE: mfibatl 12, 0                    # encoding: [0xa6,0x82,0x91,0x7d]
mfibatl %r12, 0
# CHECK-BE: mfibatu 12, 1                    # encoding: [0x7d,0x92,0x82,0xa6]
# CHECK-LE: mfibatu 12, 1                    # encoding: [0xa6,0x82,0x92,0x7d]
mfibatu %r12, 1
# CHECK-BE: mfibatl 12, 1                    # encoding: [0x7d,0x93,0x82,0xa6]
# CHECK-LE: mfibatl 12, 1                    # encoding: [0xa6,0x82,0x93,0x7d]
mfibatl %r12, 1
# CHECK-BE: mfibatu 12, 2                    # encoding: [0x7d,0x94,0x82,0xa6]
# CHECK-LE: mfibatu 12, 2                    # encoding: [0xa6,0x82,0x94,0x7d]
mfibatu %r12, 2
# CHECK-BE: mfibatl 12, 2                    # encoding: [0x7d,0x95,0x82,0xa6]
# CHECK-LE: mfibatl 12, 2                    # encoding: [0xa6,0x82,0x95,0x7d]
mfibatl %r12, 2
# CHECK-BE: mfibatu 12, 3                    # encoding: [0x7d,0x96,0x82,0xa6]
# CHECK-LE: mfibatu 12, 3                    # encoding: [0xa6,0x82,0x96,0x7d]
mfibatu %r12, 3
# CHECK-BE: mfibatl 12, 3                    # encoding: [0x7d,0x97,0x82,0xa6]
# CHECK-LE: mfibatl 12, 3                    # encoding: [0xa6,0x82,0x97,0x7d]
mfibatl %r12, 3
# CHECK-BE: mtibatu 0, 12                    # encoding: [0x7d,0x90,0x83,0xa6]
# CHECK-LE: mtibatu 0, 12                    # encoding: [0xa6,0x83,0x90,0x7d]
mtibatu 0, %r12
# CHECK-BE: mtibatl 0, 12                    # encoding: [0x7d,0x91,0x83,0xa6]
# CHECK-LE: mtibatl 0, 12                    # encoding: [0xa6,0x83,0x91,0x7d]
mtibatl 0, %r12
# CHECK-BE: mtibatu 1, 12                    # encoding: [0x7d,0x92,0x83,0xa6]
# CHECK-LE: mtibatu 1, 12                    # encoding: [0xa6,0x83,0x92,0x7d]
mtibatu 1, %r12
# CHECK-BE: mtibatl 1, 12                    # encoding: [0x7d,0x93,0x83,0xa6]
# CHECK-LE: mtibatl 1, 12                    # encoding: [0xa6,0x83,0x93,0x7d]
mtibatl 1, %r12
# CHECK-BE: mtibatu 2, 12                    # encoding: [0x7d,0x94,0x83,0xa6]
# CHECK-LE: mtibatu 2, 12                    # encoding: [0xa6,0x83,0x94,0x7d]
mtibatu 2, %r12
# CHECK-BE: mtibatl 2, 12                    # encoding: [0x7d,0x95,0x83,0xa6]
# CHECK-LE: mtibatl 2, 12                    # encoding: [0xa6,0x83,0x95,0x7d]
mtibatl 2, %r12
# CHECK-BE: mtibatu 3, 12                    # encoding: [0x7d,0x96,0x83,0xa6]
# CHECK-LE: mtibatu 3, 12                    # encoding: [0xa6,0x83,0x96,0x7d]
mtibatu 3, %r12
# CHECK-BE: mtibatl 3, 12                    # encoding: [0x7d,0x97,0x83,0xa6]
# CHECK-LE: mtibatl 3, 12                    # encoding: [0xa6,0x83,0x97,0x7d]
mtibatl 3, %r12

# CHECK-BE: mfdbatu 12, 0                    # encoding: [0x7d,0x98,0x82,0xa6]
# CHECK-LE: mfdbatu 12, 0                    # encoding: [0xa6,0x82,0x98,0x7d]
mfdbatu %r12, 0
# CHECK-BE: mfdbatl 12, 0                    # encoding: [0x7d,0x99,0x82,0xa6]
# CHECK-LE: mfdbatl 12, 0                    # encoding: [0xa6,0x82,0x99,0x7d]
mfdbatl %r12, 0
# CHECK-BE: mfdbatu 12, 1                    # encoding: [0x7d,0x9a,0x82,0xa6]
# CHECK-LE: mfdbatu 12, 1                    # encoding: [0xa6,0x82,0x9a,0x7d]
mfdbatu %r12, 1
# CHECK-BE: mfdbatl 12, 1                    # encoding: [0x7d,0x9b,0x82,0xa6]
# CHECK-LE: mfdbatl 12, 1                    # encoding: [0xa6,0x82,0x9b,0x7d]
mfdbatl %r12, 1
# CHECK-BE: mfdbatu 12, 2                    # encoding: [0x7d,0x9c,0x82,0xa6]
# CHECK-LE: mfdbatu 12, 2                    # encoding: [0xa6,0x82,0x9c,0x7d]
mfdbatu %r12, 2
# CHECK-BE: mfdbatl 12, 2                    # encoding: [0x7d,0x9d,0x82,0xa6]
# CHECK-LE: mfdbatl 12, 2                    # encoding: [0xa6,0x82,0x9d,0x7d]
mfdbatl %r12, 2
# CHECK-BE: mfdbatu 12, 3                    # encoding: [0x7d,0x9e,0x82,0xa6]
# CHECK-LE: mfdbatu 12, 3                    # encoding: [0xa6,0x82,0x9e,0x7d]
mfdbatu %r12, 3
# CHECK-BE: mfdbatl 12, 3                    # encoding: [0x7d,0x9f,0x82,0xa6]
# CHECK-LE: mfdbatl 12, 3                    # encoding: [0xa6,0x82,0x9f,0x7d]
mfdbatl %r12, 3
# CHECK-BE: mtdbatu 0, 12                    # encoding: [0x7d,0x98,0x83,0xa6]
# CHECK-LE: mtdbatu 0, 12                    # encoding: [0xa6,0x83,0x98,0x7d]
mtdbatu 0, %r12
# CHECK-BE: mtdbatl 0, 12                    # encoding: [0x7d,0x99,0x83,0xa6]
# CHECK-LE: mtdbatl 0, 12                    # encoding: [0xa6,0x83,0x99,0x7d]
mtdbatl 0, %r12
# CHECK-BE: mtdbatu 1, 12                    # encoding: [0x7d,0x9a,0x83,0xa6]
# CHECK-LE: mtdbatu 1, 12                    # encoding: [0xa6,0x83,0x9a,0x7d]
mtdbatu 1, %r12
# CHECK-BE: mtdbatl 1, 12                    # encoding: [0x7d,0x9b,0x83,0xa6]
# CHECK-LE: mtdbatl 1, 12                    # encoding: [0xa6,0x83,0x9b,0x7d]
mtdbatl 1, %r12
# CHECK-BE: mtdbatu 2, 12                    # encoding: [0x7d,0x9c,0x83,0xa6]
# CHECK-LE: mtdbatu 2, 12                    # encoding: [0xa6,0x83,0x9c,0x7d]
mtdbatu 2, %r12
# CHECK-BE: mtdbatl 2, 12                    # encoding: [0x7d,0x9d,0x83,0xa6]
# CHECK-LE: mtdbatl 2, 12                    # encoding: [0xa6,0x83,0x9d,0x7d]
mtdbatl 2, %r12
# CHECK-BE: mtdbatu 3, 12                    # encoding: [0x7d,0x9e,0x83,0xa6]
# CHECK-LE: mtdbatu 3, 12                    # encoding: [0xa6,0x83,0x9e,0x7d]
mtdbatu 3, %r12
# CHECK-BE: mtdbatl 3, 12                    # encoding: [0x7d,0x9f,0x83,0xa6]
# CHECK-LE: mtdbatl 3, 12                    # encoding: [0xa6,0x83,0x9f,0x7d]
mtdbatl 3, %r12

# CHECK-BE: tlbld 4                        # encoding: [0x7c,0x00,0x27,0xa4]
# CHECK-LE: tlbld 4                        # encoding: [0xa4,0x27,0x00,0x7c]
tlbld %r4
# CHECK-BE: tlbli 4                        # encoding: [0x7c,0x00,0x27,0xe4]
# CHECK-LE: tlbli 4                        # encoding: [0xe4,0x27,0x00,0x7c]
tlbli %r4
