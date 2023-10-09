# RUN: llvm-mc -triple=mips -mcpu=mips1 < %s | FileCheck --check-prefixes=ALL,MIPS32 %s
# RUN: llvm-mc -triple=mips -mcpu=mips64 < %s | FileCheck --check-prefixes=ALL,MIPS64 %s

# ALL: .text
seq $2, $11, $0
# ALL: sltiu $2, $11, 1
seq $2, $0, $11
# ALL: sltiu $2, $11, 1
seq $2, $0, $0
# ALL: sltiu $2, $zero, 1
seq $2, $11, $12
# ALL: xor $2, $11, $12
# ALL: sltiu $2, $2, 1
seq $2, $11, 45
# ALL: xori $2, $11, 45
seq $2, $12, 0x76666
# ALL: lui $1, 7
# ALL: ori $1, $1, 26214
# ALL: xor $2, $12, $1
# ALL: sltiu $2, $2, 1
seq $2, $3
# ALL: xor $2, $2, $3
# ALL: sltiu $2, $2, 1
seq $2, 0x8888
# ALL: xori $2, $2, 34952
# ALL: sltiu $2, $2, 1
seq $2, $3, -1546
# MIPS32: addiu $2, $3, 1546
# MIPS64: daddiu $2, $3, 1546
# ALL: sltiu $2, $2, 1
seq $2, -7546
# MIPS32: addiu $2, $2, 7546
# MIPS64: daddiu $2, $2, 7546
# ALL: sltiu $2, $2, 1
seq $4, $5, -66666
# ALL: lui $1, 65534
# ALL: ori $1, $1, 64406
# ALL: xor $4, $5, $1
# ALL: sltiu $4, $4, 1
seq $4, $5, -2147483648
# ALL: lui $1, 32768
# ALL: xor $4, $5, $1
# ALL: sltiu $4, $4, 1
seq $4, -2147483648
# ALL: lui $1, 32768
# ALL: xor $4, $4, $1
# ALL: sltiu $4, $4, 1
seq $4, $5, 0
# ALL: sltiu $4, $5, 1
seq $4, $zero, 1
# MIPS32: move $4, $zero
# MIPS64: daddu $4, $zero, $zero
