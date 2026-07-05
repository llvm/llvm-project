# RUN: not llvm-mc %s -triple=mips-unknown-unknown -mcpu=mips32 2>%t1
# RUN: FileCheck %s < %t1

.set at~
# CHECK: error: unexpected token, expected equals sign

.set at=
# CHECK: error: no register specified

.set at=~
# CHECK: error: unexpected token, expected dollar sign '$'

.set at=$
# CHECK: error: unexpected token, expected identifier or integer

.set at=$-4
# CHECK: error: unexpected token, expected identifier or integer

.set at=$1000
# CHECK: error: invalid register

.set at=$foo
# CHECK: error: invalid register

.set at=$2bar
# CHECK: error: unexpected token, expected end of statement

.set noat bar
# CHECK: error: unexpected token, expected end of statement
