; This test is designed to run 4 times, once with function attribute +c,
; once with function attribute -c for eq/ne in icmp
; The optimization should appear only with +c, otherwise default isel should be
; choosen.
;
; RUN: cat %s | sed 's/CMPCOND/eq/g' | sed 's/RESBRNORMAL/bne/g' | \
; RUN: sed 's/RESBROPT/c.bnez/g' > %t.compress_eq
; RUN: llc -mtriple=riscv32 -target-abi ilp32d -mattr=+c,+f,+d -filetype=obj \
; RUN:   -disable-block-placement < %t.compress_eq \
; RUN:   | llvm-objdump -d --triple=riscv32 --mattr=+c,+f,+d -M no-aliases - \
; RUN:   | FileCheck -check-prefix=RV32IFDC %t.compress_eq
;
; RUN: cat %s | sed -e 's/CMPCOND/eq/g' | sed -e 's/RESBRNORMAL/bne/g'\
; RUN:   | sed -e 's/RESBROPT/c.bnez/g' > %t.nocompr_eq
; RUN: llc -mtriple=riscv32 -target-abi ilp32d -mattr=-c,+f,+d -filetype=obj \
; RUN:   -disable-block-placement < %t.nocompr_eq \
; RUN:   | llvm-objdump -d --triple=riscv32 --mattr=-c,+f,+d -M no-aliases - \
; RUN:   | FileCheck -check-prefix=RV32IFD %t.nocompr_eq
;
; RUN: cat %s | sed 's/CMPCOND/ne/g' | sed 's/RESBRNORMAL/beq/g' | \
; RUN: sed 's/RESBROPT/c.beqz/g' > %t.compress_neq
; RUN: llc -mtriple=riscv32 -target-abi ilp32d -mattr=+c,+f,+d -filetype=obj \
; RUN:   -disable-block-placement < %t.compress_neq \
; RUN:   | llvm-objdump -d --triple=riscv32 --mattr=+c,+f,+d -M no-aliases - \
; RUN:   | FileCheck -check-prefix=RV32IFDC %t.compress_neq
;
; RUN: cat %s | sed -e 's/CMPCOND/ne/g' | sed -e 's/RESBRNORMAL/beq/g'\
; RUN:   | sed -e 's/RESBROPT/c.beqz/g' > %t.nocompr_neq
; RUN: llc -mtriple=riscv32 -target-abi ilp32d -mattr=-c,+f,+d -filetype=obj \
; RUN:   -disable-block-placement < %t.nocompr_neq \
; RUN:   | llvm-objdump -d --triple=riscv32 --mattr=-c,+f,+d -M no-aliases - \
; RUN:   | FileCheck -check-prefix=RV32IFD %t.nocompr_neq


; constant is small and fit in 6 bit (compress imm)
; RV32IFDC-LABEL: <f_small_pos>:
; RV32IFDC: c.li [[REG:.*]], 20
; RV32IFDC: RESBRNORMAL [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <f_small_pos>:
; RV32IFD: addi [[REG:.*]], zero, 20
; RV32IFD: RESBRNORMAL [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @f_small_pos(i32 %in0) minsize {
  %cmp = icmp CMPCOND i32 %in0, 20
  br i1 %cmp, label %if.then, label %if.else
if.then:
  %call = shl i32 %in0, 1
  br label %if.end
if.else:
  %call2 = add i32 %in0, 42
  br label %if.end

if.end:
  %toRet = phi i32 [ %call, %if.then ], [ %call2, %if.else ]
  ret i32 %toRet
}

; constant is small and fit in 6 bit (compress imm)
; RV32IFDC-LABEL: <f_small_neg>:
; RV32IFDC: c.li [[REG:.*]], -20
; RV32IFDC: RESBRNORMAL [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <f_small_neg>:
; RV32IFD: addi [[REG:.*]], zero, -20
; RV32IFD: RESBRNORMAL [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @f_small_neg(i32 %in0) minsize {
  %cmp = icmp CMPCOND i32 %in0, -20
  br i1 %cmp, label %if.then, label %if.else
if.then:
  %call = shl i32 %in0, 1
  br label %if.end
if.else:
  %call2 = add i32 %in0, 42
  br label %if.end

if.end:
  %toRet = phi i32 [ %call, %if.then ], [ %call2, %if.else ]
  ret i32 %toRet
}

; constant is small and fit in 6 bit (compress imm)
; RV32IFDC-LABEL: <f_small_edge_pos>:
; RV32IFDC: c.li [[REG:.*]], 31
; RV32IFDC: RESBRNORMAL [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <f_small_edge_pos>:
; RV32IFD: addi [[REG:.*]], zero, 31
; RV32IFD: RESBRNORMAL [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @f_small_edge_pos(i32 %in0) minsize {
  %cmp = icmp CMPCOND i32 %in0, 31
  br i1 %cmp, label %if.then, label %if.else
if.then:
  %call = shl i32 %in0, 1
  br label %if.end
if.else:
  %call2 = add i32 %in0, 42
  br label %if.end

if.end:
  %toRet = phi i32 [ %call, %if.then ], [ %call2, %if.else ]
  ret i32 %toRet
}

; constant is small and fit in 6 bit (compress imm)
; RV32IFDC-LABEL: <f_small_edge_neg>:
; RV32IFDC: c.li [[REG:.*]], -32
; RV32IFDC: RESBRNORMAL [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <f_small_edge_neg>:
; RV32IFD: addi [[REG:.*]], zero, -32
; RV32IFD: RESBRNORMAL [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @f_small_edge_neg(i32 %in0) minsize {
  %cmp = icmp CMPCOND i32 %in0, -32
  br i1 %cmp, label %if.then, label %if.else
if.then:
  %call = shl i32 %in0, 1
  br label %if.end
if.else:
  %call2 = add i32 %in0, 42
  br label %if.end

if.end:
  %toRet = phi i32 [ %call, %if.then ], [ %call2, %if.else ]
  ret i32 %toRet
}

; constant is medium and not fit in 6 bit (compress imm),
; but fit in 12 bit (imm)
; RV32IFDC-LABEL: <f_medium_ledge_pos>:
; RV32IFDC: addi [[MAYZEROREG:.*]], [[REG:.*]], -32
; RV32IFDC: RESBROPT [[MAYZEROREG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <f_medium_ledge_pos>:
; RV32IFD: addi [[REG:.*]], zero, 32
; RV32IFD: RESBRNORMAL [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @f_medium_ledge_pos(i32 %in0) minsize {
  %cmp = icmp CMPCOND i32 %in0, 32
  br i1 %cmp, label %if.then, label %if.else
if.then:
  %call = shl i32 %in0, 1
  br label %if.end
if.else:
  %call2 = add i32 %in0, 42
  br label %if.end

if.end:
  %toRet = phi i32 [ %call, %if.then ], [ %call2, %if.else ]
  ret i32 %toRet
}

; constant is medium and not fit in 6 bit (compress imm),
; but fit in 12 bit (imm)
; RV32IFDC-LABEL: <f_medium_ledge_neg>:
; RV32IFDC: addi [[MAYZEROREG:.*]], [[REG:.*]], 33
; RV32IFDC: RESBROPT [[MAYZEROREG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <f_medium_ledge_neg>:
; RV32IFD: addi [[REG:.*]], zero, -33
; RV32IFD: RESBRNORMAL [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @f_medium_ledge_neg(i32 %in0) minsize {
  %cmp = icmp CMPCOND i32 %in0, -33
  br i1 %cmp, label %if.then, label %if.else
if.then:
  %call = shl i32 %in0, 1
  br label %if.end
if.else:
  %call2 = add i32 %in0, 42
  br label %if.end

if.end:
  %toRet = phi i32 [ %call, %if.then ], [ %call2, %if.else ]
  ret i32 %toRet
}

; constant is medium and not fit in 6 bit (compress imm),
; but fit in 12 bit (imm)
; RV32IFDC-LABEL: <f_medium_pos>:
; RV32IFDC: addi [[MAYZEROREG:.*]], [[REG:.*]], -63
; RV32IFDC: RESBROPT [[MAYZEROREG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <f_medium_pos>:
; RV32IFD: addi [[REG:.*]], zero, 63
; RV32IFD: RESBRNORMAL [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @f_medium_pos(i32 %in0) minsize {
  %cmp = icmp CMPCOND i32 %in0, 63
  br i1 %cmp, label %if.then, label %if.else
if.then:
  %call = shl i32 %in0, 1
  br label %if.end
if.else:
  %call2 = add i32 %in0, 42
  br label %if.end

if.end:
  %toRet = phi i32 [ %call, %if.then ], [ %call2, %if.else ]
  ret i32 %toRet
}

; constant is medium and not fit in 6 bit (compress imm),
; but fit in 12 bit (imm)
; RV32IFDC-LABEL: <f_medium_neg>:
; RV32IFDC: addi [[MAYZEROREG:.*]], [[REG:.*]], 63
; RV32IFDC: RESBROPT [[MAYZEROREG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <f_medium_neg>:
; RV32IFD: addi [[REG:.*]], zero, -63
; RV32IFD: RESBRNORMAL [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @f_medium_neg(i32 %in0) minsize {
  %cmp = icmp CMPCOND i32 %in0, -63
  br i1 %cmp, label %if.then, label %if.else
if.then:
  %call = shl i32 %in0, 1
  br label %if.end
if.else:
  %call2 = add i32 %in0, 42
  br label %if.end

if.end:
  %toRet = phi i32 [ %call, %if.then ], [ %call2, %if.else ]
  ret i32 %toRet
}

; constant is medium and not fit in 6 bit (compress imm),
; but fit in 12 bit (imm)
; RV32IFDC-LABEL: <f_medium_bedge_pos>:
; RV32IFDC: addi [[MAYZEROREG:.*]], [[REG:.*]], -2047
; RV32IFDC: RESBROPT [[MAYZEROREG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <f_medium_bedge_pos>:
; RV32IFD: addi [[REG:.*]], zero, 2047
; RV32IFD: RESBRNORMAL [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @f_medium_bedge_pos(i32 %in0) minsize {
  %cmp = icmp CMPCOND i32 %in0, 2047
  br i1 %cmp, label %if.then, label %if.else
if.then:
  %call = shl i32 %in0, 1
  br label %if.end
if.else:
  %call2 = add i32 %in0, 42
  br label %if.end

if.end:
  %toRet = phi i32 [ %call, %if.then ], [ %call2, %if.else ]
  ret i32 %toRet
}

; constant is medium and not fit in 6 bit (compress imm),
; but fit in 12 bit (imm), negative value fit in 12 bit too.
; RV32IFDC-LABEL: <f_medium_bedge_neg>:
; RV32IFDC: addi [[MAYZEROREG:.*]], [[REG:.*]], 2047
; RV32IFDC: RESBROPT [[MAYZEROREG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <f_medium_bedge_neg>:
; RV32IFD: addi [[REG:.*]], zero, -2047
; RV32IFD: RESBRNORMAL [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @f_medium_bedge_neg(i32 %in0) minsize {
  %cmp = icmp CMPCOND i32 %in0, -2047
  br i1 %cmp, label %if.then, label %if.else
if.then:
  %call = shl i32 %in0, 1
  br label %if.end
if.else:
  %call2 = add i32 %in0, 42
  br label %if.end

if.end:
  %toRet = phi i32 [ %call, %if.then ], [ %call2, %if.else ]
  ret i32 %toRet
}

; constant is big and do not fit in 12 bit (imm), fit in i32
; RV32IFDC-LABEL: <f_big_ledge_pos>:
; RV32IFDC-NOT: RESBROPT
; --- no compress extension
; nothing to check.
define i32 @f_big_ledge_pos(i32 %in0) minsize {
  %cmp = icmp CMPCOND i32 %in0, 2048
  br i1 %cmp, label %if.then, label %if.else
if.then:
  %call = shl i32 %in0, 1
  br label %if.end
if.else:
  %call2 = add i32 %in0, 42
  br label %if.end

if.end:
  %toRet = phi i32 [ %call, %if.then ], [ %call2, %if.else ]
  ret i32 %toRet
}

; constant is big and do not fit in 12 bit (imm), fit in i32
; RV32IFDC-LABEL: <f_big_ledge_neg>:
; RV32IFDC-NOT: c.beqz
; --- no compress extension
; nothing to check.
define i32 @f_big_ledge_neg(i32 %in0) minsize {
  %cmp = icmp CMPCOND i32 %in0, -2048
  br i1 %cmp, label %if.then, label %if.else
if.then:
  %call = shl i32 %in0, 1
  br label %if.end
if.else:
  %call2 = add i32 %in0, 42
  br label %if.end

if.end:
  %toRet = phi i32 [ %call, %if.then ], [ %call2, %if.else ]
  ret i32 %toRet
}
