; This test is designed to run 4 times, once with function attribute +c,
; once with function attribute -c for eq/ne in icmp
; The optimization should appear only with +c, otherwise default isel should be
; choosen.
;
; RUN: llc -mtriple=riscv32 -target-abi ilp32d -mattr=+c,+f,+d -filetype=obj \
; RUN:   -disable-block-placement < %s \
; RUN:   | llvm-objdump -d --triple=riscv32 --mattr=+c,+f,+d -M no-aliases - \
; RUN:   | FileCheck -check-prefix=RV32IFDC %s
;
; RUN: llc -mtriple=riscv32 -target-abi ilp32d -mattr=-c,+f,+d -filetype=obj \
; RUN:   -disable-block-placement < %s \
; RUN:   | llvm-objdump -d --triple=riscv32 --mattr=-c,+f,+d -M no-aliases - \
; RUN:   | FileCheck -check-prefix=RV32IFD %s

; constant is small and fit in 6 bit (compress imm)
; RV32IFDC-LABEL: <ne_small_pos>:
; RV32IFDC: c.li [[REG:.*]], 0x14
; RV32IFDC: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <ne_small_pos>:
; RV32IFD: addi [[REG:.*]], zero, 0x14
; RV32IFD: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @ne_small_pos(i32 %in0) minsize {
  %cmp = icmp ne i32 %in0, 20
  %toRet = select i1 %cmp, i32 -99, i32 42
  ret i32 %toRet
}

; constant is small and fit in 6 bit (compress imm)
; RV32IFDC-LABEL: <ne_small_neg>:
; RV32IFDC: c.li [[REG:.*]], -0x14
; RV32IFDC: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <ne_small_neg>:
; RV32IFD: addi [[REG:.*]], zero, -0x14
; RV32IFD: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @ne_small_neg(i32 %in0) minsize {
  %cmp = icmp ne i32 %in0, -20
  %toRet = select i1 %cmp, i32 -99, i32 42
  ret i32 %toRet
}

; constant is small and fit in 6 bit (compress imm)
; RV32IFDC-LABEL: <ne_small_edge_pos>:
; RV32IFDC: c.li [[REG:.*]], 0x1f
; RV32IFDC: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <ne_small_edge_pos>:
; RV32IFD: addi [[REG:.*]], zero, 0x1f
; RV32IFD: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @ne_small_edge_pos(i32 %in0) minsize {
  %cmp = icmp ne i32 %in0, 31
  %toRet = select i1 %cmp, i32 -99, i32 42
  ret i32 %toRet
}

; constant is small and fit in 6 bit (compress imm)
; RV32IFDC-LABEL: <ne_small_edge_neg>:
; RV32IFDC: c.li [[REG:.*]], -0x20
; RV32IFDC: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <ne_small_edge_neg>:
; RV32IFD: addi [[REG:.*]], zero, -0x20
; RV32IFD: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @ne_small_edge_neg(i32 %in0) minsize {
  %cmp = icmp ne i32 %in0, -32
  %toRet = select i1 %cmp, i32 -99, i32 42
  ret i32 %toRet
}

; constant is medium and not fit in 6 bit (compress imm),
; but fit in 12 bit (imm)
; RV32IFDC-LABEL: <ne_medium_ledge_pos>:
; RV32IFDC: addi [[MAYZEROREG:.*]], [[REG:.*]], -0x21
; RV32IFDC: [[COND:c.*]] [[MAYZEROREG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <ne_medium_ledge_pos>:
; RV32IFD: addi [[REG:.*]], zero, 0x21
; RV32IFD: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @ne_medium_ledge_pos(i32 %in0) minsize {
  %cmp = icmp ne i32 %in0, 33
  %toRet = select i1 %cmp, i32 -99, i32 42
  ret i32 %toRet
}

; constant is medium and not fit in 6 bit (compress imm),
; but fit in 12 bit (imm)
; RV32IFDC-LABEL: <ne_medium_ledge_neg>:
; RV32IFDC: addi [[MAYZEROREG:.*]], [[REG:.*]], 0x21
; RV32IFDC: [[COND:c.*]] [[MAYZEROREG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <ne_medium_ledge_neg>:
; RV32IFD: addi [[REG:.*]], zero, -0x21
; RV32IFD: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @ne_medium_ledge_neg(i32 %in0) minsize {
  %cmp = icmp ne i32 %in0, -33
  %toRet = select i1 %cmp, i32 -99, i32 42
  ret i32 %toRet
}

; constant is medium and not fit in 6 bit (compress imm),
; but fit in 12 bit (imm)
; RV32IFDC-LABEL: <ne_medium_pos>:
; RV32IFDC: addi [[MAYZEROREG:.*]], [[REG:.*]], -0x3f
; RV32IFDC: [[COND:c.*]] [[MAYZEROREG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <ne_medium_pos>:
; RV32IFD: addi [[REG:.*]], zero, 0x3f
; RV32IFD: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @ne_medium_pos(i32 %in0) minsize {
  %cmp = icmp ne i32 %in0, 63
  %toRet = select i1 %cmp, i32 -99, i32 42
  ret i32 %toRet
}

; constant is medium and not fit in 6 bit (compress imm),
; but fit in 12 bit (imm)
; RV32IFDC-LABEL: <ne_medium_neg>:
; RV32IFDC: addi [[MAYZEROREG:.*]], [[REG:.*]], 0x3f
; RV32IFDC: [[COND:c.*]] [[MAYZEROREG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <ne_medium_neg>:
; RV32IFD: addi [[REG:.*]], zero, -0x3f
; RV32IFD: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @ne_medium_neg(i32 %in0) minsize {
  %cmp = icmp ne i32 %in0, -63
  %toRet = select i1 %cmp, i32 -99, i32 42
  ret i32 %toRet
}

; constant is medium and not fit in 6 bit (compress imm),
; but fit in 12 bit (imm)
; RV32IFDC-LABEL: <ne_medium_bedge_pos>:
; RV32IFDC: addi [[MAYZEROREG:.*]], [[REG:.*]], -0x7ff
; RV32IFDC: [[COND:c.*]] [[MAYZEROREG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <ne_medium_bedge_pos>:
; RV32IFD: addi [[REG:.*]], zero, 0x7ff
; RV32IFD: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @ne_medium_bedge_pos(i32 %in0) minsize {
  %cmp = icmp ne i32 %in0, 2047
  %toRet = select i1 %cmp, i32 -99, i32 42
  ret i32 %toRet
}

; constant is medium and not fit in 6 bit (compress imm),
; but fit in 12 bit (imm), negative value fit in 12 bit too.
; RV32IFDC-LABEL: <ne_medium_bedge_neg>:
; RV32IFDC: addi [[MAYZEROREG:.*]], [[REG:.*]], 0x7ff
; RV32IFDC: [[COND:c.*]] [[MAYZEROREG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <ne_medium_bedge_neg>:
; RV32IFD: addi [[REG:.*]], zero, -0x7ff
; RV32IFD: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @ne_medium_bedge_neg(i32 %in0) minsize {
  %cmp = icmp ne i32 %in0, -2047
  %toRet = select i1 %cmp, i32 -99, i32 42
  ret i32 %toRet
}

; constant is big and do not fit in 12 bit (imm), fit in i32
; RV32IFDC-LABEL: <ne_big_ledge_pos>:
; RV32IFDC-NOT: [[COND:c.b.*]]
; --- no compress extension
; nothing to check.
define i32 @ne_big_ledge_pos(i32 %in0) minsize {
  %cmp = icmp ne i32 %in0, 2048
  %toRet = select i1 %cmp, i32 -99, i32 42
  ret i32 %toRet
}

; constant is big and do not fit in 12 bit (imm), fit in i32
; RV32IFDC-LABEL: <ne_big_ledge_neg>:
; RV32IFDC-NOT: [[COND:c.b.*]]
; --- no compress extension
; nothing to check.
define i32 @ne_big_ledge_neg(i32 %in0) minsize {
  %cmp = icmp ne i32 %in0, -2048
  %toRet = select i1 %cmp, i32 -99, i32 42
  ret i32 %toRet
}


;; Same as above, but for eq

; constant is small and fit in 6 bit (compress imm)
; RV32IFDC-LABEL: <eq_small_pos>:
; RV32IFDC: c.li [[REG:.*]], 0x14
; RV32IFDC: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <eq_small_pos>:
; RV32IFD: addi [[REG:.*]], zero, 0x14
; RV32IFD: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @eq_small_pos(i32 %in0) minsize {
  %cmp = icmp eq i32 %in0, 20
  %toRet = select i1 %cmp, i32 -99, i32 42
  ret i32 %toRet
}

; constant is small and fit in 6 bit (compress imm)
; RV32IFDC-LABEL: <eq_small_neg>:
; RV32IFDC: c.li [[REG:.*]], -0x14
; RV32IFDC: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <eq_small_neg>:
; RV32IFD: addi [[REG:.*]], zero, -0x14
; RV32IFD: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @eq_small_neg(i32 %in0) minsize {
  %cmp = icmp eq i32 %in0, -20
  %toRet = select i1 %cmp, i32 -99, i32 42
  ret i32 %toRet
}

; constant is small and fit in 6 bit (compress imm)
; RV32IFDC-LABEL: <eq_small_edge_pos>:
; RV32IFDC: c.li [[REG:.*]], 0x1f
; RV32IFDC: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <eq_small_edge_pos>:
; RV32IFD: addi [[REG:.*]], zero, 0x1f
; RV32IFD: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @eq_small_edge_pos(i32 %in0) minsize {
  %cmp = icmp eq i32 %in0, 31
  %toRet = select i1 %cmp, i32 -99, i32 42
  ret i32 %toRet
}

; constant is small and fit in 6 bit (compress imm)
; RV32IFDC-LABEL: <eq_small_edge_neg>:
; RV32IFDC: c.li [[REG:.*]], -0x20
; RV32IFDC: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <eq_small_edge_neg>:
; RV32IFD: addi [[REG:.*]], zero, -0x20
; RV32IFD: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @eq_small_edge_neg(i32 %in0) minsize {
  %cmp = icmp eq i32 %in0, -32
  %toRet = select i1 %cmp, i32 -99, i32 42
  ret i32 %toRet
}

; constant is medium and not fit in 6 bit (compress imm),
; but fit in 12 bit (imm)
; RV32IFDC-LABEL: <eq_medium_ledge_pos>:
; RV32IFDC: addi [[MAYZEROREG:.*]], [[REG:.*]], -0x21
; RV32IFDC: [[COND:c.*]] [[MAYZEROREG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <eq_medium_ledge_pos>:
; RV32IFD: addi [[REG:.*]], zero, 0x21
; RV32IFD: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @eq_medium_ledge_pos(i32 %in0) minsize {
  %cmp = icmp eq i32 %in0, 33
  %toRet = select i1 %cmp, i32 -99, i32 42
  ret i32 %toRet
}

; constant is medium and not fit in 6 bit (compress imm),
; but fit in 12 bit (imm)
; RV32IFDC-LABEL: <eq_medium_ledge_neg>:
; RV32IFDC: addi [[MAYZEROREG:.*]], [[REG:.*]], 0x21
; RV32IFDC: [[COND:c.*]] [[MAYZEROREG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <eq_medium_ledge_neg>:
; RV32IFD: addi [[REG:.*]], zero, -0x21
; RV32IFD: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @eq_medium_ledge_neg(i32 %in0) minsize {
  %cmp = icmp eq i32 %in0, -33
  %toRet = select i1 %cmp, i32 -99, i32 42
  ret i32 %toRet
}

; constant is medium and not fit in 6 bit (compress imm),
; but fit in 12 bit (imm)
; RV32IFDC-LABEL: <eq_medium_pos>:
; RV32IFDC: addi [[MAYZEROREG:.*]], [[REG:.*]], -0x3f
; RV32IFDC: [[COND:c.*]] [[MAYZEROREG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <eq_medium_pos>:
; RV32IFD: addi [[REG:.*]], zero, 0x3f
; RV32IFD: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @eq_medium_pos(i32 %in0) minsize {
  %cmp = icmp eq i32 %in0, 63
  %toRet = select i1 %cmp, i32 -99, i32 42
  ret i32 %toRet
}

; constant is medium and not fit in 6 bit (compress imm),
; but fit in 12 bit (imm)
; RV32IFDC-LABEL: <eq_medium_neg>:
; RV32IFDC: addi [[MAYZEROREG:.*]], [[REG:.*]], 0x3f
; RV32IFDC: [[COND:c.*]] [[MAYZEROREG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <eq_medium_neg>:
; RV32IFD: addi [[REG:.*]], zero, -0x3f
; RV32IFD: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @eq_medium_neg(i32 %in0) minsize {
  %cmp = icmp eq i32 %in0, -63
  %toRet = select i1 %cmp, i32 -99, i32 42
  ret i32 %toRet
}

; constant is medium and not fit in 6 bit (compress imm),
; but fit in 12 bit (imm)
; RV32IFDC-LABEL: <eq_medium_bedge_pos>:
; RV32IFDC: addi [[MAYZEROREG:.*]], [[REG:.*]], -0x7ff
; RV32IFDC: [[COND:c.*]] [[MAYZEROREG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <eq_medium_bedge_pos>:
; RV32IFD: addi [[REG:.*]], zero, 0x7ff
; RV32IFD: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @eq_medium_bedge_pos(i32 %in0) minsize {
  %cmp = icmp eq i32 %in0, 2047
  %toRet = select i1 %cmp, i32 -99, i32 42
  ret i32 %toRet
}

; constant is medium and not fit in 6 bit (compress imm),
; but fit in 12 bit (imm), negative value fit in 12 bit too.
; RV32IFDC-LABEL: <eq_medium_bedge_neg>:
; RV32IFDC: addi [[MAYZEROREG:.*]], [[REG:.*]], 0x7ff
; RV32IFDC: [[COND:c.*]] [[MAYZEROREG]], [[PLACE:.*]]
; --- no compress extension
; RV32IFD-LABEL: <eq_medium_bedge_neg>:
; RV32IFD: addi [[REG:.*]], zero, -0x7ff
; RV32IFD: [[COND:b.*]] [[ANOTHER:.*]], [[REG]], [[PLACE:.*]]
define i32 @eq_medium_bedge_neg(i32 %in0) minsize {
  %cmp = icmp eq i32 %in0, -2047
  %toRet = select i1 %cmp, i32 -99, i32 42
  ret i32 %toRet
}

; constant is big and do not fit in 12 bit (imm), fit in i32
; RV32IFDC-LABEL: <eq_big_ledge_pos>:
; RV32IFDC-NOT: [[COND:c.b.*]]
; --- no compress extension
; nothing to check.
define i32 @eq_big_ledge_pos(i32 %in0) minsize {
  %cmp = icmp eq i32 %in0, 2048
  %toRet = select i1 %cmp, i32 -99, i32 42
  ret i32 %toRet
}

; constant is big and do not fit in 12 bit (imm), fit in i32
; RV32IFDC-LABEL: <eq_big_ledge_neg>:
; RV32IFDC-NOT: [[COND:c.b.*]]
; --- no compress extension
; nothing to check.
define i32 @eq_big_ledge_neg(i32 %in0) minsize {
  %cmp = icmp eq i32 %in0, -2048
  %toRet = select i1 %cmp, i32 -99, i32 42
  ret i32 %toRet
}
