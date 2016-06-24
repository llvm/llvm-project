; RUN: llc -march=mips -mcpu=mips32   < %s | FileCheck %s -check-prefixes=ALL,32
; RUN: llc -march=mips -mcpu=mips32r2 < %s | FileCheck %s -check-prefixes=ALL,32
; RUN: llc -march=mips -mcpu=mips32r6 < %s | FileCheck %s -check-prefixes=ALL,32R6
; RUN: llc -march=mips -mcpu=mips32 -mattr=dsp < %s | FileCheck %s -check-prefix=DSP
; RUN: llc -march=mips -mcpu=mips64   -target-abi n64 < %s | FileCheck %s -check-prefixes=ALL,64
; RUN: llc -march=mips -mcpu=mips64r2 -target-abi n64 < %s | FileCheck %s -check-prefixes=ALL,64
; RUN: llc -march=mips -mcpu=mips64r6 -target-abi n64 < %s | FileCheck %s -check-prefixes=ALL,64R6

; FIXME: The MIPS16 test should check its output
; RUN: llc -march=mips -mattr=mips16 < %s

; ALL-LABEL: madd1:

; 32-DAG:        sra $[[T0:[0-9]+]], $6, 31
; 32-DAG:        mtlo $6
; 32-DAG:        [[m:m]]add ${{[45]}}, ${{[45]}}
; 32-DAG:        [[m]]fhi $2
; 32-DAG:        [[m]]flo $3

; DSP-DAG:       sra $[[T0:[0-9]+]], $6, 31
; DSP-DAG:       mtlo $6, $[[AC:ac[0-3]+]]
; DSP-DAG:       madd $[[AC]], ${{[45]}}, ${{[45]}}
; DSP-DAG:       mfhi $2, $[[AC]]
; DSP-DAG:       mflo $3, $[[AC]]

; 32R6-DAG:      mul  $[[T0:[0-9]+]], ${{[45]}}, ${{[45]}}
; 32R6-DAG:      addu $[[T1:[0-9]+]], $[[T0]], $6
; 32R6-DAG:      sltu $[[T2:[0-9]+]], $[[T1]], $6
; 32R6-DAG:      sra  $[[T3:[0-9]+]], $6, 31
; 32R6-DAG:      addu $[[T4:[0-9]+]], $[[T2]], $[[T3]]
; 32R6-DAG:      muh  $[[T5:[0-9]+]], ${{[45]}}, ${{[45]}}
; 32R6-DAG:      addu $2, $[[T5]], $[[T4]]

; 64-DAG:        sll $[[T0:[0-9]+]], $4, 0
; 64-DAG:        sll $[[T1:[0-9]+]], $5, 0
; 64-DAG:        d[[m:m]]ult $[[T1]], $[[T0]]
; 64-DAG:        [[m]]flo $[[T2:[0-9]+]]
; 64-DAG:        sll $[[T3:[0-9]+]], $6, 0
; 64-DAG:        daddu $2, $[[T2]], $[[T3]]

; 64R6-DAG:      sll $[[T0:[0-9]+]], $4, 0
; 64R6-DAG:      sll $[[T1:[0-9]+]], $5, 0
; 64R6-DAG:      dmul $[[T2:[0-9]+]], $[[T1]], $[[T0]]
; 64R6-DAG:      sll $[[T3:[0-9]+]], $6, 0
; 64R6-DAG:      daddu $2, $[[T2]], $[[T3]]

define i64 @madd1(i32 %a, i32 %b, i32 %c) nounwind readnone {
entry:
  %conv = sext i32 %a to i64
  %conv2 = sext i32 %b to i64
  %mul = mul nsw i64 %conv2, %conv
  %conv4 = sext i32 %c to i64
  %add = add nsw i64 %mul, %conv4
  ret i64 %add
}

; ALL-LABEL: madd2:

; FIXME: We don't really need this instruction
; 32-DAG:        addiu $[[T0:[0-9]+]], $zero, 0
; 32-DAG:        mtlo $6
; 32-DAG:        [[m:m]]addu ${{[45]}}, ${{[45]}}
; 32-DAG:        [[m]]fhi $2
; 32-DAG:        [[m]]flo $3

; DSP-DAG:       addiu $[[T0:[0-9]+]], $zero, 0
; DSP-DAG:       mtlo $6, $[[AC:ac[0-3]+]]
; DSP-DAG:       maddu $[[AC]], ${{[45]}}, ${{[45]}}
; DSP-DAG:       mfhi $2, $[[AC]]
; DSP-DAG:       mflo $3, $[[AC]]

; 32R6-DAG:      mul  $[[T0:[0-9]+]], ${{[45]}}, ${{[45]}}
; 32R6-DAG:      addu $[[T1:[0-9]+]], $[[T0]], $6
; 32R6-DAG:      sltu $[[T2:[0-9]+]], $[[T1]], $6
; FIXME: There's a redundant move here. We should remove it
; 32R6-DAG:      muhu $[[T3:[0-9]+]], ${{[45]}}, ${{[45]}}
; 32R6-DAG:      addu $2, $[[T3]], $[[T2]]

; 64-DAG:        d[[m:m]]ult $5, $4
; 64-DAG:        [[m]]flo $[[T0:[0-9]+]]
; 64-DAG:        daddu $2, $[[T0]], $6

; 64R6-DAG:      dmul $[[T0:[0-9]+]], $5, $4
; 64R6-DAG:      daddu $2, $[[T0]], $6

define i64 @madd2(i32 zeroext %a, i32 zeroext %b, i32 zeroext %c) nounwind readnone {
entry:
  %conv = zext i32 %a to i64
  %conv2 = zext i32 %b to i64
  %mul = mul nsw i64 %conv2, %conv
  %conv4 = zext i32 %c to i64
  %add = add nsw i64 %mul, %conv4
  ret i64 %add
}

; ALL-LABEL: madd3:

; 32-DAG:        mthi $6
; 32-DAG:        mtlo $7
; 32-DAG:        [[m:m]]add ${{[45]}}, ${{[45]}}
; 32-DAG:        [[m]]fhi $2
; 32-DAG:        [[m]]flo $3

; DSP-DAG:       mthi $6, $[[AC:ac[0-3]+]]
; DSP-DAG:       mtlo $7, $[[AC]]
; DSP-DAG:       madd $[[AC]], ${{[45]}}, ${{[45]}}
; DSP-DAG:       mfhi $2, $[[AC]]
; DSP-DAG:       mflo $3, $[[AC]]

; 32R6-DAG:      mul  $[[T0:[0-9]+]], ${{[45]}}, ${{[45]}}
; 32R6-DAG:      addu $[[T1:[0-9]+]], $[[T0]], $7
; 32R6-DAG:      sltu $[[T2:[0-9]+]], $[[T1]], $7
; 32R6-DAG:      addu $[[T4:[0-9]+]], $[[T2]], $6
; 32R6-DAG:      muh  $[[T5:[0-9]+]], ${{[45]}}, ${{[45]}}
; 32R6-DAG:      addu $2, $[[T5]], $[[T4]]

; 64-DAG:        sll $[[T0:[0-9]+]], $4, 0
; 64-DAG:        sll $[[T1:[0-9]+]], $5, 0
; 64-DAG:        d[[m:m]]ult $[[T1]], $[[T0]]
; 64-DAG:        [[m]]flo $[[T2:[0-9]+]]
; 64-DAG:        daddu $2, $[[T2]], $6

; 64R6-DAG:      sll $[[T0:[0-9]+]], $4, 0
; 64R6-DAG:      sll $[[T1:[0-9]+]], $5, 0
; 64R6-DAG:      dmul $[[T2:[0-9]+]], $[[T1]], $[[T0]]
; 64R6-DAG:      daddu $2, $[[T2]], $6

define i64 @madd3(i32 %a, i32 %b, i64 %c) nounwind readnone {
entry:
  %conv = sext i32 %a to i64
  %conv2 = sext i32 %b to i64
  %mul = mul nsw i64 %conv2, %conv
  %add = add nsw i64 %mul, %c
  ret i64 %add
}

; ALL-LABEL: msub1:

; 32-DAG:        sra $[[T0:[0-9]+]], $6, 31
; 32-DAG:        mtlo $6
; 32-DAG:        [[m:m]]sub ${{[45]}}, ${{[45]}}
; 32-DAG:        [[m]]fhi $2
; 32-DAG:        [[m]]flo $3

; DSP-DAG:       sra $[[T0:[0-9]+]], $6, 31
; DSP-DAG:       mtlo $6, $[[AC:ac[0-3]+]]
; DSP-DAG:       msub $[[AC]], ${{[45]}}, ${{[45]}}
; DSP-DAG:       mfhi $2, $[[AC]]
; DSP-DAG:       mflo $3, $[[AC]]

; 32R6-DAG:      muh  $[[T0:[0-9]+]], ${{[45]}}, ${{[45]}}
; 32R6-DAG:      mul  $[[T1:[0-9]+]], ${{[45]}}, ${{[45]}}
; 32R6-DAG:      sltu $[[T3:[0-9]+]], $6, $[[T1]]
; 32R6-DAG:      addu $[[T4:[0-9]+]], $[[T3]], $[[T0]]
; 32R6-DAG:      sra  $[[T5:[0-9]+]], $6, 31
; 32R6-DAG:      subu $2, $[[T5]], $[[T4]]
; 32R6-DAG:      subu $3, $6, $[[T1]]

; 64-DAG:        sll $[[T0:[0-9]+]], $4, 0
; 64-DAG:        sll $[[T1:[0-9]+]], $5, 0
; 64-DAG:        d[[m:m]]ult $[[T1]], $[[T0]]
; 64-DAG:        [[m]]flo $[[T2:[0-9]+]]
; 64-DAG:        sll $[[T3:[0-9]+]], $6, 0
; 64-DAG:        dsubu $2, $[[T3]], $[[T2]]

; 64R6-DAG:      sll $[[T0:[0-9]+]], $4, 0
; 64R6-DAG:      sll $[[T1:[0-9]+]], $5, 0
; 64R6-DAG:      dmul $[[T2:[0-9]+]], $[[T1]], $[[T0]]
; 64R6-DAG:      sll $[[T3:[0-9]+]], $6, 0
; 64R6-DAG:      dsubu $2, $[[T3]], $[[T2]]

define i64 @msub1(i32 %a, i32 %b, i32 %c) nounwind readnone {
entry:
  %conv = sext i32 %c to i64
  %conv2 = sext i32 %a to i64
  %conv4 = sext i32 %b to i64
  %mul = mul nsw i64 %conv4, %conv2
  %sub = sub nsw i64 %conv, %mul
  ret i64 %sub
}

; ALL-LABEL: msub2:

; FIXME: We don't really need this instruction
; 32-DAG:        addiu $[[T0:[0-9]+]], $zero, 0
; 32-DAG:        mtlo $6
; 32-DAG:        [[m:m]]subu ${{[45]}}, ${{[45]}}
; 32-DAG:        [[m]]fhi $2
; 32-DAG:        [[m]]flo $3

; DSP-DAG:       addiu $[[T0:[0-9]+]], $zero, 0
; DSP-DAG:       mtlo $6, $[[AC:ac[0-3]+]]
; DSP-DAG:       msubu $[[AC]], ${{[45]}}, ${{[45]}}
; DSP-DAG:       mfhi $2, $[[AC]]
; DSP-DAG:       mflo $3, $[[AC]]

; 32R6-DAG:      muhu $[[T0:[0-9]+]], ${{[45]}}, ${{[45]}}
; 32R6-DAG:      mul $[[T1:[0-9]+]], ${{[45]}}, ${{[45]}}

; 32R6-DAG:      sltu $[[T2:[0-9]+]], $6, $[[T1]]
; 32R6-DAG:      addu $[[T3:[0-9]+]], $[[T2]], $[[T0]]
; 32R6-DAG:      negu $2, $[[T3]]
; 32R6-DAG:      subu $3, $6, $[[T1]]

; 64-DAG:        d[[m:m]]ult $5, $4
; 64-DAG:        [[m]]flo $[[T0:[0-9]+]]
; 64-DAG:        dsubu $2, $6, $[[T0]]

; 64R6-DAG:      dmul $[[T0:[0-9]+]], $5, $4
; 64R6-DAG:      dsubu $2, $6, $[[T0]]

define i64 @msub2(i32 zeroext %a, i32 zeroext %b, i32 zeroext %c) nounwind readnone {
entry:
  %conv = zext i32 %c to i64
  %conv2 = zext i32 %a to i64
  %conv4 = zext i32 %b to i64
  %mul = mul nsw i64 %conv4, %conv2
  %sub = sub nsw i64 %conv, %mul
  ret i64 %sub
}

; ALL-LABEL: msub3:

; FIXME: We don't really need this instruction
; 32-DAG:        mthi $6
; 32-DAG:        mtlo $7
; 32-DAG:        [[m:m]]sub ${{[45]}}, ${{[45]}}
; 32-DAG:        [[m]]fhi $2
; 32-DAG:        [[m]]flo $3

; DSP-DAG:       addiu $[[T0:[0-9]+]], $zero, 0
; DSP-DAG:       mtlo $6, $[[AC:ac[0-3]+]]
; DSP-DAG:       msub $[[AC]], ${{[45]}}, ${{[45]}}
; DSP-DAG:       mfhi $2, $[[AC]]
; DSP-DAG:       mflo $3, $[[AC]]

; 32R6-DAG:      muh $[[T0:[0-9]+]], ${{[45]}}, ${{[45]}}
; 32R6-DAG:      mul $[[T1:[0-9]+]], ${{[45]}}, ${{[45]}}
; 32R6-DAG:      sltu $[[T2:[0-9]+]], $7, $[[T1]]
; 32R6-DAG:      addu $[[T3:[0-9]+]], $[[T2]], $[[T0]]
; 32R6-DAG:      subu $2, $6, $[[T3]]
; 32R6-DAG:      subu $3, $7, $[[T1]]

; 64-DAG:        sll $[[T0:[0-9]+]], $4, 0
; 64-DAG:        sll $[[T1:[0-9]+]], $5, 0
; 64-DAG:        d[[m:m]]ult $[[T1]], $[[T0]]
; 64-DAG:        [[m]]flo $[[T2:[0-9]+]]
; 64-DAG:        dsubu $2, $6, $[[T2]]

; 64R6-DAG:      sll $[[T0:[0-9]+]], $4, 0
; 64R6-DAG:      sll $[[T1:[0-9]+]], $5, 0
; 64R6-DAG:      dmul $[[T2:[0-9]+]], $[[T1]], $[[T0]]
; 64R6-DAG:      dsubu $2, $6, $[[T2]]

define i64 @msub3(i32 %a, i32 %b, i64 %c) nounwind readnone {
entry:
  %conv = sext i32 %a to i64
  %conv3 = sext i32 %b to i64
  %mul = mul nsw i64 %conv3, %conv
  %sub = sub nsw i64 %c, %mul
  ret i64 %sub
}
