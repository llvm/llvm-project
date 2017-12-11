; RUN: llc -march=mipsel -relocation-model=pic  \
; RUN:     -verify-machineinstrs -mips-tail-calls=1 < %s | FileCheck %s -check-prefixes=ALL,PIC32
; RUN: llc -march=mipsel -relocation-model=static  \
; RUN:     -verify-machineinstrs -mips-tail-calls=1 < %s | FileCheck %s -check-prefixes=ALL,STATIC32
; RUN: llc -march=mips64el -mcpu=mips64r2  -relocation-model=pic \
; RUN:     -verify-machineinstrs -mips-tail-calls=1 < %s | FileCheck %s -check-prefixes=ALL,PIC64
; RUN: llc -march=mips64el -mcpu=mips64r2  -relocation-model=static \
; RUN:     -verify-machineinstrs -mips-tail-calls=1 < %s | FileCheck %s -check-prefixes=ALL,STATIC64
; RUN: llc -march=mipsel -mattr=mips16 -relocation-model=pic \
; RUN:      -verify-machineinstrs -mips-tail-calls=1 < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,PIC16

; RUN: llc -march=mipsel -relocation-model=pic -mattr=+micromips -mips-tail-calls=1 < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,PIC32MM
; RUN: llc -march=mipsel -relocation-model=static -mattr=+micromips \
; RUN:     -mips-tail-calls=1 < %s | FileCheck %s -check-prefixes=ALL,STATIC32

; RUN: llc -march=mipsel -relocation-model=pic -mcpu=mips32r6 -mips-tail-calls=1 < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,PIC32R6
; RUN: llc -march=mipsel -relocation-model=static -mcpu=mips32r2 \
; RUN:     -mips-tail-calls=1 < %s | FileCheck %s -check-prefixes=ALL,STATIC32
; RUN: llc -march=mips64el -relocation-model=pic -mcpu=mips64r2  \
; RUN:     -mips-tail-calls=1 < %s | FileCheck %s -check-prefix=PIC64
; RUN: llc -march=mips64el -relocation-model=pic -mcpu=mips64r6  \
; RUN:     -mips-tail-calls=1 < %s | FileCheck %s -check-prefix=STATIC64

; RUN: llc -march=mipsel -relocation-model=pic -mcpu=mips32r6 -mattr=+micromips \
; RUN:      -mips-tail-calls=1 < %s | FileCheck %s -check-prefixes=ALL,PIC32MM
; RUN: llc -march=mipsel -relocation-model=static -mcpu=mips32r6 \
; RUN:     -mattr=+micromips -mips-tail-calls=1 < %s | FileCheck %s -check-prefixes=ALL,STATIC32MMR6

@g0 = common global i32 0, align 4
@g1 = common global i32 0, align 4
@g2 = common global i32 0, align 4
@g3 = common global i32 0, align 4
@g4 = common global i32 0, align 4
@g5 = common global i32 0, align 4
@g6 = common global i32 0, align 4
@g7 = common global i32 0, align 4
@g8 = common global i32 0, align 4
@g9 = common global i32 0, align 4

define i32 @caller1(i32 %a0) nounwind {
entry:
; ALL-LABEL: caller1:
; PIC32: jalr $25
; PIC32MM: jalr $25
; PIC32R6: jalr $25
; STATIC32: jal
; STATIC32MMR6: jal
; N64: jalr $25
; N64R6: jalr $25
; PIC16: jalrc

  %call = tail call i32 @callee1(i32 1, i32 1, i32 1, i32 %a0) nounwind
  ret i32 %call
}

declare i32 @callee1(i32, i32, i32, i32)

define i32 @caller2(i32 %a0, i32 %a1, i32 %a2, i32 %a3) nounwind {
entry:
; ALL-LABEL: caller2
; PIC32: jalr $25
; PIC32MM: jalr $25
; PIC32R6: jalr $25
; STATIC32: jal
; STATIC32MMR6: jal
; N64: jalr $25
; N64R6: jalr $25
; PIC16: jalrc

  %call = tail call i32 @callee2(i32 1, i32 %a0, i32 %a1, i32 %a2, i32 %a3) nounwind
  ret i32 %call
}

declare i32 @callee2(i32, i32, i32, i32, i32)

define i32 @caller3(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4) nounwind {
entry:
; ALL-LABEL: caller3:
; PIC32: jalr $25
; PIC32R6: jalr $25
; PIC32MM: jalr $25
; STATIC32: jal
; STATIC32MMR6: jal
; N64: jalr $25
; N64R6: jalr $25
; PIC16: jalrc

  %call = tail call i32 @callee3(i32 1, i32 1, i32 1, i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4) nounwind
  ret i32 %call
}

declare i32 @callee3(i32, i32, i32, i32, i32, i32, i32, i32)

define i32 @caller4(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7) nounwind {
entry:
; ALL-LABEL: caller4:
; PIC32: jalr $25
; PIC32R6: jalr $25
; PIC32MM: jalr $25
; STATIC32: jal
; SATATIC32MMR6: jal
; PIC64: jalr $25
; STATIC64: jal
; N64R6: jalr $25
; PIC16: jalrc

  %call = tail call i32 @callee4(i32 1, i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7) nounwind
  ret i32 %call
}

declare i32 @callee4(i32, i32, i32, i32, i32, i32, i32, i32, i32)

define i32 @caller5() nounwind readonly {
entry:
; ALL-LABEL: caller5:
; PIC32: jr $25
; PIC32R6: jr $25
; PIC32MM: jr
; STATIC32: j
; STATIC32MMR6: bc
; PIC64: jr $25
; STATIC64: j
; PIC16: jalrc

  %0 = load i32, i32* @g0, align 4
  %1 = load i32, i32* @g1, align 4
  %2 = load i32, i32* @g2, align 4
  %3 = load i32, i32* @g3, align 4
  %4 = load i32, i32* @g4, align 4
  %5 = load i32, i32* @g5, align 4
  %6 = load i32, i32* @g6, align 4
  %7 = load i32, i32* @g7, align 4
  %8 = load i32, i32* @g8, align 4
  %9 = load i32, i32* @g9, align 4
  %call = tail call fastcc i32 @callee5(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9)
  ret i32 %call
}

define internal fastcc i32 @callee5(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7, i32 %a8, i32 %a9) nounwind readnone noinline {
entry:
  %add = add nsw i32 %a1, %a0
  %add1 = add nsw i32 %add, %a2
  %add2 = add nsw i32 %add1, %a3
  %add3 = add nsw i32 %add2, %a4
  %add4 = add nsw i32 %add3, %a5
  %add5 = add nsw i32 %add4, %a6
  %add6 = add nsw i32 %add5, %a7
  %add7 = add nsw i32 %add6, %a8
  %add8 = add nsw i32 %add7, %a9
  ret i32 %add8
}

declare i32 @callee8(i32, ...)

define i32 @caller8_0() nounwind {
entry:
; ALL-LABEL: caller8_0:
; PIC32: jr $25
; PIC32R6: jrc $25
; PIC32MM: jrc
; STATIC32: j
; STATIC32MMR6: bc
; PIC64: jr $25
; PIC64R6: jrc $25
; STATIC64: j
; PIC16: jalrc

  %call = tail call fastcc i32 @caller8_1()
  ret i32 %call
}

define internal fastcc i32 @caller8_1() nounwind noinline {
entry:
; ALL-LABEL: caller8_1:
; PIC32: jalr $25
; PIC32R6: jalr $25
; PIC32MM: jalr $25
; STATIC32: jal
; STATIC32MMR6: jal
; PIC64: jalr $25
; STATIC64: jal
; PIC16: jalrc

  %call = tail call i32 (i32, ...) @callee8(i32 2, i32 1) nounwind
  ret i32 %call
}

%struct.S = type { [2 x i32] }

@gs1 = external global %struct.S

declare i32 @callee9(%struct.S* byval)

define i32 @caller9_0() nounwind {
entry:
; ALL-LABEL: caller9_0:
; PIC32: jr $25
; PIC32R6: jrc $25
; PIC32MM: jrc
; STATIC32: j
; STATIC32MMR6: bc
; PIC64: jr $25
; STATIC64: j
; PIC64R6: jrc $25
; PIC16: jalrc
  %call = tail call fastcc i32 @caller9_1()
  ret i32 %call
}

define internal fastcc i32 @caller9_1() nounwind noinline {
entry:
; ALL-LABEL: caller9_1:
; PIC32: jalr $25
; PIC32R6: jalrc $25
; PIC32MM: jalr $25
; STATIC32: jal
; STATIC32MMR6: jal
; STATIC64: jal
; PIC64: jalr $25
; PIC64R6: jalrc $25
; PIC16: jalrc

  %call = tail call i32 @callee9(%struct.S* byval @gs1) nounwind
  ret i32 %call
}

declare i32 @callee10(i32, i32, i32, i32, i32, i32, i32, i32, i32)

define i32 @caller10(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7, i32 %a8) nounwind {
entry:
; ALL-LABEL: caller10:
; PIC32: jalr $25
; PIC32R6: jalr $25
; PIC32MM: jalr $25
; STATIC32: jal
; STATIC32MMR6: jal
; STATIC64: jal
; PIC64: jalr $25
; PIC64R6: jalr $25
; PIC16: jalrc

  %call = tail call i32 @callee10(i32 %a8, i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7) nounwind
  ret i32 %call
}

declare i32 @callee11(%struct.S* byval)

define i32 @caller11() nounwind noinline {
entry:
; ALL-LABEL: caller11:
; PIC32: jalr $25
; PIC32R6: jalrc $25
; PIC32MM: jalr $25
; STATIC32: jal
; STATIC32MMR6: jal
; STATIC64: jal
; PIC64: jalr $25
; PIC64R6: jalrc $25
; PIC16: jalrc

  %call = tail call i32 @callee11(%struct.S* byval @gs1) nounwind
  ret i32 %call
}

declare i32 @callee12()

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind

define i32 @caller12(%struct.S* nocapture byval %a0) nounwind {
entry:
; ALL-LABEL: caller12:
; PIC32: jalr $25
; PIC32R6: jalrc $25
; PIC32MM: jalr $25
; STATIC32: jal
; STATIC32MMR6: jal
; STATIC64: jal
; PIC64: jalr $25
; PIC64R6: jalrc $25
; PIC16: jalrc

  %0 = bitcast %struct.S* %a0 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* bitcast (%struct.S* @gs1 to i8*), i8* %0, i32 8, i32 4, i1 false)
  %call = tail call i32 @callee12() nounwind
  ret i32 %call
}

declare i32 @callee13(i32, ...)

define i32 @caller13() nounwind {
entry:
; ALL-LABEL: caller13:
; PIC32: jalr $25
; PIC32R6: jalr $25
; PIC32MM: jalr $25
; STATIC32: jal
; STATIC32MMR6: jal
; STATIC64: jal
; PIC64R6: jalr $25
; PIC64: jalr $25
; PIC16: jalrc

  %call = tail call i32 (i32, ...) @callee13(i32 1, i32 2) nounwind
  ret i32 %call
}

; Check that there is a chain edge between the load and store nodes.
;
; ALL-LABEL: caller14:
; PIC32: lw ${{[0-9]+}}, 48($sp)
; PIC32: sw $4, 16($sp)

; PIC32MM: lw ${{[0-9]+}}, 48($sp)
; PIC32MM: sw16 $4, 16(${{[0-9]+}})

define void @caller14(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) {
entry:
  tail call void @callee14(i32 %e, i32 %b, i32 %c, i32 %d, i32 %a)
  ret void
}

declare void @callee14(i32, i32, i32, i32, i32)
