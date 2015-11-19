; RUN: llc -march=mipsel -relocation-model=pic -enable-mips-tail-calls < %s | \
; RUN: FileCheck %s -check-prefix=PIC32
; RUN: llc -march=mipsel -relocation-model=static \
; RUN: -enable-mips-tail-calls < %s | FileCheck %s -check-prefix=STATIC32
; RUN: llc -march=mips64el -mcpu=mips64r2 -mattr=+n64 -enable-mips-tail-calls \
; RUN: < %s | FileCheck %s -check-prefix=N64
; RUN: llc -march=mipsel -mattr=mips16 -relocation-model=pic \
; RUN: -enable-mips-tail-calls < %s | FileCheck %s -check-prefix=PIC16

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
; PIC32-NOT: jalr
; STATIC32-NOT: jal
; N64-NOT: jalr
; PIC16: jalrc

  %call = tail call i32 @callee1(i32 1, i32 1, i32 1, i32 %a0) nounwind
  ret i32 %call
}

declare i32 @callee1(i32, i32, i32, i32)

define i32 @caller2(i32 %a0, i32 %a1, i32 %a2, i32 %a3) nounwind {
entry:
; PIC32: jalr
; STATIC32: jal
; N64-NOT: jalr
; PIC16: jalrc

  %call = tail call i32 @callee2(i32 1, i32 %a0, i32 %a1, i32 %a2, i32 %a3) nounwind
  ret i32 %call
}

declare i32 @callee2(i32, i32, i32, i32, i32)

define i32 @caller3(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4) nounwind {
entry:
; PIC32: jalr
; STATIC32: jal
; N64-NOT: jalr
; PIC16: jalrc

  %call = tail call i32 @callee3(i32 1, i32 1, i32 1, i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4) nounwind
  ret i32 %call
}

declare i32 @callee3(i32, i32, i32, i32, i32, i32, i32, i32)

define i32 @caller4(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7) nounwind {
entry:
; PIC32: jalr
; STATIC32: jal
; N64: jalr
; PIC16: jalrc

  %call = tail call i32 @callee4(i32 1, i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7) nounwind
  ret i32 %call
}

declare i32 @callee4(i32, i32, i32, i32, i32, i32, i32, i32, i32)

define i32 @caller5() nounwind readonly {
entry:
; PIC32: .ent caller5
; PIC32-NOT: jalr
; PIC32: .end caller5
; STATIC32: .ent caller5
; STATIC32-NOT: jal
; STATIC32: .end caller5
; N64: .ent caller5
; N64-NOT: jalr
; N64: .end caller5
; PIC16: .ent caller5
; PIC16: jalrc
; PIC16: .end caller5

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
  %call = tail call fastcc i32 @caller8_1()
  ret i32 %call
}

define internal fastcc i32 @caller8_1() nounwind noinline {
entry:
; PIC32: .ent caller8_1
; PIC32: jalr
; PIC32: .end caller8_1
; STATIC32: .ent caller8_1
; STATIC32: jal
; STATIC32: .end caller8_1
; N64: .ent caller8_1
; N64-NOT: jalr
; N64: .end caller8_1
; PIC16: .ent caller8_1
; PIC16: jalrc
; PIC16: .end caller8_1

  %call = tail call i32 (i32, ...) @callee8(i32 2, i32 1) nounwind
  ret i32 %call
}

%struct.S = type { [2 x i32] }

@gs1 = external global %struct.S

declare i32 @callee9(%struct.S* byval)

define i32 @caller9_0() nounwind {
entry:
  %call = tail call fastcc i32 @caller9_1()
  ret i32 %call
}

define internal fastcc i32 @caller9_1() nounwind noinline {
entry:
; PIC32: .ent caller9_1
; PIC32: jalr
; PIC32: .end caller9_1
; STATIC32: .ent caller9_1
; STATIC32: jal
; STATIC32: .end caller9_1
; N64: .ent caller9_1
; N64: jalr
; N64: .end caller9_1
; PIC16: .ent caller9_1
; PIC16: jalrc
; PIC16: .end caller9_1

  %call = tail call i32 @callee9(%struct.S* byval @gs1) nounwind
  ret i32 %call
}

declare i32 @callee10(i32, i32, i32, i32, i32, i32, i32, i32, i32)

define i32 @caller10(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7, i32 %a8) nounwind {
entry:
; PIC32: .ent caller10
; PIC32-NOT: jalr
; STATIC32: .ent caller10
; STATIC32-NOT: jal
; N64: .ent caller10
; N64-NOT: jalr
; PIC16: .ent caller10
; PIC16: jalrc

  %call = tail call i32 @callee10(i32 %a8, i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7) nounwind
  ret i32 %call
}

declare i32 @callee11(%struct.S* byval)

define i32 @caller11() nounwind noinline {
entry:
; PIC32: .ent caller11
; PIC32: jalr
; STATIC32: .ent caller11
; STATIC32: jal
; N64: .ent caller11
; N64: jalr
; PIC16: .ent caller11
; PIC16: jalrc

  %call = tail call i32 @callee11(%struct.S* byval @gs1) nounwind
  ret i32 %call
}

declare i32 @callee12()

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind

define i32 @caller12(%struct.S* nocapture byval %a0) nounwind {
entry:
; PIC32: .ent caller12
; PIC32: jalr
; STATIC32: .ent caller12
; STATIC32: jal
; N64: .ent caller12
; N64: jalr
; PIC16: .ent caller12
; PIC16: jalrc

  %0 = bitcast %struct.S* %a0 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* bitcast (%struct.S* @gs1 to i8*), i8* %0, i32 8, i1 false)
  %call = tail call i32 @callee12() nounwind
  ret i32 %call
}

declare i32 @callee13(i32, ...)

define i32 @caller13() nounwind {
entry:
; PIC32: .ent caller13
; PIC32-NOT: jalr
; STATIC32: .ent caller13
; STATIC32-NOT: jal
; N64: .ent caller13
; N64-NOT: jalr
; PIC16: .ent caller13
; PIC16: jalrc

  %call = tail call i32 (i32, ...) @callee13(i32 1, i32 2) nounwind
  ret i32 %call
}

; Check that there is a chain edge between the load and store nodes.
;
; PIC32-LABEL: caller14:
; PIC32: lw ${{[0-9]+}}, 16($sp)
; PIC32: sw $4, 16($sp)

define void @caller14(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) {
entry:
  tail call void @callee14(i32 %e, i32 %b, i32 %c, i32 %d, i32 %a)
  ret void
}

declare void @callee14(i32, i32, i32, i32, i32)
