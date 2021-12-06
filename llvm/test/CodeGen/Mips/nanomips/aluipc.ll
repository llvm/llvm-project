; RUN: llc -mtriple=nanomips -verify-machineinstrs < %s | FileCheck %s

@Foo1 = global [3 x i16] zeroinitializer, align 2
@Foo2 = global [3 x i8] zeroinitializer, align 1
@Foo3 = global [3 x i16] zeroinitializer, align 2
@Foo4 = global [3 x i8] zeroinitializer, align 1

define signext i16 @aluipc_i16() {
; CHECK-LABEL: aluipc_i16
; CHECK: aluipc $a0, %pcrel_hi(Foo1+2)
; CHECK: lh $a0, %lo(Foo1+2)($a0)
entry:
  %0 = load i16, i16* getelementptr inbounds ([3 x i16], [3 x i16]* @Foo1, i32 0, i32 1), align 2
  ret i16 %0
}

define signext i16 @not_aluipc_i16() {
; CHECK-LABEL: not_aluipc_i16
; CHECK-NOT: aluipc ${{[ast][0-7]}}, %pcrel_hi(Foo1+2)
entry:
  %0 = load i16, i16* getelementptr inbounds ([3 x i16], [3 x i16]* @Foo1, i32 0, i32 1), align 2
  %1 = load i16, i16* getelementptr inbounds ([3 x i16], [3 x i16]* @Foo1, i32 0, i32 2), align 2
  %add = add i16 %1, %0
  ret i16 %add
}

define signext i8 @aluipc_i8() {
; CHECK-LABEL: aluipc_i8
; CHECK: aluipc $a0, %pcrel_hi(Foo2+1)
; CHECK: lb $a0, %lo(Foo2+1)($a0)
entry:
  %0 = load i8, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @Foo2, i32 0, i32 1), align 1
  ret i8 %0
}

define signext i8 @not_aluipc_i8() {
; CHECK-LABEL: not_aluipc_i8
; CHECK-NOT: aluipc ${{[ast][0-7]}}, %pcrel_hi(Foo2+1)
entry:
  %0 = load i8, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @Foo2, i32 0, i32 1), align 1
  %1 = load i8, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @Foo2, i32 0, i32 2), align 1
  %add = add i8 %1, %0
  ret i8 %add
}

define zeroext i16 @aluipc_u16() {
; CHECK-LABEL: aluipc_u16
; CHECK: aluipc $a0, %pcrel_hi(Foo3+2)
; CHECK: lhu $a0, %lo(Foo3+2)($a0)
entry:
  %0 = load i16, i16* getelementptr inbounds ([3 x i16], [3 x i16]* @Foo3, i32 0, i32 1), align 2
  ret i16 %0
}

define zeroext i16 @not_aluipc_u16() {
; CHECK-LABEL: not_aluipc_u16
; CHECK-NOT: aluipc ${{[ast][0-7]}}, %pcrel_hi(Foo3+2)
entry:
  %0 = load i16, i16* getelementptr inbounds ([3 x i16], [3 x i16]* @Foo3, i32 0, i32 1), align 2
  %1 = load i16, i16* getelementptr inbounds ([3 x i16], [3 x i16]* @Foo3, i32 0, i32 2), align 2
  %add = add i16 %1, %0
  ret i16 %add
}

define zeroext i8 @aluipc_u8() {
; CHECK-LABEL: aluipc_u8
; CHECK: aluipc $a0, %pcrel_hi(Foo4+1)
; CHECK: lbu $a0, %lo(Foo4+1)($a0)
entry:
  %0 = load i8, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @Foo4, i32 0, i32 1), align 1
  ret i8 %0
}

define zeroext i8 @not_aluipc_u8() {
; CHECK-LABEL: not_aluipc_u8
; CHECK-NOT: aluipc ${{[ast][0-7]}}, %pcrel_hi(Foo4+1)
entry:
  %0 = load i8, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @Foo4, i32 0, i32 1), align 1
  %1 = load i8, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @Foo4, i32 0, i32 2), align 1
  %add = add i8 %1, %0
  ret i8 %add
}
