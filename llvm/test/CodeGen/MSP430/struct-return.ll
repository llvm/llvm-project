; RUN: llc < %s | FileCheck %s

target datalayout = "e-p:16:16:16-i8:8:8-i16:16:16-i32:16:32-n8:16"
target triple = "msp430---elf"

; Pass large structures by reference (MSP430 EABI p. 3.5)

%s = type { i64, i64 }

define %s @fred() #0 {
; CHECK-LABEL: fred:
; CHECK: mov	#2314, 14(r12)
; CHECK: mov	#2828, 12(r12)
; CHECK: mov	#3342, 10(r12)
; CHECK: mov	#3840, 8(r12)
; CHECK: mov	#258, 6(r12)
; CHECK: mov	#772, 4(r12)
; CHECK: mov	#1286, 2(r12)
; CHECK: mov	#1800, 0(r12)
  ret %s {i64 72623859790382856, i64 651345242494996224}
}

%struct.S = type { i16, i16, i16 }

@a = common global i16 0, align 2
@b = common global i16 0, align 2
@c = common global i16 0, align 2

define void @test() #1 {
; CHECK-LABEL: test:
  %1 = alloca %struct.S, align 2
; CHECK:      mov	r1, r12
; CHECK-NEXT: call	#sret
  call void @sret(ptr nonnull sret(%struct.S) %1) #3
  ret void
}

define void @sret(ptr noalias nocapture sret(%struct.S)) #0 {
; CHECK-LABEL: sret:
; CHECK: mov	&a, 0(r12)
; CHECK: mov	&b, 2(r12)
; CHECK: mov	&c, 4(r12)
  %2 = getelementptr inbounds %struct.S, ptr %0, i16 0, i32 0
  %3 = load i16, ptr @a, align 2
  store i16 %3, ptr %2, align 2
  %4 = getelementptr inbounds %struct.S, ptr %0, i16 0, i32 1
  %5 = load i16, ptr @b, align 2
  store i16 %5, ptr %4, align 2
  %6 = getelementptr inbounds %struct.S, ptr %0, i16 0, i32 2
  %7 = load i16, ptr @c, align 2
  store i16 %7, ptr %6, align 2
  ret void
}

attributes #0 = { nounwind }
