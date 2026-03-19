; RUN: llc  -mtriple=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16_h
; RUN: llc  -mtriple=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16_b

@.str = private unnamed_addr constant [9 x i8] c"%hd %c \0A\00", align 1
@sp = common global ptr null, align 4
@cp = common global ptr null, align 4

declare i32 @printf(ptr nocapture, ...) nounwind

define void @test() nounwind {
entry:
  %s = alloca i16, align 4
  %c = alloca i8, align 4
  store i16 16, ptr %s, align 4
  store i8 99, ptr %c, align 4
  store ptr %s, ptr @sp, align 4
  store ptr %c, ptr @cp, align 4
  %call.i.i = call i32 (ptr, ...) @printf(ptr @.str, i32 16, i32 99) nounwind
  %0 = load ptr, ptr @sp, align 4
  store i16 32, ptr %0, align 2
  %1 = load ptr, ptr @cp, align 4
  store i8 97, ptr %1, align 1
  %2 = load i16, ptr %s, align 4
  %3 = load i8, ptr %c, align 4
  %conv.i = sext i16 %2 to i32
  %conv1.i = sext i8 %3 to i32
  %call.i = call i32 (ptr, ...) @printf(ptr @.str, i32 %conv.i, i32 %conv1.i) nounwind
  ret void
; 16_b-LABEL: test:
; 16_h-LABEL: test:
; 16_b:	sb	${{[0-9]+}}, [[offset1:[0-9]+]](${{[0-9]+}})
; 16_b: lb      ${{[0-9]+}}, [[offset1]](${{[0-9]+}})
; 16_h:	sh	${{[0-9]+}}, [[offset2:[0-9]+]](${{[0-9]+}})
; 16_h: lh      ${{[0-9]+}}, [[offset2]](${{[0-9]+}})
}

declare void @llvm.lifetime.start.p0(i64, ptr nocapture) nounwind

declare void @llvm.lifetime.end.p0(i64, ptr nocapture) nounwind

