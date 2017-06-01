; RUN: llc  -march=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16_h
; RUN: llc  -march=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16_b

@.str = private unnamed_addr constant [9 x i8] c"%hd %c \0A\00", align 1
@sp = common global i16* null, align 4
@cp = common global i8* null, align 4

declare i32 @printf(i8* nocapture, ...) nounwind

define void @test() nounwind {
entry:
  %s = alloca i16, align 4
  %c = alloca i8, align 4
  store i16 16, i16* %s, align 4
  store i8 99, i8* %c, align 4
  store i16* %s, i16** @sp, align 4
  store i8* %c, i8** @cp, align 4
  %call.i.i = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str, i32 0, i32 0), i32 16, i32 99) nounwind
  %0 = load i16*, i16** @sp, align 4
  store i16 32, i16* %0, align 2
  %1 = load i8*, i8** @cp, align 4
  store i8 97, i8* %1, align 1
  %2 = load i16, i16* %s, align 4
  %3 = load i8, i8* %c, align 4
  %conv.i = sext i16 %2 to i32
  %conv1.i = sext i8 %3 to i32
  %call.i = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str, i32 0, i32 0), i32 %conv.i, i32 %conv1.i) nounwind
  ret void
; 16_b-LABEL: test:
; 16_h-LABEL: test:
; 16_b:	sb	${{[0-9]+}}, [[offset1:[0-9]+]](${{[0-9]+}})
; 16_b: lb      ${{[0-9]+}}, [[offset1]](${{[0-9]+}})
; 16_h:	sh	${{[0-9]+}}, [[offset2:[0-9]+]](${{[0-9]+}})
; 16_h: lh      ${{[0-9]+}}, [[offset2]](${{[0-9]+}})
}

declare void @llvm.lifetime.start(i64, i8* nocapture) nounwind

declare void @llvm.lifetime.end(i64, i8* nocapture) nounwind

