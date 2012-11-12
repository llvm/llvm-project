; RUN: llc  -march=mipsel -mcpu=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16_h
; RUN: llc  -march=mipsel -mcpu=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16_b

@.str = private unnamed_addr constant [9 x i8] c"%hd %c \0A\00", align 1
@sp = common global i16* null, align 4
@cp = common global i8* null, align 4

define void @p1(i16 signext %s, i8 signext %c) nounwind {
entry:
  %conv = sext i16 %s to i32
  %conv1 = sext i8 %c to i32
  %call = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0), i32 %conv, i32 %conv1) nounwind
  ret void
}

declare i32 @printf(i8* nocapture, ...) nounwind

define void @p2() nounwind {
entry:
  %0 = load i16** @sp, align 4
  %1 = load i16* %0, align 2
  %2 = load i8** @cp, align 4
  %3 = load i8* %2, align 1
  %conv.i = sext i16 %1 to i32
  %conv1.i = sext i8 %3 to i32
  %call.i = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0), i32 %conv.i, i32 %conv1.i) nounwind
  %4 = load i16** @sp, align 4
  store i16 32, i16* %4, align 2
  %5 = load i8** @cp, align 4
  store i8 97, i8* %5, align 1
  ret void
}

define void @test() nounwind {
entry:
  %s = alloca i16, align 4
  %c = alloca i8, align 4
  store i16 16, i16* %s, align 4
  store i8 99, i8* %c, align 4
  store i16* %s, i16** @sp, align 4
  store i8* %c, i8** @cp, align 4
  %call.i.i = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0), i32 16, i32 99) nounwind
  %0 = load i16** @sp, align 4
  store i16 32, i16* %0, align 2
  %1 = load i8** @cp, align 4
  store i8 97, i8* %1, align 1
  %2 = load i16* %s, align 4
  %3 = load i8* %c, align 4
  %conv.i = sext i16 %2 to i32
  %conv1.i = sext i8 %3 to i32
  %call.i = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0), i32 %conv.i, i32 %conv1.i) nounwind
  ret void
; 16_b: test:
; 16_h: test:
; 16_b:	sb	${{[0-9]+}}, [[offset1:[0-9]+]](${{[0-9]+}})
; 16_b: lb      ${{[0-9]+}}, [[offset1]](${{[0-9]+}})
; 16_h:	sh	${{[0-9]+}}, [[offset2:[0-9]+]](${{[0-9]+}})
; 16_h: lh      ${{[0-9]+}}, [[offset2]](${{[0-9]+}})
}

define i32 @main() nounwind {
entry:
  %s.i = alloca i16, align 4
  %c.i = alloca i8, align 4
  %0 = bitcast i16* %s.i to i8*
  call void @llvm.lifetime.start(i64 -1, i8* %0) nounwind
  call void @llvm.lifetime.start(i64 -1, i8* %c.i) nounwind
  store i16 16, i16* %s.i, align 4
  store i8 99, i8* %c.i, align 4
  store i16* %s.i, i16** @sp, align 4
  store i8* %c.i, i8** @cp, align 4
  %call.i.i.i = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0), i32 16, i32 99) nounwind
  %1 = load i16** @sp, align 4
  store i16 32, i16* %1, align 2
  %2 = load i8** @cp, align 4
  store i8 97, i8* %2, align 1
  %3 = load i16* %s.i, align 4
  %4 = load i8* %c.i, align 4
  %conv.i.i = sext i16 %3 to i32
  %conv1.i.i = sext i8 %4 to i32
  %call.i.i = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0), i32 %conv.i.i, i32 %conv1.i.i) nounwind
  call void @llvm.lifetime.end(i64 -1, i8* %0) nounwind
  call void @llvm.lifetime.end(i64 -1, i8* %c.i) nounwind
  ret i32 0
}

declare void @llvm.lifetime.start(i64, i8* nocapture) nounwind

declare void @llvm.lifetime.end(i64, i8* nocapture) nounwind

