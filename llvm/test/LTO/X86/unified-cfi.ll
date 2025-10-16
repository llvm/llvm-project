; Test for the expected CFI codegen in a module with CFI metadata.
; RUN: opt -unified-lto -thinlto-bc -o %t0.o %s
; RUN: llvm-lto --exported-symbol=main -filetype=asm -o - %t0.o | FileCheck %s

; CHECK-LABEL: main

; CHECK: jbe
; CHECK-NEXT: ud2

; ModuleID = 'llvm/test/LTO/X86/unified-cfi.ll'
source_filename = "cfi.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-scei-ps4"

@func = hidden global [3 x i32 ()*] [i32 ()* @_Z1av, i32 ()* @_Z1bv, i32 ()* @_Z1cv], align 16
@.src = private unnamed_addr constant [8 x i8] c"cfi.cpp\00", align 1
@anon.9260195284c792ab5c6ef4d97bfcf95d.0 = private unnamed_addr constant { i16, i16, [9 x i8] } { i16 -1, i16 0, [9 x i8] c"'int ()'\00" }

; Function Attrs: noinline nounwind optnone sspstrong uwtable
define hidden i32 @_Z1av() #0 !type !3 !type !4 {
entry:
  ret i32 1
}

; Function Attrs: noinline nounwind optnone sspstrong uwtable
define hidden i32 @_Z1bv() #0 !type !3 !type !4 {
entry:
  ret i32 2
}

; Function Attrs: noinline nounwind optnone sspstrong uwtable
define hidden i32 @_Z1cv() #0 !type !3 !type !4 {
entry:
  ret i32 3
}

; Function Attrs: noinline norecurse nounwind optnone sspstrong uwtable
define hidden i32 @main(i32 %argc, i8** %argv) #1 !type !5 !type !6 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4
  store i8** %argv, i8*** %argv.addr, align 8
  %0 = load i32, i32* %argc.addr, align 4
  %idxprom = sext i32 %0 to i64
  %arrayidx = getelementptr inbounds [3 x i32 ()*], [3 x i32 ()*]* @func, i64 0, i64 %idxprom
  %1 = load i32 ()*, i32 ()** %arrayidx, align 8
  %2 = bitcast i32 ()* %1 to i8*, !nosanitize !7
  %3 = call i1 @llvm.type.test(i8* %2, metadata !"_ZTSFivE"), !nosanitize !7
  br i1 %3, label %cont, label %trap, !nosanitize !7

trap:                                             ; preds = %entry
  call void @llvm.trap() #4, !nosanitize !7
  unreachable, !nosanitize !7

cont:                                             ; preds = %entry
  %call = call i32 %1()
  ret i32 %call
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare i1 @llvm.type.test(i8*, metadata) #2

; Function Attrs: cold noreturn nounwind
declare void @llvm.trap() #3

attributes #0 = { noinline nounwind optnone sspstrong uwtable }
attributes #1 = { noinline norecurse nounwind optnone sspstrong uwtable }
attributes #2 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #3 = { cold noreturn nounwind }
attributes #4 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 2}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{!"clang version 7.0.0 (PS4 clang version 99.99.0.1562 432a534f checking)"}
!3 = !{i64 0, !"_ZTSFivE"}
!4 = !{i64 0, !"_ZTSFivE.generalized"}
!5 = !{i64 0, !"_ZTSFiiPPcE"}
!6 = !{i64 0, !"_ZTSFiiPvE.generalized"}
!7 = !{}
