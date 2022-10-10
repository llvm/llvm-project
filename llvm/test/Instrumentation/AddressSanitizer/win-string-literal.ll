; RUN: opt < %s -passes=asan -S | FileCheck %s

; Generated like so:
; $ clang -S -emit-llvm -Xclang -disable-llvm-passes -fsanitize=address -O1 t.cpp -o t.ll
; const char *getstr() { return "asdf"; }

; CHECK: $"??_C@_04JIHMPGLA@asdf?$AA@" = comdat any

; CHECK: @"??_C@_04JIHMPGLA@asdf?$AA@" =
; CHECK-SAME: linkonce_odr dso_local constant { [5 x i8], [27 x i8] }
; CHECK-SAME: { [5 x i8] c"asdf\00", [27 x i8] zeroinitializer }, comdat, align 32

; CHECK:      @"__asan_global_??_C@_04JIHMPGLA@asdf?$AA@" =
; CHECK-SAME: private global { i64, i64, i64, i64, i64, i64, i64, i64 }
; CHECK-SAME: { i64 ptrtoint ({ [5 x i8], [27 x i8] }* @"??_C@_04JIHMPGLA@asdf?$AA@" to i64),
; CHECK-SAME:   i64 5, i64 32, i64 ptrtoint ([7 x i8]* @___asan_gen_.1 to i64), i64 ptrtoint ([8
; CHECK-SAME:   x i8]* @___asan_gen_ to i64), i64 0, i64 0, i64 0 }, section ".ASAN$GL",
; CHECK-SAME:   comdat($"??_C@_04JIHMPGLA@asdf?$AA@"), align 64

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.13.26131"

$"??_C@_04JIHMPGLA@asdf?$AA@" = comdat any

@"??_C@_04JIHMPGLA@asdf?$AA@" = linkonce_odr dso_local unnamed_addr constant [5 x i8] c"asdf\00", comdat, align 1

; Function Attrs: nounwind sanitize_address uwtable
define dso_local i8* @"?getstr@@YAPEBDXZ"() #0 {
entry:
  ret i8* getelementptr inbounds ([5 x i8], [5 x i8]* @"??_C@_04JIHMPGLA@asdf?$AA@", i32 0, i32 0)
}

attributes #0 = { nounwind sanitize_address uwtable }

!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!4}

!1 = !{!"t.cpp", i32 1, i32 31}
!2 = !{i32 1, !"wchar_size", i32 2}
!3 = !{i32 7, !"PIC Level", i32 2}
!4 = !{!"clang version 7.0.0 "}
