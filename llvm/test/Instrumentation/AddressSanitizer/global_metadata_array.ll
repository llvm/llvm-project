; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -asan-globals-live-support=0 -mtriple=x86_64-unknown-linux-gnu -S | FileCheck --check-prefix=CHECK %s
; RUN: opt < %s -passes='asan-pipeline' -asan-globals-live-support=0 -mtriple=x86_64-unknown-linux-gnu -S | FileCheck --check-prefix=CHECK %s
; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -asan-globals-live-support=0 -mtriple=x86_64-apple-macosx10.11.0 -S | FileCheck --check-prefix=CHECK %s
; RUN: opt < %s -passes='asan-pipeline' -asan-globals-live-support=0 -mtriple=x86_64-apple-macosx10.11.0 -S | FileCheck --check-prefix=CHECK %s
; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -asan-globals-live-support=0 -mtriple=x86_64-pc-windows-msvc19.0.24215 -S | FileCheck --check-prefix=CHECK %s
; RUN: opt < %s -passes='asan-pipeline' -asan-globals-live-support=0 -mtriple=x86_64-pc-windows-msvc19.0.24215 -S | FileCheck --check-prefix=CHECK %s
; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -asan-globals-live-support=0 -asan-mapping-scale=5 -mtriple=x86_64-unknown-linux-gnu -S | FileCheck --check-prefixes=CHECK,CHECK-S5 %s
; RUN: opt < %s -passes='asan-pipeline' -asan-globals-live-support=0 -asan-mapping-scale=5 -mtriple=x86_64-unknown-linux-gnu -S | FileCheck --check-prefixes=CHECK,CHECK-S5 %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Globals:
@global = global i32 0, align 4
@dyn_init_global = global i32 0, align 4
@blacklisted_global = global i32 0, align 4
@_ZZ4funcvE10static_var = internal global i32 0, align 4
@.str = private unnamed_addr constant [14 x i8] c"Hello, world!\00", align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_asan_globals.cpp, i8* null }]

; Check that globals were instrumented:
; CHECK: @global = global { i32, [60 x i8] } zeroinitializer, align 32
; CHECK: @.str = internal constant { [14 x i8], [50 x i8] } { [14 x i8] c"Hello, world!\00", [50 x i8] zeroinitializer }{{.*}}, align 32

; Check emitted location descriptions:
; CHECK: [[VARNAME:@___asan_gen_.[0-9]+]] = private unnamed_addr constant [7 x i8] c"global\00", align 1
; CHECK: [[FILENAME:@___asan_gen_.[0-9]+]] = private unnamed_addr constant [22 x i8] c"/tmp/asan-globals.cpp\00", align 1
; CHECK: [[LOCDESCR:@___asan_gen_.[0-9]+]] = private unnamed_addr constant { [22 x i8]*, i32, i32 } { [22 x i8]* [[FILENAME]], i32 5, i32 5 }

; Check that location descriptors and global names were passed into __asan_register_globals:
; CHECK: i64 ptrtoint ([7 x i8]* [[VARNAME]] to i64)
; CHECK: i64 ptrtoint ({ [22 x i8]*, i32, i32 }* [[LOCDESCR]] to i64)
; Check alignment of metadata_array.
; CHECK-S5-SAME: {{align 32$}}

; Function Attrs: nounwind sanitize_address
define internal void @__cxx_global_var_init() #0 section ".text.startup" {
entry:
  %0 = load i32, i32* @global, align 4
  store i32 %0, i32* @dyn_init_global, align 4
  ret void
}

; Function Attrs: nounwind sanitize_address
define void @_Z4funcv() #1 {
entry:
  %literal = alloca i8*, align 8
  store i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str, i32 0, i32 0), i8** %literal, align 8
  ret void
}

; Function Attrs: nounwind sanitize_address
define internal void @_GLOBAL__sub_I_asan_globals.cpp() #0 section ".text.startup" {
entry:
  call void @__cxx_global_var_init()
  ret void
}

attributes #0 = { nounwind sanitize_address }
attributes #1 = { nounwind sanitize_address "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.asan.globals = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32* @global, !6, !"global", i1 false, i1 false}
!1 = !{i32* @dyn_init_global, !7, !"dyn_init_global", i1 true, i1 false}
!2 = !{i32* @blacklisted_global, null, null, i1 false, i1 true}
!3 = !{i32* @_ZZ4funcvE10static_var, !8, !"static_var", i1 false, i1 false}
!4 = !{[14 x i8]* @.str, !9, !"<string literal>", i1 false, i1 false}

!5 = !{!"clang version 3.5.0 (211282)"}

!6 = !{!"/tmp/asan-globals.cpp", i32 5, i32 5}
!7 = !{!"/tmp/asan-globals.cpp", i32 7, i32 5}
!8 = !{!"/tmp/asan-globals.cpp", i32 12, i32 14}
!9 = !{!"/tmp/asan-globals.cpp", i32 14, i32 25}
