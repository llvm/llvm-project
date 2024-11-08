; RUN: rm -rf %t && split-file %s %t
; RUN: opt < %t/a.ll -passes=asan -asan-globals-live-support=0 -mtriple=x86_64-unknown-linux-gnu -S | FileCheck --check-prefix=CHECK %s
; RUN: opt < %t/a.ll -passes=asan -asan-globals-live-support=0 -mtriple=x86_64-apple-macosx10.11.0 -S | FileCheck --check-prefix=CHECK %s
; RUN: opt < %t/a.ll -passes=asan -asan-globals-live-support=0 -mtriple=x86_64-pc-windows-msvc19.0.24215 -S | FileCheck --check-prefix=CHECK %s
; RUN: opt < %t/a.ll -passes=asan -asan-globals-live-support=0 -asan-mapping-scale=5 -mtriple=x86_64-unknown-linux-gnu -S | FileCheck --check-prefixes=CHECK,CHECK-S5 %s

; RUN: opt < %t/empty.ll -passes=asan -asan-globals-live-support=0 -mtriple=x86_64-unknown-linux-gnu -S | FileCheck --check-prefix=ELF-NOGC %s

;--- a.ll
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Globals:
@global = global i32 0, align 4
@dyn_init_global = global i32 0, align 4
@blocked_global = global i32 0, align 4
@_ZZ4funcvE10static_var = internal global i32 0, align 4
@.str = private unnamed_addr constant [14 x i8] c"Hello, world!\00", align 1
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_asan_globals.cpp, ptr null }]

; Check that globals were instrumented:
; CHECK: @global = global { i32, [28 x i8] } zeroinitializer, align 32
; CHECK: @.str = internal constant { [14 x i8], [18 x i8] } { [14 x i8] c"Hello, world!\00", [18 x i8] zeroinitializer }{{.*}}, align 32

; Check emitted location descriptions:
; CHECK: [[VARNAME:@___asan_gen_global[.0-9]*]] = private unnamed_addr constant [7 x i8] c"global\00", align 1

; Check that location descriptors and global names were passed into __asan_register_globals:
; CHECK: i64 ptrtoint (ptr [[VARNAME]] to i64)
; Check alignment of metadata_array.
; CHECK-S5-SAME: {{align 32$}}

; Function Attrs: nounwind sanitize_address
define internal void @__cxx_global_var_init() #0 section ".text.startup" {
entry:
  %0 = load i32, ptr @global, align 4
  store i32 %0, ptr @dyn_init_global, align 4
  ret void
}

; Function Attrs: nounwind sanitize_address
define void @_Z4funcv() #1 {
entry:
  %literal = alloca ptr, align 8
  store ptr @.str, ptr %literal, align 8
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

!llvm.ident = !{!5}

!5 = !{!"clang version 3.5.0 (211282)"}

!6 = !{!"/tmp/asan-globals.cpp", i32 5, i32 5}
!7 = !{!"/tmp/asan-globals.cpp", i32 7, i32 5}
!8 = !{!"/tmp/asan-globals.cpp", i32 12, i32 14}
!9 = !{!"/tmp/asan-globals.cpp", i32 14, i32 25}

;; In the presence of instrumented global variables, asan.module_ctor do not use comdat.
; CHECK: define internal void @asan.module_ctor() #[[#ATTR:]] {

; ELF-NOGC: define internal void @asan.module_ctor() #[[#ATTR:]] comdat {

;--- empty.ll
