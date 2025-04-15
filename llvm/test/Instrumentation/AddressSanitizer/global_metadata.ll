; RUN: opt < %s -passes=asan             -asan-use-private-alias=0 -asan-globals-live-support=1 -S | FileCheck %s --check-prefixes=CHECK,NOALIAS

; RUN: opt < %s -passes=asan             -asan-use-private-alias=1 -asan-globals-live-support=1 -S | FileCheck %s --check-prefixes=CHECK,ALIAS
; RUN: opt < %s -passes=asan             -asan-use-private-alias=1 -asan-globals-live-support=1 -asan-mapping-scale=5 -S | FileCheck %s --check-prefixes=CHECK,ALIAS

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Globals:
@global = global i32 0, align 4
@dyn_init_global = global i32 0, align 4
@blocked_global = global i32 0, align 4
@_ZZ4funcvE10static_var = internal global i32 0, align 4
@.str = private unnamed_addr constant [14 x i8] c"Hello, world!\00", align 1
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_asan_globals.cpp, ptr null }]

; Check that globals were instrumented:

; CHECK: @global = global { i32, [28 x i8] } zeroinitializer, comdat, align 32
; CHECK: @.str = internal constant { [14 x i8], [18 x i8] } { [14 x i8] c"Hello, world!\00", [18 x i8] zeroinitializer }, comdat({{.*}}), align 32

; Check emitted location descriptions:
; CHECK: [[VARNAME:@___asan_gen_global[.0-9]*]] = private unnamed_addr constant [7 x i8] c"global\00", align 1
; NOALIAS: @__asan_global_global = {{.*}}i64 ptrtoint (ptr @global to i64){{.*}} section "asan_globals"{{.*}}, !associated
; NOALIAS: @__asan_global_.str = {{.*}}i64 ptrtoint (ptr @{{.str|1}} to i64){{.*}} section "asan_globals"{{.*}}, !associated
; ALIAS: @__asan_global_global = {{.*}}i64 ptrtoint (ptr @0 to i64){{.*}} section "asan_globals"{{.*}}, !associated
; ALIAS: @__asan_global_.str = {{.*}}i64 ptrtoint (ptr @4 to i64){{.*}} section "asan_globals"{{.*}}, !associated

; The metadata has to be inserted to llvm.compiler.used to avoid being stripped
; during LTO.
; CHECK: @llvm.compiler.used = appending global [10 x ptr] [ptr @global, ptr @dyn_init_global, ptr @blocked_global, ptr @_ZZ4funcvE10static_var, ptr @.str, ptr @__asan_global_global, ptr @__asan_global_dyn_init_global, ptr @__asan_global_blocked_global, ptr @__asan_global__ZZ4funcvE10static_var, ptr @__asan_global_.str], section "llvm.metadata"

; Check that location descriptors and global names were passed into __asan_register_globals:
; CHECK: call void @__asan_register_elf_globals(i64 ptrtoint (ptr @___asan_globals_registered to i64), i64 ptrtoint (ptr @__start_asan_globals to i64), i64 ptrtoint (ptr @__stop_asan_globals to i64))

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
