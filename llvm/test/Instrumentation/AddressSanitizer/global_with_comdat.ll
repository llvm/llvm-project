; Check that we disable the use of comdat sections for the garbage collection
; of globals.
; This is to avoid false negative ODR violations detection at link time.
; We keep using comdats for garbage collection if odr indicators are
; enabled as indicator symbols will cause link time odr violations.
; This is to fix PR 47925.
;
; RUN: opt < %s -passes=asan -asan-globals-live-support=1 -asan-use-odr-indicator=0 -S | FileCheck %s --check-prefixes=CHECK,NOCOMDAT
; Check that enabling odr indicators enables comdat for globals.
; RUN: opt < %s -passes=asan -asan-globals-live-support=1 -S | FileCheck %s --check-prefixes=CHECK,COMDAT
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

; COMDAT: $global = comdat any
; COMDAT: $dyn_init_global = comdat any
; COMDAT: $_ZZ4funcvE10static_var.{{[01-9a-f]+}} = comdat any
; COMDAT: $.str.{{[01-9a-f]+}} = comdat any

; NOCOMDAT-NOT: $global = comdat any
; NOCOMDAT-NOT: $dyn_init_global = comdat any
; NOCOMDAT-NOT: $_ZZ4funcvE10static_var.{{[01-9a-f]+}} = comdat any
; NOCOMDAT-NOT: $.str.{{[01-9a-f]+}} = comdat any

; COMDAT: @global = global { i32, [28 x i8] } zeroinitializer, comdat, align 32
; COMDAT: @dyn_init_global = global { i32, [28 x i8] } zeroinitializer, comdat, align 32
; COMDAT: @_ZZ4funcvE10static_var = internal global { i32, [28 x i8] } zeroinitializer, comdat($_ZZ4funcvE10static_var.{{[01-9a-f]+}}), align 32
; COMDAT: @.str = internal constant { [14 x i8], [18 x i8] } { [14 x i8] c"Hello, world!\00", [18 x i8] zeroinitializer }, comdat($.str.{{[01-9a-f]+}}), align 32

; NOCOMDAT: @global = global { i32, [28 x i8] } zeroinitializer, align 32
; NOCOMDAT: @dyn_init_global = global { i32, [28 x i8] } zeroinitializer, align 32
; NOCOMDAT: @_ZZ4funcvE10static_var = internal global { i32, [28 x i8] } zeroinitializer, align 32
; NOCOMDAT: @.str = internal constant { [14 x i8], [18 x i8] } { [14 x i8] c"Hello, world!\00", [18 x i8] zeroinitializer }, align 32

; Check emitted location descriptions:
; CHECK: [[VARNAME:@___asan_gen_.[0-9]+]] = private unnamed_addr constant [7 x i8] c"global\00", align 1
; COMDAT: @__asan_global_global = {{.*}}i64 ptrtoint (ptr @__odr_asan_gen_global to i64){{.*}} section "asan_globals"{{.*}}, comdat($global), !associated
; COMDAT: @__asan_global_.str = {{.*}}i64 ptrtoint (ptr @___asan_gen_ to i64){{.*}} section "asan_globals"{{.*}}, comdat($.str.{{.*}}), !associated

; The metadata has to be inserted to llvm.compiler.used to avoid being stripped
; during LTO.
; CHECK: @llvm.compiler.used {{.*}} ptr @__asan_global_global, {{.*}} section "llvm.metadata"

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
