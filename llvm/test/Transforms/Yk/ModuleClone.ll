; RUN: llc -stop-after=yk-basicblock-tracer-pass --yk-module-clone --yk-basicblock-tracer < %s  | FileCheck %s

source_filename = "ModuleClone.c"
target triple = "x86_64-pc-linux-gnu"

@.str = private unnamed_addr constant [13 x i8] c"Hello, world\00", align 1
@my_global = global i32 42, align 4

declare i32 @printf(ptr, ...)
declare dso_local ptr @yk_mt_new(ptr noundef)
declare dso_local void @yk_mt_hot_threshold_set(ptr noundef, i32 noundef)
declare dso_local i64 @yk_location_new()
declare dso_local void @yk_mt_control_point(ptr noundef, ptr noundef)
declare dso_local i32 @fprintf(ptr noundef, ptr noundef, ...)
declare dso_local void @yk_location_drop(i64)
declare dso_local void @yk_mt_shutdown(ptr noundef)

define dso_local i32 @func_inc_with_address_taken(i32 %x) {
entry:
  %0 = add i32 %x, 1
  ret i32 %0
}

define dso_local i32 @my_func(i32 %x) {
entry:
  %0 = add i32 %x, 1
  %func_ptr = alloca ptr, align 8
  store ptr @func_inc_with_address_taken, ptr %func_ptr, align 8
  %1 = load ptr, ptr %func_ptr, align 8
  %2 = call i32 %1(i32 42)
  ret i32 %2
}

define dso_local i32 @main() {
entry:
  %0 = call i32 @my_func(i32 10)
  %1 = load i32, ptr @my_global
  %2 = call i32 (ptr, ...) @printf(ptr @.str, i32 %1)
  ret i32 0
}

; ======================================================================
; Original functions - should have trace calls
; ======================================================================
; File header checks
; CHECK: source_filename = "ModuleClone.c"
; CHECK: target triple = "x86_64-pc-linux-gnu"

; Global variable and string checks
; CHECK: @.str = private unnamed_addr constant [13 x i8] c"Hello, world\00", align 1
; CHECK: @my_global = global i32 42, align 4

; Declaration checks
; CHECK: declare i32 @printf(ptr, ...)
; CHECK: declare dso_local ptr @yk_mt_new(ptr noundef)
; CHECK: declare dso_local void @yk_mt_hot_threshold_set(ptr noundef, i32 noundef)
; CHECK: declare dso_local i64 @yk_location_new()
; CHECK: declare dso_local void @yk_mt_control_point(ptr noundef, ptr noundef)
; CHECK: declare dso_local i32 @fprintf(ptr noundef, ptr noundef, ...)
; CHECK: declare dso_local void @yk_location_drop(i64)
; CHECK: declare dso_local void @yk_mt_shutdown(ptr noundef)

; Check original function: my_func
; CHECK-LABEL: define dso_local i32 @my_func(i32 %x)
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @__yk_trace_basicblock({{.*}})
; CHECK-NEXT: %0 = add i32 %x, 1
; CHECK-NEXT: %func_ptr = alloca ptr, align 8
; CHECK-NEXT: store ptr @func_inc_with_address_taken, ptr %func_ptr, align 8
; CHECK-NEXT: %1 = load ptr, ptr %func_ptr, align 8
; CHECK-NEXT: %2 = call i32 %1(i32 42)
; CHECK-NEXT: ret i32 %2

; Check original function: main
; CHECK-LABEL: define dso_local i32 @main()
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @__yk_trace_basicblock({{.*}})
; CHECK-NEXT: %0 = call i32 @my_func(i32 10)
; CHECK-NEXT: %1 = load i32, ptr @my_global
; CHECK-NEXT: %2 = call i32 (ptr, ...) @printf

; Check that func_inc_with_address_taken is present in its original form
; CHECK-LABEL: define dso_local i32 @func_inc_with_address_taken(i32 %x)
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @__yk_trace_basicblock({{.*}})
; CHECK-NEXT: %0 = add i32 %x, 1
; CHECK-NEXT: ret i32 %0

; ======================================================================
; Functions whose addresses are taken should not be cloned
; ======================================================================
; Functions with their addresses taken should not be cloned. 
; `func_inc_with_address_taken` is used by pointer and thus remains unaltered.
; CHECK-NOT: define dso_local i32 @__yk_clone_func_inc_with_address_taken

; ======================================================================
; Cloned functions - should have no trace calls
; ======================================================================
; Check cloned function: __yk_clone_my_func
; CHECK-LABEL: define dso_local i32 @__yk_clone_my_func(i32 %x)
; CHECK-NEXT: entry:
; CHECK-NOT: call void @__yk_trace_basicblock({{.*}})
; CHECK-NEXT: %0 = add i32 %x, 1
; CHECK-NEXT: %func_ptr = alloca ptr, align 8
; CHECK-NEXT: store ptr @func_inc_with_address_taken, ptr %func_ptr, align 8
; CHECK-NEXT: %1 = load ptr, ptr %func_ptr, align 8
; CHECK-NEXT: %2 = call i32 %1(i32 42)
; CHECK-NEXT: ret i32 %2

; Check cloned function: __yk_clone_main
; CHECK-LABEL: define dso_local i32 @__yk_clone_main()
; CHECK-NEXT: entry:
; CHECK-NOT: call void @__yk_trace_basicblock({{.*}})
; CHECK-NEXT: %0 = call i32 @__yk_clone_my_func(i32 10)
; CHECK-NEXT: %1 = load i32, ptr @my_global
; CHECK-NEXT: %2 = call i32 (ptr, ...) @printf
