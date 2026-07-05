; RUN: llc -mtriple x86_64-pc-windows-msvc -o - %s | FileCheck %s

; struct S { int x; };
; void foo() {
;   struct S __declspec(align(32)) o;
;   __try { o.x; }
;   __finally { o.x; }
; }
; void bar() {
;   struct S o;
;   __try { o.x; }
;   __finally { o.x; }
; }

%struct.S = type { i32 }

define dso_local void @"?foo@@YAXXZ"() #0 {
entry:
; CHECK-LABEL: foo
; CHECK: movq  %rsp, %rdx
; CHECK-NOT: movq  %rbp, %rdx

  %o = alloca %struct.S, align 32
  call void (...) @llvm.localescape(ptr %o)
  %0 = call ptr @llvm.localaddress()
  call void @"?fin$0@0@foo@@"(i8 0, ptr %0)
  ret void
}

; void bar(void)
; {
;   int x;
;   void (*fn)(int);
;
;   __try {
;     x = 1;
;     fn(x);
;   } __finally {
;     x = 2;
;   }
; }

define dso_local void @"?bar@@YAXXZ"() personality ptr @__C_specific_handler {
entry:
; CHECK-LABEL: bar
; CHECK: movq  %rbp, %rdx
; CHECK-NOT: movq  %rsp, %rdx
  %x = alloca i32, align 4
  %fn = alloca ptr, align 8
  call void (...) @llvm.localescape(ptr %x)
  store i32 1, ptr %x, align 4
  %0 = load ptr, ptr %fn, align 8
  %1 = load i32, ptr %x, align 4
  invoke void %0(i32 %1)
  to label %invoke.cont unwind label %ehcleanup
  invoke.cont:                                      ; preds = %entry
  %2 = call ptr @llvm.localaddress()
  call void @"?fin$0@0@bar@@"(i8 0, ptr %2)
  ret void
  ehcleanup:                                        ; preds = %entry
  %3 = cleanuppad within none []
  %4 = call ptr @llvm.localaddress()
  call void @"?fin$0@0@bar@@"(i8 1, ptr %4) [ "funclet"(token %3) ]
  cleanupret from %3 unwind to caller
}

declare void @"?fin$0@0@foo@@"(i8 %abnormal_termination, ptr %frame_pointer)

declare void @"?fin$0@0@bar@@"(i8 %abnormal_termination, ptr %frame_pointer)

declare ptr @llvm.localrecover(ptr, ptr, i32)

declare ptr @llvm.localaddress()

declare void @llvm.localescape(...)

declare dso_local i32 @__C_specific_handler(...)
