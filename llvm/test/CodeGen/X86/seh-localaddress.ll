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
  call void (...) @llvm.localescape(%struct.S* %o)
  %x = getelementptr inbounds %struct.S, %struct.S* %o, i32 0, i32 0
  %0 = call i8* @llvm.localaddress()
  call void @"?fin$0@0@foo@@"(i8 0, i8* %0)
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

define dso_local void @"?bar@@YAXXZ"() personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*) {
entry:
; CHECK-LABEL: bar
; CHECK: movq  %rbp, %rdx
; CHECK-NOT: movq  %rsp, %rdx
  %x = alloca i32, align 4
  %fn = alloca void (i32)*, align 8
  call void (...) @llvm.localescape(i32* %x)
  store i32 1, i32* %x, align 4
  %0 = load void (i32)*, void (i32)** %fn, align 8
  %1 = load i32, i32* %x, align 4
  invoke void %0(i32 %1)
  to label %invoke.cont unwind label %ehcleanup
  invoke.cont:                                      ; preds = %entry
  %2 = call i8* @llvm.localaddress()
  call void @"?fin$0@0@bar@@"(i8 0, i8* %2)
  ret void
  ehcleanup:                                        ; preds = %entry
  %3 = cleanuppad within none []
  %4 = call i8* @llvm.localaddress()
  call void @"?fin$0@0@bar@@"(i8 1, i8* %4) [ "funclet"(token %3) ]
  cleanupret from %3 unwind to caller
}

declare void @"?fin$0@0@foo@@"(i8 %abnormal_termination, i8* %frame_pointer)

declare void @"?fin$0@0@bar@@"(i8 %abnormal_termination, i8* %frame_pointer)

declare i8* @llvm.localrecover(i8*, i8*, i32)

declare i8* @llvm.localaddress()

declare void @llvm.localescape(...)

declare dso_local i32 @__C_specific_handler(...)
