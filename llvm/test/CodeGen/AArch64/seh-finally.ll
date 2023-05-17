; RUN: llc -mtriple arm64-windows -o - %s | FileCheck %s

; struct S { int x; };
; void foo(int n);
; void foo(struct S o);
; void simple_seh() {
;   struct S o;
; 
;   __try { foo(o.x); }
;   __finally { foo(o.x); }
; }
; void stack_realign() {
;   struct S __declspec(align(32)) o;
; 
;   __try { foo(o.x); }
;   __finally { foo(o.x); }
; }
; void vla_present(int n) {
;   int vla[n];
; 
;   __try { foo(n); }
;   __finally { foo(n); }
; }
; void vla_and_realign(int n) {
;   struct S __declspec(align(32)) o;
;   int vla[n];
; 
;   __try { foo(o.x); }
;   __finally { foo(o.x); }
; }

%struct.S = type { i32 }

; Test simple SEH (__try/__finally).
define void @simple_seh() #0 personality ptr @__C_specific_handler {
entry:
; CHECK-LABEL: simple_seh
; CHECK: add     x29, sp, #16
; CHECK: mov     x0, #-2
; CHECK: stur    x0, [x29, #16]
; CHECK: .set .Lsimple_seh$frame_escape_0, -8
; CHECK: ldur    w0, [x29, #-8]
; CHECK: bl      foo

  %o = alloca %struct.S, align 8
  call void (...) @llvm.localescape(ptr %o)
  %0 = load i32, ptr %o, align 4
  invoke void @foo(i32 %0) #5
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  %1 = call ptr @llvm.localaddress()
  call void @fin_simple_seh(i8 0, ptr %1)
  ret void

ehcleanup:                                        ; preds = %entry
  %2 = cleanuppad within none []
  %3 = call ptr @llvm.localaddress()
  call void @fin_simple_seh(i8 1, ptr %3) [ "funclet"(token %2) ]
  cleanupret from %2 unwind to caller
}

define void @fin_simple_seh(i8 %abnormal_termination, ptr %frame_pointer) {
entry:
; CHECK-LABEL: fin_simple_seh
; CHECK: movz    x8, #:abs_g1_s:.Lsimple_seh$frame_escape_0
; CHECK: movk    x8, #:abs_g0_nc:.Lsimple_seh$frame_escape_0
; CHECK: strb    w0, [sp, #15]
; CHECK: ldr     w8, [x1, x8]
; CHECK: bl      foo

  %frame_pointer.addr = alloca ptr, align 8
  %abnormal_termination.addr = alloca i8, align 1
  %0 = call ptr @llvm.localrecover(ptr @simple_seh, ptr %frame_pointer, i32 0)
  store ptr %frame_pointer, ptr %frame_pointer.addr, align 8
  store i8 %abnormal_termination, ptr %abnormal_termination.addr, align 1
  %1 = load i32, ptr %0, align 4
  call void @foo(i32 %1)
  ret void
}

; Test SEH when stack realignment is needed in case highly aligned stack objects are present.
define void @stack_realign() #0 personality ptr @__C_specific_handler {
entry:
; CHECK-LABEL: stack_realign
; CHECK: add     x29, sp, #8
; CHECK: sub     x9, sp, #16
; CHECK: and     sp, x9, #0xffffffffffffffe0
; CHECK: mov     x19, sp
; CHECK: mov     x0, #-2
; CHECK: stur    x0, [x29, #24]
; CHECK: .set .Lstack_realign$frame_escape_0, 0
; CHECK: ldr     w0, [x19]
; CHECK: bl      foo

  %o = alloca %struct.S, align 32
  call void (...) @llvm.localescape(ptr %o)
  %0 = load i32, ptr %o, align 32
  invoke void @foo(i32 %0) #5
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  %1 = call ptr @llvm.localaddress()
  call void @fin_stack_realign(i8 0, ptr %1)
  ret void

ehcleanup:                                        ; preds = %entry
  %2 = cleanuppad within none []
  %3 = call ptr @llvm.localaddress()
  call void @fin_stack_realign(i8 1, ptr %3) [ "funclet"(token %2) ]
  cleanupret from %2 unwind to caller
}

define void @fin_stack_realign(i8 %abnormal_termination, ptr %frame_pointer) {
entry:
; CHECK-LABEL: fin_stack_realign
; CHECK: movz    x8, #:abs_g1_s:.Lstack_realign$frame_escape_0
; CHECK: movk    x8, #:abs_g0_nc:.Lstack_realign$frame_escape_0
; CHECK: strb    w0, [sp, #15]
; CHECK: ldr     w8, [x1, x8]
; CHECK: bl      foo

  %frame_pointer.addr = alloca ptr, align 8
  %abnormal_termination.addr = alloca i8, align 1
  %0 = call ptr @llvm.localrecover(ptr @stack_realign, ptr %frame_pointer, i32 0)
  store ptr %frame_pointer, ptr %frame_pointer.addr, align 8
  store i8 %abnormal_termination, ptr %abnormal_termination.addr, align 1
  %1 = load i32, ptr %0, align 32
  call void @foo(i32 %1)
  ret void
}

; Test SEH when variable size objects are present on the stack. Note: Escaped vla's are current not supported by SEH.
define void @vla_present(i32 %n) #0 personality ptr @__C_specific_handler {
entry:
; CHECK-LABEL: vla_present
; CHECK: add     x29, sp, #32
; CHECK: mov     x1, #-2
; CHECK: stur    x1, [x29, #16]
; CHECK: .set .Lvla_present$frame_escape_0, -4
; CHECK: stur    w0, [x29, #-4]
; CHECK: ldur    w8, [x29, #-4]
; CHECK: mov     x9, sp
; CHECK: stur    x9, [x29, #-16]
; CHECK: stur    x8, [x29, #-24]
; CHECK: ldur    w0, [x29, #-4]
; CHECK: bl      foo

  %n.addr = alloca i32, align 4
  %saved_stack = alloca ptr, align 8
  %__vla_expr0 = alloca i64, align 8
  call void (...) @llvm.localescape(ptr %n.addr)
  store i32 %n, ptr %n.addr, align 4
  %0 = load i32, ptr %n.addr, align 4
  %1 = zext i32 %0 to i64
  %2 = call ptr @llvm.stacksave()
  store ptr %2, ptr %saved_stack, align 8
  %vla = alloca i32, i64 %1, align 4
  store i64 %1, ptr %__vla_expr0, align 8
  %3 = load i32, ptr %n.addr, align 4
  invoke void @foo(i32 %3) #5
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  %4 = call ptr @llvm.localaddress()
  call void @fin_vla_present(i8 0, ptr %4)
  %5 = load ptr, ptr %saved_stack, align 8
  call void @llvm.stackrestore(ptr %5)
  ret void

ehcleanup:                                        ; preds = %entry
  %6 = cleanuppad within none []
  %7 = call ptr @llvm.localaddress()
  call void @fin_vla_present(i8 1, ptr %7) [ "funclet"(token %6) ]
  cleanupret from %6 unwind to caller
}

define void @fin_vla_present(i8 %abnormal_termination, ptr %frame_pointer) {
entry:
; CHECK-LABEL: fin_vla_present
; CHECK: movz    x8, #:abs_g1_s:.Lvla_present$frame_escape_0
; CHECK: movk    x8, #:abs_g0_nc:.Lvla_present$frame_escape_0
; CHECK: strb    w0, [sp, #15]
; CHECK: ldr     w8, [x1, x8]
; CHECK: bl      foo

  %frame_pointer.addr = alloca ptr, align 8
  %abnormal_termination.addr = alloca i8, align 1
  %0 = call ptr @llvm.localrecover(ptr @vla_present, ptr %frame_pointer, i32 0)
  store ptr %frame_pointer, ptr %frame_pointer.addr, align 8
  store i8 %abnormal_termination, ptr %abnormal_termination.addr, align 1
  %1 = load i32, ptr %0, align 4
  call void @foo(i32 %1)
  ret void
}

; Test when both vla's and highly aligned objects are present on stack.
define void @vla_and_realign(i32 %n) #0 personality ptr @__C_specific_handler {
entry:
; CHECK-LABEL: vla_and_realign
; CHECK: add     x29, sp, #8
; CHECK: sub     x9, sp, #48
; CHECK: and     sp, x9, #0xffffffffffffffe0
; CHECK: mov     x19, sp
; CHECK: mov     x1, #-2
; CHECK: stur    x1, [x29, #24]
; CHECK: .set .Lvla_and_realign$frame_escape_0, 32
; CHECK: str     w0, [x29, #36]
; CHECK: ldr     w8, [x29, #36]
; CHECK: mov     x9, sp
; CHECK: str     x9, [x29, #16]
; CHECK: str     x8, [x19, #24]
; CHECK: ldr     w0, [x19, #32]
; CHECK: bl      foo

  %n.addr = alloca i32, align 4
  %o = alloca %struct.S, align 32
  %saved_stack = alloca ptr, align 8
  %__vla_expr0 = alloca i64, align 8
  call void (...) @llvm.localescape(ptr %o)
  store i32 %n, ptr %n.addr, align 4
  %0 = load i32, ptr %n.addr, align 4
  %1 = zext i32 %0 to i64
  %2 = call ptr @llvm.stacksave()
  store ptr %2, ptr %saved_stack, align 8
  %vla = alloca i32, i64 %1, align 4
  store i64 %1, ptr %__vla_expr0, align 8
  %3 = load i32, ptr %o, align 32
  invoke void @foo(i32 %3) #5
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  %4 = call ptr @llvm.localaddress()
  call void @fin_vla_and_realign(i8 0, ptr %4)
  %5 = load ptr, ptr %saved_stack, align 8
  call void @llvm.stackrestore(ptr %5)
  ret void

ehcleanup:                                        ; preds = %entry
  %6 = cleanuppad within none []
  %7 = call ptr @llvm.localaddress()
  call void @fin_vla_and_realign(i8 1, ptr %7) [ "funclet"(token %6) ]
  cleanupret from %6 unwind to caller
}

define void @fin_vla_and_realign(i8 %abnormal_termination, ptr %frame_pointer) {
entry:
; CHECK-LABEL: fin_vla_and_realign
; CHECK: movz    x8, #:abs_g1_s:.Lvla_and_realign$frame_escape_0
; CHECK: movk    x8, #:abs_g0_nc:.Lvla_and_realign$frame_escape_0
; CHECK: strb    w0, [sp, #15]
; CHECK: ldr     w8, [x1, x8]
; CHECK: bl      foo

  %frame_pointer.addr = alloca ptr, align 8
  %abnormal_termination.addr = alloca i8, align 1
  %0 = call ptr @llvm.localrecover(ptr @vla_and_realign, ptr %frame_pointer, i32 0)
  store ptr %frame_pointer, ptr %frame_pointer.addr, align 8
  store i8 %abnormal_termination, ptr %abnormal_termination.addr, align 1
  %1 = load i32, ptr %0, align 32
  call void @foo(i32 %1)
  ret void
}

declare void @foo(i32)
declare void @llvm.stackrestore(ptr)
declare ptr @llvm.stacksave()
declare ptr @llvm.localrecover(ptr, ptr, i32)
declare ptr @llvm.localaddress()
declare void @llvm.localescape(...)
declare i32 @__C_specific_handler(...)

attributes #0 = { noinline optnone }
