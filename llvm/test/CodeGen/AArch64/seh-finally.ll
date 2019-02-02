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
define void @simple_seh() #0 personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*) {
entry:
; CHECK-LABEL: simple_seh
; CHECK: add     x29, sp, #16
; CHECK: orr     x1, xzr, #0xfffffffffffffffe
; CHECK: stur    x1, [x29, #-16]
; CHECK: .set .Lsimple_seh$frame_escape_0, -8
; CHECK: ldur    w0, [x29, #-8]
; CHECK: bl      foo

  %o = alloca %struct.S, align 4
  call void (...) @llvm.localescape(%struct.S* %o)
  %x = getelementptr inbounds %struct.S, %struct.S* %o, i32 0, i32 0
  %0 = load i32, i32* %x, align 4
  invoke void @foo(i32 %0) #5
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  %1 = call i8* @llvm.localaddress()
  call void @fin_simple_seh(i8 0, i8* %1)
  ret void

ehcleanup:                                        ; preds = %entry
  %2 = cleanuppad within none []
  %3 = call i8* @llvm.localaddress()
  call void @fin_simple_seh(i8 1, i8* %3) [ "funclet"(token %2) ]
  cleanupret from %2 unwind to caller
}

define void @fin_simple_seh(i8 %abnormal_termination, i8* %frame_pointer) {
entry:
; CHECK-LABEL: fin_simple_seh
; CHECK: movz    x8, #:abs_g1_s:.Lsimple_seh$frame_escape_0
; CHECK: movk    x8, #:abs_g0_nc:.Lsimple_seh$frame_escape_0
; CHECK: strb    w0, [sp, #15]
; CHECK: ldr     w0, [x1, x8]
; CHECK: bl      foo

  %frame_pointer.addr = alloca i8*, align 8
  %abnormal_termination.addr = alloca i8, align 1
  %0 = call i8* @llvm.localrecover(i8* bitcast (void ()* @simple_seh to i8*), i8* %frame_pointer, i32 0)
  %o = bitcast i8* %0 to %struct.S*
  store i8* %frame_pointer, i8** %frame_pointer.addr, align 8
  store i8 %abnormal_termination, i8* %abnormal_termination.addr, align 1
  %x = getelementptr inbounds %struct.S, %struct.S* %o, i32 0, i32 0
  %1 = load i32, i32* %x, align 4
  call void @foo(i32 %1)
  ret void
}

; Test SEH when stack realignment is needed in case highly aligned stack objects are present.
define void @stack_realign() #0 personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*) {
entry:
; CHECK-LABEL: stack_realign
; CHECK: add     x29, sp, #16
; CHECK: sub     x9, sp, #64
; CHECK: and     sp, x9, #0xffffffffffffffe0
; CHECK: mov     x19, sp
; CHECK: orr     x1, xzr, #0xfffffffffffffffe
; CHECK: stur    x1, [x19, #16]
; CHECK: .set .Lstack_realign$frame_escape_0, 32
; CHECK: ldr     w0, [x19, #32]
; CHECK: bl      foo

  %o = alloca %struct.S, align 32
  call void (...) @llvm.localescape(%struct.S* %o)
  %x = getelementptr inbounds %struct.S, %struct.S* %o, i32 0, i32 0
  %0 = load i32, i32* %x, align 32
  invoke void @foo(i32 %0) #5
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  %1 = call i8* @llvm.localaddress()
  call void @fin_stack_realign(i8 0, i8* %1)
  ret void

ehcleanup:                                        ; preds = %entry
  %2 = cleanuppad within none []
  %3 = call i8* @llvm.localaddress()
  call void @fin_stack_realign(i8 1, i8* %3) [ "funclet"(token %2) ]
  cleanupret from %2 unwind to caller
}

define void @fin_stack_realign(i8 %abnormal_termination, i8* %frame_pointer) {
entry:
; CHECK-LABEL: fin_stack_realign
; CHECK: movz    x8, #:abs_g1_s:.Lstack_realign$frame_escape_0
; CHECK: movk    x8, #:abs_g0_nc:.Lstack_realign$frame_escape_0
; CHECK: strb    w0, [sp, #15]
; CHECK: ldr     w0, [x1, x8]
; CHECK: bl      foo

  %frame_pointer.addr = alloca i8*, align 8
  %abnormal_termination.addr = alloca i8, align 1
  %0 = call i8* @llvm.localrecover(i8* bitcast (void ()* @stack_realign to i8*), i8* %frame_pointer, i32 0)
  %o = bitcast i8* %0 to %struct.S*
  store i8* %frame_pointer, i8** %frame_pointer.addr, align 8
  store i8 %abnormal_termination, i8* %abnormal_termination.addr, align 1
  %x = getelementptr inbounds %struct.S, %struct.S* %o, i32 0, i32 0
  %1 = load i32, i32* %x, align 32
  call void @foo(i32 %1)
  ret void
}

; Test SEH when variable size objects are present on the stack. Note: Escaped vla's are current not supported by SEH.
define void @vla_present(i32 %n) #0 personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*) {
entry:
; CHECK-LABEL: vla_present
; CHECK: add     x29, sp, #32
; CHECK: orr     x1, xzr, #0xfffffffffffffffe
; CHECK: stur    x1, [x29, #-32]
; CHECK: .set .Lvla_present$frame_escape_0, -4
; CHECK: stur    w0, [x29, #-4]
; CHECK: ldur    w8, [x29, #-4]
; CHECK: mov     x9, sp
; CHECK: stur    x9, [x29, #-16]
; CHECK: stur    x8, [x29, #-24]
; CHECK: ldur    w0, [x29, #-4]
; CHECK: bl      foo

  %n.addr = alloca i32, align 4
  %saved_stack = alloca i8*, align 8
  %__vla_expr0 = alloca i64, align 8
  call void (...) @llvm.localescape(i32* %n.addr)
  store i32 %n, i32* %n.addr, align 4
  %0 = load i32, i32* %n.addr, align 4
  %1 = zext i32 %0 to i64
  %2 = call i8* @llvm.stacksave()
  store i8* %2, i8** %saved_stack, align 8
  %vla = alloca i32, i64 %1, align 4
  store i64 %1, i64* %__vla_expr0, align 8
  %3 = load i32, i32* %n.addr, align 4
  invoke void @foo(i32 %3) #5
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  %4 = call i8* @llvm.localaddress()
  call void @fin_vla_present(i8 0, i8* %4)
  %5 = load i8*, i8** %saved_stack, align 8
  call void @llvm.stackrestore(i8* %5)
  ret void

ehcleanup:                                        ; preds = %entry
  %6 = cleanuppad within none []
  %7 = call i8* @llvm.localaddress()
  call void @fin_vla_present(i8 1, i8* %7) [ "funclet"(token %6) ]
  cleanupret from %6 unwind to caller
}

define void @fin_vla_present(i8 %abnormal_termination, i8* %frame_pointer) {
entry:
; CHECK-LABEL: fin_vla_present
; CHECK: movz    x8, #:abs_g1_s:.Lvla_present$frame_escape_0
; CHECK: movk    x8, #:abs_g0_nc:.Lvla_present$frame_escape_0
; CHECK: strb    w0, [sp, #15]
; CHECK: ldr     w0, [x1, x8]
; CHECK: bl      foo

  %frame_pointer.addr = alloca i8*, align 8
  %abnormal_termination.addr = alloca i8, align 1
  %0 = call i8* @llvm.localrecover(i8* bitcast (void (i32)* @vla_present to i8*), i8* %frame_pointer, i32 0)
  %n.addr = bitcast i8* %0 to i32*
  store i8* %frame_pointer, i8** %frame_pointer.addr, align 8
  store i8 %abnormal_termination, i8* %abnormal_termination.addr, align 1
  %1 = load i32, i32* %n.addr, align 4
  call void @foo(i32 %1)
  ret void
}

; Test when both vla's and highly aligned objects are present on stack.
define void @vla_and_realign(i32 %n) #0 personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*) {
entry:
; CHECK-LABEL: vla_and_realign
; CHECK: add     x29, sp, #16
; CHECK: sub     x9, sp, #64
; CHECK: and     sp, x9, #0xffffffffffffffe0
; CHECK: mov     x19, sp
; CHECK: orr     x1, xzr, #0xfffffffffffffffe
; CHECK: stur    x1, [x19]
; CHECK: .set .Lvla_and_realign$frame_escape_0, 32
; CHECK: stur    w0, [x29, #-4]
; CHECK: ldur    w8, [x29, #-4]
; CHECK: mov     x9, sp
; CHECK: str     x9, [x19, #24]
; CHECK: str     x8, [x19, #16]
; CHECK: ldr     w0, [x19, #32]
; CHECK: bl      foo

  %n.addr = alloca i32, align 4
  %o = alloca %struct.S, align 32
  %saved_stack = alloca i8*, align 8
  %__vla_expr0 = alloca i64, align 8
  call void (...) @llvm.localescape(%struct.S* %o)
  store i32 %n, i32* %n.addr, align 4
  %0 = load i32, i32* %n.addr, align 4
  %1 = zext i32 %0 to i64
  %2 = call i8* @llvm.stacksave()
  store i8* %2, i8** %saved_stack, align 8
  %vla = alloca i32, i64 %1, align 4
  store i64 %1, i64* %__vla_expr0, align 8
  %x = getelementptr inbounds %struct.S, %struct.S* %o, i32 0, i32 0
  %3 = load i32, i32* %x, align 32
  invoke void @foo(i32 %3) #5
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  %4 = call i8* @llvm.localaddress()
  call void @fin_vla_and_realign(i8 0, i8* %4)
  %5 = load i8*, i8** %saved_stack, align 8
  call void @llvm.stackrestore(i8* %5)
  ret void

ehcleanup:                                        ; preds = %entry
  %6 = cleanuppad within none []
  %7 = call i8* @llvm.localaddress()
  call void @fin_vla_and_realign(i8 1, i8* %7) [ "funclet"(token %6) ]
  cleanupret from %6 unwind to caller
}

define void @fin_vla_and_realign(i8 %abnormal_termination, i8* %frame_pointer) {
entry:
; CHECK-LABEL: fin_vla_and_realign
; CHECK: movz    x8, #:abs_g1_s:.Lvla_and_realign$frame_escape_0
; CHECK: movk    x8, #:abs_g0_nc:.Lvla_and_realign$frame_escape_0
; CHECK: strb    w0, [sp, #15]
; CHECK: ldr     w0, [x1, x8]
; CHECK: bl      foo

  %frame_pointer.addr = alloca i8*, align 8
  %abnormal_termination.addr = alloca i8, align 1
  %0 = call i8* @llvm.localrecover(i8* bitcast (void (i32)* @vla_and_realign to i8*), i8* %frame_pointer, i32 0)
  %o = bitcast i8* %0 to %struct.S*
  store i8* %frame_pointer, i8** %frame_pointer.addr, align 8
  store i8 %abnormal_termination, i8* %abnormal_termination.addr, align 1
  %x = getelementptr inbounds %struct.S, %struct.S* %o, i32 0, i32 0
  %1 = load i32, i32* %x, align 32
  call void @foo(i32 %1)
  ret void
}

declare void @foo(i32)
declare void @llvm.stackrestore(i8*)
declare i8* @llvm.stacksave()
declare i8* @llvm.localrecover(i8*, i8*, i32)
declare i8* @llvm.localaddress()
declare void @llvm.localescape(...)
declare i32 @__C_specific_handler(...)

attributes #0 = { noinline optnone }
