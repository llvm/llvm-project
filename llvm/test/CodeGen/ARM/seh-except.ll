; RUN: llc -mtriple=thumbv7-windows-msvc -o - %s | FileCheck %s

; struct S { int x; };
; void foo(int n);
; 
; void simple_except() {
;   struct S o;
;   __try {
;     foo(o.x);
;   } __except((foo(o.x), 1)) {
;     foo(o.x);
;   }
; }
; 
; void stack_realign() {
;   struct S __declspec(align(32)) o;
;   __try {
;     foo(o.x);
;   } __except((foo(o.x), 1)) {
;     foo(o.x);
;   }
; }
; 
; void vla_present(int n) {
;   int vla[n];
;   __try {
;     foo(n);
;   } __except((foo(n), 1)) {
;     foo(n);
;   }
; }
; 
; void vla_and_realign(int n) {
;   struct S __declspec(align(32)) o;
;   int vla[n];
;   __try {
;     foo(o.x);
;   } __except((foo(o.x), 1)) {
;     foo(o.x);
;   }
; }
; 
; void stack_realign_filter() {
;   struct S o;
;   __try {
;     foo(o.x);
;   } __except(([] [[msvc::forceinline]] () {
;                 struct S __declspec(align(32)) p;
;                 foo(p.x);
;               }(), foo(o.x), 1)) {
;     foo(o.x);
;   }
; }

%struct.S = type { i32 }

; Function Attrs: nounwind
define dso_local arm_aapcs_vfpcc void @simple_except() #0 personality ptr @__C_specific_handler {
; CHECK-LABEL: simple_except:
; CHECK: .seh_proc simple_except
; CHECK: .seh_handler __C_specific_handler, %unwind, %except
; CHECK: push {r6, lr}
; CHECK: .seh_save_regs {r6, lr}
; CHECK: sub sp, #8
; CHECK: .seh_stackalloc 8
; CHECK: .seh_endprologue
; CHECK: mov r6, sp
; CHECK: $Msimple_except$frame_escape_0 = 4
; CHECK: ldr r0, [r6, #4]
; CHECK: bl foo
entry:
  %o = alloca %struct.S, align 4
  %__exception_code = alloca i32, align 4
  call void (...) @llvm.localescape(ptr %o)
  call void @llvm.lifetime.start.p0(ptr %o) #6
  %x = getelementptr inbounds nuw %struct.S, ptr %o, i32 0, i32 0
  %0 = load i32, ptr %x, align 4
  invoke arm_aapcs_vfpcc void @foo(i32 noundef %0) #7
          to label %invoke.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %1 = catchswitch within none [label %__except.ret] unwind label %ehcleanup

__except.ret:                                     ; preds = %catch.dispatch
  %2 = catchpad within %1 [ptr @"?filt$0@0@simple_except@@"]
  catchret from %2 to label %__except

__except:                                         ; preds = %__except.ret
; CHECK: ldr r0, [r6, #4]
; CHECK: bl foo
  %3 = call i32 @llvm.eh.exceptioncode(token %2)
  store i32 %3, ptr %__exception_code, align 4
  %x1 = getelementptr inbounds nuw %struct.S, ptr %o, i32 0, i32 0
  %4 = load i32, ptr %x1, align 4
  call arm_aapcs_vfpcc void @foo(i32 noundef %4)
  br label %__try.cont

__try.cont:                                       ; preds = %__except, %invoke.cont
  call void @llvm.lifetime.end.p0(ptr %o) #6
  ret void

invoke.cont:                                      ; preds = %entry
  br label %__try.cont

ehcleanup:                                        ; preds = %catch.dispatch
  %5 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(ptr %o) #6
  cleanupret from %5 unwind to caller
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none))

; Function Attrs: nounwind
define internal arm_aapcs_vfpcc i32 @"?filt$0@0@simple_except@@"(ptr noundef %exception_pointers, ptr noundef %frame_pointer) #0 {
; CHECK-LABEL: "?filt$0@0@simple_except@@":
; CHECK: push.w {r11, lr}
; CHECK: sub sp, #16
; CHECK: movw r[[OFFSET:[0-9]+]], :lower16:{{.*}}frame_escape_0
; CHECK-NEXT: movt r[[OFFSET]], :upper16:{{.*}}frame_escape_0
; CHECK: ldr r0, [r6, r[[OFFSET]]]
; CHECK: bl foo
entry:
  %frame_pointer.addr = alloca ptr, align 4
  %exception_pointers.addr = alloca ptr, align 4
  %0 = call ptr @llvm.eh.recoverfp(ptr @simple_except, ptr %frame_pointer)
  %o = call ptr @llvm.localrecover(ptr @simple_except, ptr %0, i32 0)
  %__exception_code = alloca i32, align 4
  store ptr %frame_pointer, ptr %frame_pointer.addr, align 4
  store ptr %exception_pointers, ptr %exception_pointers.addr, align 4
  %1 = getelementptr inbounds nuw { ptr, ptr }, ptr %exception_pointers, i32 0, i32 0
  %2 = load ptr, ptr %1, align 4
  %3 = load i32, ptr %2, align 4
  store i32 %3, ptr %__exception_code, align 4
  %x = getelementptr inbounds nuw %struct.S, ptr %o, i32 0, i32 0
  %4 = load i32, ptr %x, align 4
  call arm_aapcs_vfpcc void @foo(i32 noundef %4)
  ret i32 1
}

declare ptr @llvm.eh.recoverfp(ptr, ptr)
declare ptr @llvm.localrecover(ptr, ptr, i32 immarg)
declare dso_local arm_aapcs_vfpcc void @foo(i32 noundef)

declare dso_local arm_aapcs_vfpcc i32 @__C_specific_handler(...)

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.exceptioncode(token)

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none))

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.localescape(...)

; Function Attrs: nounwind
define dso_local arm_aapcs_vfpcc void @stack_realign() #0 personality ptr @__C_specific_handler {
; CHECK-LABEL: stack_realign:
; CHECK: push {r4, r6}
; CHECK: push.w {r11, lr}
; CHECK: mov r11, sp
; CHECK: sub sp, #48
; CHECK: bfc {{r[0-9]+}}, #0, #5
; CHECK: mov r6, sp
; CHECK: $Mstack_realign$frame_escape_0 = 32
; CHECK: ldr r0, [sp, #32]
; CHECK: bl foo
entry:
  %o = alloca %struct.S, align 32
  %__exception_code = alloca i32, align 4
  call void (...) @llvm.localescape(ptr %o)
  call void @llvm.lifetime.start.p0(ptr %o) #6
  %x = getelementptr inbounds nuw %struct.S, ptr %o, i32 0, i32 0
  %0 = load i32, ptr %x, align 32
  invoke arm_aapcs_vfpcc void @foo(i32 noundef %0) #7
          to label %invoke.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %1 = catchswitch within none [label %__except.ret] unwind label %ehcleanup

__except.ret:                                     ; preds = %catch.dispatch
  %2 = catchpad within %1 [ptr @"?filt$0@0@stack_realign@@"]
  catchret from %2 to label %__except

__except:                                         ; preds = %__except.ret
; CHECK: ldr r0, [sp, #32]
; CHECK: bl foo
  %3 = call i32 @llvm.eh.exceptioncode(token %2)
  store i32 %3, ptr %__exception_code, align 4
  %x1 = getelementptr inbounds nuw %struct.S, ptr %o, i32 0, i32 0
  %4 = load i32, ptr %x1, align 32
  call arm_aapcs_vfpcc void @foo(i32 noundef %4)
  br label %__try.cont

__try.cont:                                       ; preds = %__except, %invoke.cont
  call void @llvm.lifetime.end.p0(ptr %o) #6
  ret void

invoke.cont:                                      ; preds = %entry
  br label %__try.cont

ehcleanup:                                        ; preds = %catch.dispatch
  %5 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(ptr %o) #6
  cleanupret from %5 unwind to caller
}

; Function Attrs: nounwind
define internal arm_aapcs_vfpcc i32 @"?filt$0@0@stack_realign@@"(ptr noundef %exception_pointers, ptr noundef %frame_pointer) #0 {
; CHECK-LABEL: "?filt$0@0@stack_realign@@":
; CHECK: movw r[[OFFSET:[0-9]+]], :lower16:{{.*}}frame_escape_0
; CHECK-NEXT: movt r[[OFFSET]], :upper16:{{.*}}frame_escape_0
; CHECK: ldr r0, [r6, r[[OFFSET]]]
; CHECK: bl foo
entry:
  %frame_pointer.addr = alloca ptr, align 4
  %exception_pointers.addr = alloca ptr, align 4
  %0 = call ptr @llvm.eh.recoverfp(ptr @stack_realign, ptr %frame_pointer)
  %o = call ptr @llvm.localrecover(ptr @stack_realign, ptr %0, i32 0)
  %__exception_code = alloca i32, align 4
  store ptr %frame_pointer, ptr %frame_pointer.addr, align 4
  store ptr %exception_pointers, ptr %exception_pointers.addr, align 4
  %1 = getelementptr inbounds nuw { ptr, ptr }, ptr %exception_pointers, i32 0, i32 0
  %2 = load ptr, ptr %1, align 4
  %3 = load i32, ptr %2, align 4
  store i32 %3, ptr %__exception_code, align 4
  %x = getelementptr inbounds nuw %struct.S, ptr %o, i32 0, i32 0
  %4 = load i32, ptr %x, align 32
  call arm_aapcs_vfpcc void @foo(i32 noundef %4)
  ret i32 1
}

; Function Attrs: nounwind
define dso_local arm_aapcs_vfpcc void @vla_present(i32 noundef %n) #0 personality ptr @__C_specific_handler {
; CHECK-LABEL: vla_present:
; CHECK: push {r6, lr}
; CHECK: sub sp, #16
; CHECK: mov r6, sp
; CHECK: $Mvla_present$frame_escape_0 = 12
; CHECK: bl foo
entry:
  %n.addr = alloca i32, align 4
  %saved_stack = alloca ptr, align 4
  %__vla_expr0 = alloca i32, align 4
  %__exception_code = alloca i32, align 4
  call void (...) @llvm.localescape(ptr %n.addr)
  store i32 %n, ptr %n.addr, align 4
  %0 = load i32, ptr %n.addr, align 4
  %1 = call ptr @llvm.stacksave.p0()
  store ptr %1, ptr %saved_stack, align 4
  %vla = alloca i32, i32 %0, align 4
  store i32 %0, ptr %__vla_expr0, align 4
  %2 = load i32, ptr %n.addr, align 4
  invoke arm_aapcs_vfpcc void @foo(i32 noundef %2) #7
          to label %invoke.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %3 = catchswitch within none [label %__except.ret] unwind to caller

__except.ret:                                     ; preds = %catch.dispatch
  %4 = catchpad within %3 [ptr @"?filt$0@0@vla_present@@"]
  catchret from %4 to label %__except

__except:                                         ; preds = %__except.ret
; CHECK: ldr r0, [r6, #12]
; CHECK: bl foo
  %5 = call i32 @llvm.eh.exceptioncode(token %4)
  store i32 %5, ptr %__exception_code, align 4
  %6 = load i32, ptr %n.addr, align 4
  call arm_aapcs_vfpcc void @foo(i32 noundef %6)
  br label %__try.cont

__try.cont:                                       ; preds = %__except, %invoke.cont
  %7 = load ptr, ptr %saved_stack, align 4
  call void @llvm.stackrestore.p0(ptr %7)
  ret void

invoke.cont:                                      ; preds = %entry
  br label %__try.cont
}

declare ptr @llvm.stacksave.p0() #5

; Function Attrs: nounwind
define internal arm_aapcs_vfpcc i32 @"?filt$0@0@vla_present@@"(ptr noundef %exception_pointers, ptr noundef %frame_pointer) #0 {
; CHECK-LABEL: "?filt$0@0@vla_present@@":
; CHECK: movw r[[OFFSET:[0-9]+]], :lower16:{{.*}}frame_escape_0
; CHECK-NEXT: movt r[[OFFSET]], :upper16:{{.*}}frame_escape_0
; CHECK: ldr r0, [r6, r[[OFFSET]]]
; CHECK: bl foo
entry:
  %frame_pointer.addr = alloca ptr, align 4
  %exception_pointers.addr = alloca ptr, align 4
  %0 = call ptr @llvm.eh.recoverfp(ptr @vla_present, ptr %frame_pointer)
  %n.addr = call ptr @llvm.localrecover(ptr @vla_present, ptr %0, i32 0)
  %__exception_code = alloca i32, align 4
  store ptr %frame_pointer, ptr %frame_pointer.addr, align 4
  store ptr %exception_pointers, ptr %exception_pointers.addr, align 4
  %1 = getelementptr inbounds nuw { ptr, ptr }, ptr %exception_pointers, i32 0, i32 0
  %2 = load ptr, ptr %1, align 4
  %3 = load i32, ptr %2, align 4
  store i32 %3, ptr %__exception_code, align 4
  %4 = load i32, ptr %n.addr, align 4
  call arm_aapcs_vfpcc void @foo(i32 noundef %4)
  ret i32 1
}

declare void @llvm.stackrestore.p0(ptr) #5

; Function Attrs: nounwind
define dso_local arm_aapcs_vfpcc void @vla_and_realign(i32 noundef %n) #0 personality ptr @__C_specific_handler {
; CHECK-LABEL: vla_and_realign:
; CHECK: push {r4, r6}
; CHECK: push.w {r11, lr}
; CHECK: mov r11, sp
; CHECK: sub sp, #48
; CHECK: bfc {{r[0-9]+}}, #0, #5
; CHECK: mov r6, sp
; CHECK: $Mvla_and_realign$frame_escape_0 = 32
; CHECK: ldr r0, [sp, #32]
; CHECK: bl foo
entry:
  %n.addr = alloca i32, align 4
  %o = alloca %struct.S, align 32
  %saved_stack = alloca ptr, align 4
  %__vla_expr0 = alloca i32, align 4
  %__exception_code = alloca i32, align 4
  call void (...) @llvm.localescape(ptr %o)
  store i32 %n, ptr %n.addr, align 4
  call void @llvm.lifetime.start.p0(ptr %o) #6
  %0 = load i32, ptr %n.addr, align 4
  %1 = call ptr @llvm.stacksave.p0()
  store ptr %1, ptr %saved_stack, align 4
  %vla = alloca i32, i32 %0, align 4
  store i32 %0, ptr %__vla_expr0, align 4
  %x = getelementptr inbounds nuw %struct.S, ptr %o, i32 0, i32 0
  %2 = load i32, ptr %x, align 32
  invoke arm_aapcs_vfpcc void @foo(i32 noundef %2) #7
          to label %invoke.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %3 = catchswitch within none [label %__except.ret] unwind label %ehcleanup

__except.ret:                                     ; preds = %catch.dispatch
  %4 = catchpad within %3 [ptr @"?filt$0@0@vla_and_realign@@"]
  catchret from %4 to label %__except

__except:                                         ; preds = %__except.ret
; CHECK: ldr r0, [sp, #32]
; CHECK: bl foo
  %5 = call i32 @llvm.eh.exceptioncode(token %4)
  store i32 %5, ptr %__exception_code, align 4
  %x1 = getelementptr inbounds nuw %struct.S, ptr %o, i32 0, i32 0
  %6 = load i32, ptr %x1, align 32
  call arm_aapcs_vfpcc void @foo(i32 noundef %6)
  br label %__try.cont

__try.cont:                                       ; preds = %__except, %invoke.cont
  %7 = load ptr, ptr %saved_stack, align 4
  call void @llvm.stackrestore.p0(ptr %7)
  call void @llvm.lifetime.end.p0(ptr %o) #6
  ret void

invoke.cont:                                      ; preds = %entry
  br label %__try.cont

ehcleanup:                                        ; preds = %catch.dispatch
  %8 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(ptr %o) #6
  cleanupret from %8 unwind to caller
}

; Function Attrs: nounwind
define internal arm_aapcs_vfpcc i32 @"?filt$0@0@vla_and_realign@@"(ptr noundef %exception_pointers, ptr noundef %frame_pointer) #0 {
; CHECK-LABEL: "?filt$0@0@vla_and_realign@@":
; CHECK: movw r[[OFFSET:[0-9]+]], :lower16:{{.*}}frame_escape_0
; CHECK-NEXT: movt r[[OFFSET]], :upper16:{{.*}}frame_escape_0
; CHECK: ldr r0, [r6, r[[OFFSET]]]
; CHECK: bl foo
entry:
  %frame_pointer.addr = alloca ptr, align 4
  %exception_pointers.addr = alloca ptr, align 4
  %0 = call ptr @llvm.eh.recoverfp(ptr @vla_and_realign, ptr %frame_pointer)
  %o = call ptr @llvm.localrecover(ptr @vla_and_realign, ptr %0, i32 0)
  %__exception_code = alloca i32, align 4
  store ptr %frame_pointer, ptr %frame_pointer.addr, align 4
  store ptr %exception_pointers, ptr %exception_pointers.addr, align 4
  %1 = getelementptr inbounds nuw { ptr, ptr }, ptr %exception_pointers, i32 0, i32 0
  %2 = load ptr, ptr %1, align 4
  %3 = load i32, ptr %2, align 4
  store i32 %3, ptr %__exception_code, align 4
  %x = getelementptr inbounds nuw %struct.S, ptr %o, i32 0, i32 0
  %4 = load i32, ptr %x, align 32
  call arm_aapcs_vfpcc void @foo(i32 noundef %4)
  ret i32 1
}

%class.anon = type { i8 }

define void @stack_realign_filter() #0 personality ptr @__C_specific_handler {
; CHECK-LABEL: stack_realign_filter:
; CHECK: push {r6, lr}
; CHECK: mov r6, sp
; CHECK: $Mstack_realign_filter$frame_escape_0 = 4
; CHECK: bl foo
entry:
  %o = alloca %struct.S, align 4
  %__exception_code = alloca i32, align 4
  call void (...) @llvm.localescape(ptr %o)
  %x = getelementptr inbounds nuw %struct.S, ptr %o, i32 0, i32 0
  %0 = load i32, ptr %x, align 4
  invoke void @foo(i32 noundef %0)
          to label %invoke.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %1 = catchswitch within none [label %__except.ret] unwind to caller

__except.ret:                                     ; preds = %catch.dispatch
  %2 = catchpad within %1 [ptr @"?filt$0@0@stack_realign_filter@@"]
  catchret from %2 to label %__except

__except:                                         ; preds = %__except.ret
; CHECK: ldr r0, [r6, #4]
; CHECK: bl foo
  %3 = call i32 @llvm.eh.exceptioncode(token %2)
  store i32 %3, ptr %__exception_code, align 4
  %x1 = getelementptr inbounds nuw %struct.S, ptr %o, i32 0, i32 0
  %4 = load i32, ptr %x1, align 4
  call void @foo(i32 noundef %4)
  br label %__try.cont

__try.cont:                                       ; preds = %__except, %invoke.cont
  ret void

invoke.cont:                                      ; preds = %entry
  br label %__try.cont
}

define internal arm_aapcs_vfpcc i32 @"?filt$0@0@stack_realign_filter@@"(ptr noundef %exception_pointers, ptr noundef %frame_pointer) #0 {
; CHECK-LABEL: "?filt$0@0@stack_realign_filter@@":
; CHECK: push.w {r11, lr}
; CHECK: mov r11, sp
; CHECK: bfc r4, #0, #5
; CHECK-NOT: mov r6, sp
; CHECK: ldr r0, [sp, #{{[0-9]+}}]
; CHECK-NEXT: bl foo
; CHECK: ldr r0, [r6, r{{.*}}]
; CHECK-NEXT: bl foo
entry:
  %this.addr.i = alloca ptr, align 4
  %p.i = alloca %struct.S, align 32
  %frame_pointer.addr = alloca ptr, align 4
  %exception_pointers.addr = alloca ptr, align 4
  %0 = call ptr @llvm.eh.recoverfp(ptr @stack_realign_filter, ptr %frame_pointer)
  %o = call ptr @llvm.localrecover(ptr @stack_realign_filter, ptr %0, i32 0)
  %__exception_code = alloca i32, align 4
  %ref.tmp = alloca %class.anon, align 1
  store ptr %frame_pointer, ptr %frame_pointer.addr, align 4
  store ptr %exception_pointers, ptr %exception_pointers.addr, align 4
  %1 = getelementptr inbounds nuw { ptr, ptr }, ptr %exception_pointers, i32 0, i32 0
  %2 = load ptr, ptr %1, align 4
  %3 = load i32, ptr %2, align 4
  store i32 %3, ptr %__exception_code, align 4
  store ptr %ref.tmp, ptr %this.addr.i, align 4
  %this1.i = load ptr, ptr %this.addr.i, align 4
  %4 = load i32, ptr %p.i, align 32
  call void @foo(i32 noundef %4)
  %x = getelementptr inbounds nuw %struct.S, ptr %o, i32 0, i32 0
  %5 = load i32, ptr %x, align 4
  call void @foo(i32 noundef %5)
  ret i32 1
}

attributes #0 = { noinline optnone }
