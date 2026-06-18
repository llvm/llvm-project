; RUN: llc -mtriple thumbv7-windows-msvc -o - %s | FileCheck %s

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
define arm_aapcs_vfpcc void @simple_seh() #0 personality ptr @__C_specific_handler {
entry:
; CHECK-LABEL: simple_seh:
; CHECK: mov r6, sp
; CHECK: $Msimple_seh$frame_escape_0 = 4
; CHECK: ldr r0, [r6, #4]
; CHECK: bl foo

  %o = alloca %struct.S, align 4
  call void (...) @llvm.localescape(ptr %o)
  %x = getelementptr inbounds nuw %struct.S, ptr %o, i32 0, i32 0
  %0 = load i32, ptr %x, align 4
  invoke arm_aapcs_vfpcc void @foo(i32 noundef %0) #5
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
; CHECK: movs r0, #0
; CHECK: mov r1, r6
; CHECK: bl "?fin$0@0@simple_seh@@"
  %1 = call ptr @llvm.localaddress()
  call arm_aapcs_vfpcc void @"?fin$0@0@simple_seh@@"(i8 noundef zeroext 0, ptr noundef %1)
  ret void

ehcleanup:                                        ; preds = %entry
; CHECK-LABEL: "?dtor$2@?0?simple_seh@4HA":
; CHECK: movs r0, #1
; CHECK: mov r1, r6
; CHECK: bl "?fin$0@0@simple_seh@@"
  %2 = cleanuppad within none []
  %3 = call ptr @llvm.localaddress()
  call arm_aapcs_vfpcc void @"?fin$0@0@simple_seh@@"(i8 noundef zeroext 1, ptr noundef %3) [ "funclet"(token %2) ]
  cleanupret from %2 unwind to caller
}

define arm_aapcs_vfpcc void @"?fin$0@0@simple_seh@@"(i8 noundef zeroext %abnormal_termination, ptr noundef %frame_pointer) #1 {
entry:
; CHECK-LABEL: "?fin$0@0@simple_seh@@":
; CHECK: movw r0, :lower16:$Msimple_seh$frame_escape_0
; CHECK: movt r0, :upper16:$Msimple_seh$frame_escape_0
; CHECK: ldr r0, [r1, r0]
; CHECK: bl foo

  %o = call ptr @llvm.localrecover(ptr @simple_seh, ptr %frame_pointer, i32 0)
  %x = getelementptr inbounds nuw %struct.S, ptr %o, i32 0, i32 0
  %0 = load i32, ptr %x, align 4
  call arm_aapcs_vfpcc void @foo(i32 noundef %0)
  ret void
}

; Test SEH when stack realignment is needed in case highly aligned stack objects are present.
define arm_aapcs_vfpcc void @stack_realign() #0 personality ptr @__C_specific_handler {
entry:
; CHECK-LABEL: stack_realign:
; CHECK: bfc r4, #0, #5
; CHECK: mov sp, r4
; CHECK: mov r6, sp
; CHECK: $Mstack_realign$frame_escape_0 = 0
; CHECK: ldr r0, [sp]
; CHECK: bl foo

  %o = alloca %struct.S, align 32
  call void (...) @llvm.localescape(ptr %o)
  %x = getelementptr inbounds nuw %struct.S, ptr %o, i32 0, i32 0
  %0 = load i32, ptr %x, align 32
  invoke arm_aapcs_vfpcc void @foo(i32 noundef %0) #5
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
; CHECK: movs r0, #0
; CHECK: mov r1, r6
; CHECK: bl "?fin$0@0@stack_realign@@"
  %1 = call ptr @llvm.localaddress()
  call arm_aapcs_vfpcc void @"?fin$0@0@stack_realign@@"(i8 noundef zeroext 0, ptr noundef %1)
  ret void

ehcleanup:                                        ; preds = %entry
; CHECK-LABEL: "?dtor$2@?0?stack_realign@4HA":
; CHECK: movs r0, #1
; CHECK: mov r1, r6
; CHECK: bl "?fin$0@0@stack_realign@@"
  %2 = cleanuppad within none []
  %3 = call ptr @llvm.localaddress()
  call arm_aapcs_vfpcc void @"?fin$0@0@stack_realign@@"(i8 noundef zeroext 1, ptr noundef %3) [ "funclet"(token %2) ]
  cleanupret from %2 unwind to caller
}

define arm_aapcs_vfpcc void @"?fin$0@0@stack_realign@@"(i8 noundef zeroext %abnormal_termination, ptr noundef %frame_pointer) #1 {
entry:
; CHECK-LABEL: "?fin$0@0@stack_realign@@":
; CHECK: movw r0, :lower16:$Mstack_realign$frame_escape_0
; CHECK: movt r0, :upper16:$Mstack_realign$frame_escape_0
; CHECK: ldr r0, [r1, r0]
; CHECK: bl foo

  %o = call ptr @llvm.localrecover(ptr @stack_realign, ptr %frame_pointer, i32 0)
  %x = getelementptr inbounds nuw %struct.S, ptr %o, i32 0, i32 0
  %0 = load i32, ptr %x, align 32
  call arm_aapcs_vfpcc void @foo(i32 noundef %0)
  ret void
}

; Test SEH when variable size objects are present on the stack. Note: Escaped vla's are current not supported by SEH.
define arm_aapcs_vfpcc void @vla_present(i32 noundef %n) #0 personality ptr @__C_specific_handler {
entry:
; CHECK-LABEL: vla_present:
; CHECK: mov r6, sp
; CHECK: $Mvla_present$frame_escape_0 = 12
; CHECK: bl foo

  %n.addr = alloca i32, align 4
  %saved_stack = alloca ptr, align 4
  %__vla_expr0 = alloca i32, align 4
  call void (...) @llvm.localescape(ptr %n.addr)
  store i32 %n, ptr %n.addr, align 4
  %0 = load i32, ptr %n.addr, align 4
  %1 = call ptr @llvm.stacksave.p0()
  store ptr %1, ptr %saved_stack, align 4
  %vla = alloca i32, i32 %0, align 4
  store i32 %0, ptr %__vla_expr0, align 4
  %2 = load i32, ptr %n.addr, align 4
  invoke arm_aapcs_vfpcc void @foo(i32 noundef %2) #5
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
; CHECK: movs r0, #0
; CHECK: mov r1, r6
; CHECK: bl "?fin$0@0@vla_present@@"
  %3 = call ptr @llvm.localaddress()
  call arm_aapcs_vfpcc void @"?fin$0@0@vla_present@@"(i8 noundef zeroext 0, ptr noundef %3)
  %4 = load ptr, ptr %saved_stack, align 4
  call void @llvm.stackrestore.p0(ptr %4)
  ret void

ehcleanup:                                        ; preds = %entry
; CHECK-LABEL: "?dtor$2@?0?vla_present@4HA":
; CHECK: movs r0, #1
; CHECK: mov r1, r6
; CHECK: bl "?fin$0@0@vla_present@@"
  %5 = cleanuppad within none []
  %6 = call ptr @llvm.localaddress()
  call arm_aapcs_vfpcc void @"?fin$0@0@vla_present@@"(i8 noundef zeroext 1, ptr noundef %6) [ "funclet"(token %5) ]
  cleanupret from %5 unwind to caller
}

define arm_aapcs_vfpcc void @"?fin$0@0@vla_present@@"(i8 noundef zeroext %abnormal_termination, ptr noundef %frame_pointer) #1 {
entry:
; CHECK-LABEL: "?fin$0@0@vla_present@@":
; CHECK: movw r0, :lower16:$Mvla_present$frame_escape_0
; CHECK: movt r0, :upper16:$Mvla_present$frame_escape_0
; CHECK: ldr r0, [r1, r0]
; CHECK: bl foo

  %n.addr = call ptr @llvm.localrecover(ptr @vla_present, ptr %frame_pointer, i32 0)
  %0 = load i32, ptr %n.addr, align 4
  call arm_aapcs_vfpcc void @foo(i32 noundef %0)
  ret void
}

; Test when both vla's and highly aligned objects are present on stack.
define arm_aapcs_vfpcc void @vla_and_realign(i32 noundef %n) #0 personality ptr @__C_specific_handler {
entry:
; CHECK-LABEL: vla_and_realign:
; CHECK: bfc r4, #0, #5
; CHECK: mov sp, r4
; CHECK: mov r6, sp
; CHECK: $Mvla_and_realign$frame_escape_0 = 0
; CHECK: bl foo

  %o = alloca %struct.S, align 32
  call void (...) @llvm.localescape(ptr %o)
  %0 = call ptr @llvm.stacksave.p0()
  %x = getelementptr inbounds nuw %struct.S, ptr %o, i32 0, i32 0
  %1 = load i32, ptr %x, align 32
  invoke arm_aapcs_vfpcc void @foo(i32 noundef %1) #5
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
; CHECK: movs r0, #0
; CHECK: mov r1, r6
; CHECK: bl "?fin$0@0@vla_and_realign@@"
  %2 = call ptr @llvm.localaddress()
  call arm_aapcs_vfpcc void @"?fin$0@0@vla_and_realign@@"(i8 noundef zeroext 0, ptr noundef %2)
  call void @llvm.stackrestore.p0(ptr %0)
  ret void

ehcleanup:                                        ; preds = %entry
; CHECK-LABEL: "?dtor$2@?0?vla_and_realign@4HA":
; CHECK: movs r0, #1
; CHECK: mov r1, r6
; CHECK: bl "?fin$0@0@vla_and_realign@@"
  %3 = cleanuppad within none []
  %4 = call ptr @llvm.localaddress()
  call arm_aapcs_vfpcc void @"?fin$0@0@vla_and_realign@@"(i8 noundef zeroext 1, ptr noundef %4) [ "funclet"(token %3) ]
  cleanupret from %3 unwind to caller
}

define arm_aapcs_vfpcc void @"?fin$0@0@vla_and_realign@@"(i8 noundef zeroext %abnormal_termination, ptr noundef %frame_pointer) #1 {
entry:
; CHECK-LABEL: "?fin$0@0@vla_and_realign@@":
; CHECK: movw r0, :lower16:$Mvla_and_realign$frame_escape_0
; CHECK: movt r0, :upper16:$Mvla_and_realign$frame_escape_0
; CHECK: ldr r0, [r1, r0]
; CHECK: bl foo

  %o = call ptr @llvm.localrecover(ptr @vla_and_realign, ptr %frame_pointer, i32 0)
  %x = getelementptr inbounds nuw %struct.S, ptr %o, i32 0, i32 0
  %0 = load i32, ptr %x, align 32
  call arm_aapcs_vfpcc void @foo(i32 noundef %0)
  ret void
}

declare arm_aapcs_vfpcc void @foo(i32 noundef)
declare void @llvm.stackrestore.p0(ptr)
declare ptr @llvm.stacksave.p0()
declare ptr @llvm.localrecover(ptr, ptr, i32 immarg)
declare ptr @llvm.localaddress()
declare void @llvm.localescape(...)
declare i32 @__C_specific_handler(...)

attributes #0 = { noinline optnone }
attributes #1 = { noinline }
