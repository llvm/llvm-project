; RUN: llc -mtriple=thumbv6m-eabi -frame-pointer=none %s -o - --verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,CHECK-NOFP,CHECK-ATPCS
; RUN: llc -mtriple=thumbv6m-eabi -frame-pointer=all %s -o - --verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,CHECK-FP-ATPCS,CHECK-ATPCS
; RUN: llc -mtriple=thumbv6m-eabi -frame-pointer=none -mattr=+aapcs-frame-chain-leaf %s -o - --verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,CHECK-NOFP,CHECK-AAPCS
; RUN: llc -mtriple=thumbv6m-eabi -frame-pointer=all -mattr=+aapcs-frame-chain-leaf %s -o - --verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,CHECK-FP-AAPCS,CHECK-AAPCS

; struct S { int x[128]; } s;
; int f(int *, int, int, int, struct S);
; int g(int *, int, int, int, int, int);
; int h(int *, int *, int *);
; int u(int *, int *, int *, struct S, struct S);

%struct.S = type { [128 x i32] }
%struct.__va_list = type { i8* }

@s = common dso_local global %struct.S zeroinitializer, align 4

declare void @llvm.va_start(i8*)
declare dso_local i32 @i(i32) local_unnamed_addr
declare dso_local i32 @g(i32*, i32, i32, i32, i32, i32) local_unnamed_addr
declare dso_local i32 @f(i32*, i32, i32, i32, %struct.S* byval(%struct.S) align 4) local_unnamed_addr
declare dso_local i32 @h(i32*, i32*, i32*) local_unnamed_addr
declare dso_local i32 @u(i32*, i32*, i32*, %struct.S* byval(%struct.S) align 4, %struct.S* byval(%struct.S) align 4) local_unnamed_addr

;
; Test access to arguments, passed on stack (including varargs)
;

; Usual case, access via SP if FP is not available
; int test_args_sp(int a, int b, int c, int d, int e) {
;   int v[4];
;   return g(v, a, b, c, d, e);
; }
define dso_local i32 @test_args_sp(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) local_unnamed_addr {
entry:
  %v = alloca [4 x i32], align 4
  %0 = bitcast [4 x i32]* %v to i8*
  %arraydecay = getelementptr inbounds [4 x i32], [4 x i32]* %v, i32 0, i32 0
  %call = call i32 @g(i32* nonnull %arraydecay, i32 %a, i32 %b, i32 %c, i32 %d, i32 %e)
  ret i32 %call
}
; CHECK-LABEL: test_args_sp
; Load `e`
; CHECK-NOFP: ldr    r0, [sp, #32]
; CHECK-FP-ATPCS: ldr  r0, [r7, #8]
; CHECK-FP-AAPCS: mov    r0, r11
; CHECK-FP-AAPCS: ldr    r0, [r0, #8]
; CHECK-NEXT:  str    r3, [sp]
; Pass `e` on stack
; CHECK-NEXT:  str    r0, [sp, #4]
; CHECK:       bl    g

; int test_varargs_sp(int a, ...) {
;   int v[4];
;   __builtin_va_list ap;
;   __builtin_va_start(ap, a);
;   return g(v, a, 0, 0, 0, 0);
; }
define dso_local i32 @test_varargs_sp(i32 %a, ...) local_unnamed_addr  {
entry:
  %v = alloca [4 x i32], align 4
  %ap = alloca %struct.__va_list, align 4
  %0 = bitcast [4 x i32]* %v to i8*
  %1 = bitcast %struct.__va_list* %ap to i8*
  call void @llvm.va_start(i8* nonnull %1)
  %arraydecay = getelementptr inbounds [4 x i32], [4 x i32]* %v, i32 0, i32 0
  %call = call i32 @g(i32* nonnull %arraydecay, i32 %a, i32 0, i32 0, i32 0, i32 0)
  ret i32 %call
}
; CHECK-LABEL: test_varargs_sp
; Three incoming varargs in registers
; CHECK:       sub sp, #12
; CHECK:       sub sp, #28
; Incoming arguments area is accessed via SP if FP is not available
; CHECK-NOFP:  add r0, sp, #36
; CHECK-NOFP:  stm r0!, {r1, r2, r3}
; CHECK-FP-ATPCS: mov r0, r7
; CHECK-FP-ATPCS: adds r0, #8
; CHECK-FP-ATPCS: stm r0!, {r1, r2, r3}
; CHECK-FP-AAPCS: mov r0, r11
; CHECK-FP-AAPCS: str r1, [r0, #8]
; CHECK-FP-AAPCS: mov r0, r11
; CHECK-FP-AAPCS: str r2, [r0, #12]
; CHECK-FP-AAPCS: mov r0, r11
; CHECK-FP-AAPCS: str r3, [r0, #16]

; Re-aligned stack, access via FP
; int test_args_realign(int a, int b, int c, int d, int e) {
;   __attribute__((aligned(16))) int v[4];
;   return g(v, a, b, c, d, e);
; }
; Function Attrs: nounwind
define dso_local i32 @test_args_realign(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) local_unnamed_addr  {
entry:
  %v = alloca [4 x i32], align 16
  %0 = bitcast [4 x i32]* %v to i8*
  %arraydecay = getelementptr inbounds [4 x i32], [4 x i32]* %v, i32 0, i32 0
  %call = call i32 @g(i32* nonnull %arraydecay, i32 %a, i32 %b, i32 %c, i32 %d, i32 %e)
  ret i32 %call
}
; CHECK-LABEL: test_args_realign
; Setup frame pointer
; CHECK-ATPCS: add r7, sp, #8
; CHECK-AAPCS: mov r11, sp
; Align stack
; CHECK:       mov  r4, sp
; CHECK-NEXT:  lsrs r4, r4, #4
; CHECK-NEXT:  lsls r4, r4, #4
; CHECK-NEXT:  mov  sp, r4
; Load `e` via FP
; CHECK-ATPCS: ldr r0, [r7, #8]
; CHECK-AAPCS: mov r0, r11
; CHECK-AAPCS: ldr r0, [r0, #8]
; CHECK-NEXT:  str r3, [sp]
; Pass `e` as argument
; CHECK-NEXT:  str r0, [sp, #4]
; CHECK:       bl    g

; int test_varargs_realign(int a, ...) {
;   __attribute__((aligned(16))) int v[4];
;   __builtin_va_list ap;
;   __builtin_va_start(ap, a);
;   return g(v, a, 0, 0, 0, 0);
; }
define dso_local i32 @test_varargs_realign(i32 %a, ...) local_unnamed_addr  {
entry:
  %v = alloca [4 x i32], align 16
  %ap = alloca %struct.__va_list, align 4
  %0 = bitcast [4 x i32]* %v to i8*
  %1 = bitcast %struct.__va_list* %ap to i8*
  call void @llvm.va_start(i8* nonnull %1)
  %arraydecay = getelementptr inbounds [4 x i32], [4 x i32]* %v, i32 0, i32 0
  %call = call i32 @g(i32* nonnull %arraydecay, i32 %a, i32 0, i32 0, i32 0, i32 0)
  ret i32 %call
}
; CHECK-LABEL: test_varargs_realign
; Three incoming register varargs
; CHECK:       sub sp, #12
; Setup frame pointer
; CHECK-ATPCS: add r7, sp, #8
; CHECK-AAPCS: mov r11, sp
; Align stack
; CHECK:       mov  r4, sp
; CHECK-NEXT:  lsrs r4, r4, #4
; CHECK-NEXT:  lsls r4, r4, #4
; CHECK-NEXT:  mov  sp, r4
; Incoming register varargs stored via FP
; CHECK-ATPCS: mov r0, r7
; CHECK-ATPCS-NEXT: adds r0, #8
; CHECK-ATPCS-NEXT: stm r0!, {r1, r2, r3}
; CHECK-AAPCS: mov r0, r11
; CHECK-AAPCS: str r1, [r0, #8]
; CHECK-AAPCS: mov r0, r11
; CHECK-AAPCS: str r2, [r0, #12]
; CHECK-AAPCS: mov r0, r11
; CHECK-AAPCS: str r3, [r0, #16]
; VLAs present, access via FP
; int test_args_vla(int a, int b, int c, int d, int e) {
;   int v[a];
;   return g(v, a, b, c, d, e);
; }
define dso_local i32 @test_args_vla(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) local_unnamed_addr  {
entry:
  %vla = alloca i32, i32 %a, align 4
  %call = call i32 @g(i32* nonnull %vla, i32 %a, i32 %b, i32 %c, i32 %d, i32 %e)
  ret i32 %call
}
; CHECK-LABEL: test_args_vla
; Setup frame pointer
; CHECK-ATPCS: add r7, sp, #12
; CHECK-AAPCS: mov r11, sp
; Allocate outgoing stack arguments space
; CHECK:       sub sp, #8
; Load `e` via FP
; CHECK-ATPCS: ldr r5, [r7, #8]
; CHECK-AAPCS: mov r5, r11
; CHECK-AAPCS: ldr r5, [r5, #8]
; Pass `d` and `e` as arguments
; CHECK-NEXT:  str r3, [sp]
; CHECK-NEXT:  str r5, [sp, #4]
; CHECK:       bl  g

; int test_varargs_vla(int a, ...) {
;   int v[a];
;   __builtin_va_list ap;
;   __builtin_va_start(ap, a);
;   return g(v, a, 0, 0, 0, 0);
; }
define dso_local i32 @test_varargs_vla(i32 %a, ...) local_unnamed_addr  {
entry:
  %ap = alloca %struct.__va_list, align 4
  %vla = alloca i32, i32 %a, align 4
  %0 = bitcast %struct.__va_list* %ap to i8*
  call void @llvm.va_start(i8* nonnull %0)
  %call = call i32 @g(i32* nonnull %vla, i32 %a, i32 0, i32 0, i32 0, i32 0)
  ret i32 %call
}
; CHECK-LABEL: test_varargs_vla
; Three incoming register varargs
; CHECK:       sub sp, #12
; Setup frame pointer
; CHECK-ATPCS: add r7, sp, #8
; CHECK-AAPCS: mov r11, sp
; Register varargs stored via FP
; CHECK-ATPCS-DAG:  str r3, [r7, #16]
; CHECK-ATPCS-DAG:  str r2, [r7, #12]
; CHECK-ATPCS-DAG:  str r1, [r7, #8]
; CHECK-AAPCS-DAG:  mov r5, r11
; CHECK-AAPCS-DAG:  str r1, [r5, #8]
; CHECK-AAPCS-DAG:  mov r1, r11
; CHECK-AAPCS-DAG:  str r3, [r1, #16]
; CHECK-AAPCS-DAG:  mov r1, r11
; CHECK-AAPCS-DAG:  str r2, [r1, #12]

; Moving SP, access via SP
; int test_args_moving_sp(int a, int b, int c, int d, int e) {
;   int v[4];
;   return f(v, a, b + c + d, e, s) + h(v, v+1, v+2);
; }
define dso_local i32 @test_args_moving_sp(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) local_unnamed_addr  {
entry:
  %v = alloca [4 x i32], align 4
  %0 = bitcast [4 x i32]* %v to i8*
  %arraydecay = getelementptr inbounds [4 x i32], [4 x i32]* %v, i32 0, i32 0
  %add = add nsw i32 %c, %b
  %add1 = add nsw i32 %add, %d
  %call = call i32 @f(i32* nonnull %arraydecay, i32 %a, i32 %add1, i32 %e, %struct.S* byval(%struct.S) nonnull align 4 @s)
  %add.ptr = getelementptr inbounds [4 x i32], [4 x i32]* %v, i32 0, i32 1
  %add.ptr5 = getelementptr inbounds [4 x i32], [4 x i32]* %v, i32 0, i32 2
  %call6 = call i32 @h(i32* nonnull %arraydecay, i32* nonnull %add.ptr, i32* nonnull %add.ptr5)
  %add7 = add nsw i32 %call6, %call
  ret i32 %add7
}
; CHECK-LABEL: test_args_moving_sp
; 20 bytes callee-saved area without FP
; CHECK-NOFP: push {r4, r5, r6, r7, lr}
; 20 bytes callee-saved area for ATPCS
; CHECK-FP-ATPCS: push {r4, r5, r6, r7, lr}
; 24 bytes callee-saved area for AAPCS as codegen prefers an even number of GPRs spilled
; CHECK-FP-AAPCS: push {lr}
; CHECK-FP-AAPCS: mov lr, r11
; CHECK-FP-AAPCS: push {lr}
; CHECK-FP-AAPCS: push {r4, r5, r6, r7}
; 20 bytes locals without FP
; CHECK-NOFP:       sub sp, #20
; 28 bytes locals with FP for ATPCS
; CHECK-FP-ATPCS:       sub sp, #28
; 24 bytes locals with FP for AAPCS
; CHECK-FP-AAPCS:       sub sp, #24
; Setup base pointer
; CHECK:       mov r6, sp
; Allocate outgoing arguments space
; CHECK:       sub sp, #508
; CHECK:       sub sp, #4
; Load `e` via BP if FP is not present (40 = 20 + 20)
; CHECK-NOFP:  ldr r3, [r6, #40]
; Load `e` via FP otherwise
; CHECK-FP-ATPCS: ldr r3, [r7, #8]
; CHECK-FP-AAPCS: mov r0, r11
; CHECK-FP-AAPCS: ldr r3, [r0, #8]
; CHECK:       bl  f
; Stack restored before next call
; CHECK-NEXT:  add sp, #508
; CHECK-NEXT:  add sp, #4
; CHECK:       bl  h

; int test_varargs_moving_sp(int a, ...) {
;   int v[4];
;   __builtin_va_list ap;
;   __builtin_va_start(ap, a);
;   return f(v, a, 0, 0, s) + h(v, v+1, v+2);
; }
define dso_local i32 @test_varargs_moving_sp(i32 %a, ...) local_unnamed_addr  {
entry:
  %v = alloca [4 x i32], align 4
  %ap = alloca %struct.__va_list, align 4
  %0 = bitcast [4 x i32]* %v to i8*
  %1 = bitcast %struct.__va_list* %ap to i8*
  call void @llvm.va_start(i8* nonnull %1)
  %arraydecay = getelementptr inbounds [4 x i32], [4 x i32]* %v, i32 0, i32 0
  %call = call i32 @f(i32* nonnull %arraydecay, i32 %a, i32 0, i32 0, %struct.S* byval(%struct.S) nonnull align 4 @s)
  %add.ptr = getelementptr inbounds [4 x i32], [4 x i32]* %v, i32 0, i32 1
  %add.ptr5 = getelementptr inbounds [4 x i32], [4 x i32]* %v, i32 0, i32 2
  %call6 = call i32 @h(i32* nonnull %arraydecay, i32* nonnull %add.ptr, i32* nonnull %add.ptr5)
  %add = add nsw i32 %call6, %call
  ret i32 %add
}
; CHECK-LABEL: test_varargs_moving_sp
; Three incoming register varargs
; CHECK:       sub sp, #12
; 16 bytes callee-saves without FP
; CHECK-NOFP: push {r4, r5, r6, lr}
; 24 bytes callee-saves with FP
; CHECK-FP-ATPCS: push {r4, r5, r6, r7, lr}
; CHECK-FP-AAPCS: push {lr}
; CHECK-FP-AAPCS: mov lr, r11
; CHECK-FP-AAPCS: push {lr}
; CHECK-FP-AAPCS: push {r4, r5, r6, r7}
; Locals area
; CHECK-NOFP:       sub sp, #20
; CHECK-FP-ATPCS:   sub sp, #24
; CHECK-FP-AAPCS:   sub sp, #20
; Incoming varargs stored via BP if FP is not present (36 = 20 + 16)
; CHECK-NOFP:      mov r0, r6
; CHECK-NOFP-NEXT: adds r0, #36
; CHECK-NOFP-NEXT: stm r0!, {r1, r2, r3}
; Incoming varargs stored via FP otherwise
; CHECK-FP-ATPCS:      mov r0, r7
; CHECK-FP-ATPCS-NEXT: adds r0, #8
; CHECK-FP-ATPCS-NEXT: stm r0!, {r1, r2, r3}
; CHECK-FP-AAPCS:      mov r0, r11
; CHECK-FP-AAPCS-NEXT: str r1, [r0, #8]
; CHECK-FP-AAPCS-NEXT: mov r0, r11
; CHECK-FP-AAPCS-NEXT: str r2, [r0, #12]
; CHECK-FP-AAPCS-NEXT: mov r0, r11
; CHECK-FP-AAPCS-NEXT: str r3, [r0, #16]

; struct S { int x[128]; } s;
; int test(S a, int b) {
;   return i(b);
; }
define dso_local i32 @test_args_large_offset(%struct.S* byval(%struct.S) align 4 %0, i32 %1) local_unnamed_addr {
  %3 = alloca i32, align 4
  store i32 %1, i32* %3, align 4
  %4 = load i32, i32* %3, align 4
  %5 = call i32 @i(i32 %4)
  ret i32 %5
}
; CHECK-LABEL: test_args_large_offset
; Without FP: Access to large offset is made using SP
; CHECK-NOFP:     ldr r0, [sp, #520]
; With FP: Access to large offset is made through a const pool using FP
; CHECK-FP:       ldr r0, .LCPI0_0
; CHECK-FP-ATPCS: ldr r0, [r0, r7]
; CHECK-FP-AAPCS: add r0, r11
; CHECK-FP-AAPCS: ldr r0, [r0]
; CHECK: bl i

;
; Access to locals
;

; Usual case, access via SP.
; int test_local(int n) {
;   int v[4];
;   int x, y, z;
;   h(&x, &y, &z);
;   return g(v, x, y, z, 0, 0);
; }
define dso_local i32 @test_local(i32 %n) local_unnamed_addr  {
entry:
  %v = alloca [4 x i32], align 4
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %z = alloca i32, align 4
  %0 = bitcast [4 x i32]* %v to i8*
  %1 = bitcast i32* %x to i8*
  %2 = bitcast i32* %y to i8*
  %3 = bitcast i32* %z to i8*
  %call = call i32 @h(i32* nonnull %x, i32* nonnull %y, i32* nonnull %z)
  %arraydecay = getelementptr inbounds [4 x i32], [4 x i32]* %v, i32 0, i32 0
  %4 = load i32, i32* %x, align 4
  %5 = load i32, i32* %y, align 4
  %6 = load i32, i32* %z, align 4
  %call1 = call i32 @g(i32* nonnull %arraydecay, i32 %4, i32 %5, i32 %6, i32 0, i32 0)
  ret i32 %call1
}
; CHECK-LABEL: test_local
; Arguments to `h` relative to SP
; CHECK:       add r0, sp, #20
; CHECK-NEXT:  add r1, sp, #16
; CHECK-NEXT:  add r2, sp, #12
; CHECK-NEXT:  bl  h
; Load `x`, `y`, and `z` via SP
; CHECK:       ldr r1, [sp, #20]
; CHECK-NEXT:  ldr r2, [sp, #16]
; CHECK-NEXT:  ldr r3, [sp, #12]
; CHECK:       bl  g

; Re-aligned stack, access via SP.
; int test_local_realign(int n) {
;   __attribute__((aligned(16))) int v[4];
;   int x, y, z;
;   h(&x, &y, &z);
;   return g(v, x, y, z, 0, 0);
; }
define dso_local i32 @test_local_realign(i32 %n) local_unnamed_addr  {
entry:
  %v = alloca [4 x i32], align 16
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %z = alloca i32, align 4
  %0 = bitcast [4 x i32]* %v to i8*
  %1 = bitcast i32* %x to i8*
  %2 = bitcast i32* %y to i8*
  %3 = bitcast i32* %z to i8*
  %call = call i32 @h(i32* nonnull %x, i32* nonnull %y, i32* nonnull %z)
  %arraydecay = getelementptr inbounds [4 x i32], [4 x i32]* %v, i32 0, i32 0
  %4 = load i32, i32* %x, align 4
  %5 = load i32, i32* %y, align 4
  %6 = load i32, i32* %z, align 4
  %call1 = call i32 @g(i32* nonnull %arraydecay, i32 %4, i32 %5, i32 %6, i32 0, i32 0)
  ret i32 %call1
}
; CHECK-LABEL: test_local_realign
; Setup frame pointer
; CHECK-ATPCS: add r7, sp, #8
; CHECK-AAPCS: mov r11, sp
; Re-align stack
; CHECK:       mov r4, sp
; CHECK-NEXT:  lsrs r4, r4, #4
; CHECK-NEXT:  lsls r4, r4, #4
; CHECK-NEXT:  mov  sp, r4
; Arguments to `h` computed relative to SP
; CHECK:       add r0, sp, #28
; CHECK-NEXT:  add r1, sp, #24
; CHECK-NEXT:  add r2, sp, #20
; CHECK-NEXT:  bl  h
; Load `x`, `y`, and `z` via SP for passing to `g`
; CHECK:       ldr r1, [sp, #28]
; CHECK-NEXT:  ldr r2, [sp, #24]
; CHECK-NEXT:  ldr r3, [sp, #20]
; CHECK:       bl  g

; VLAs, access via BP.
; int test_local_vla(int n) {
;   int v[n];
;   int x, y, z;
;   h(&x, &y, &z);
;   return g(v, x, y, z, 0, 0);
; }
define dso_local i32 @test_local_vla(i32 %n) local_unnamed_addr  {
entry:
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %z = alloca i32, align 4
  %vla = alloca i32, i32 %n, align 4
  %0 = bitcast i32* %x to i8*
  %1 = bitcast i32* %y to i8*
  %2 = bitcast i32* %z to i8*
  %call = call i32 @h(i32* nonnull %x, i32* nonnull %y, i32* nonnull %z)
  %3 = load i32, i32* %x, align 4
  %4 = load i32, i32* %y, align 4
  %5 = load i32, i32* %z, align 4
  %call1 = call i32 @g(i32* nonnull %vla, i32 %3, i32 %4, i32 %5, i32 0, i32 0)
  ret i32 %call1
}
; CHECK-LABEL: test_local_vla
; Setup frame pointer
; CHECK-ATPCS: add r7, sp, #12
; CHECK-AAPCS: mov r11, sp
; Locas area
; CHECK-ATPCS: sub sp, #12
; CHECK-AAPCS: sub sp, #16
; Setup base pointer
; CHECK:       mov  r6, sp
; CHECK-ATPCS: mov  r5, r6
; CHECK-AAPCS: adds  r5, r6, #4
; Arguments to `h` compute relative to BP
; CHECK:       adds r0, r6, #7
; CHECK-ATPCS-NEXT:  adds r0, #1
; CHECK-ATPCS-NEXT:  adds r1, r6, #4
; CHECK-ATPCS-NEXT:  mov  r2, r6
; CHECK-AAPCS-NEXT:  adds r0, #5
; CHECK-AAPCS-NEXT:  adds r1, r6, #7
; CHECK-AAPCS-NEXT:  adds r1, #1
; CHECK-AAPCS-NEXT:  adds r2, r6, #4
; CHECK-NEXT:  bl   h
; Load `x`, `y`, `z` via BP (r5 should still have the value of r6 from the move
; above)
; CHECK:       ldr r3, [r5]
; CHECK-NEXT:  ldr r2, [r5, #4]
; CHECK-NEXT:  ldr r1, [r5, #8]
; CHECK:       bl  g

;  Moving SP, access via SP.
; int test_local_moving_sp(int n) {
;   int v[4];
;   int x, y, z;
;   return u(v, &x, &y, s, s) + u(v, &y, &z, s, s);
; }
define dso_local i32 @test_local_moving_sp(i32 %n) local_unnamed_addr {
entry:
  %v = alloca [4 x i32], align 4
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %z = alloca i32, align 4
  %0 = bitcast [4 x i32]* %v to i8*
  %1 = bitcast i32* %x to i8*
  %2 = bitcast i32* %y to i8*
  %3 = bitcast i32* %z to i8*
  %arraydecay = getelementptr inbounds [4 x i32], [4 x i32]* %v, i32 0, i32 0
  %call = call i32 @u(i32* nonnull %arraydecay, i32* nonnull %x, i32* nonnull %y, %struct.S* byval(%struct.S) nonnull align 4 @s, %struct.S* byval(%struct.S) nonnull align 4 @s)
  %call2 = call i32 @u(i32* nonnull %arraydecay, i32* nonnull %y, i32* nonnull %z, %struct.S* byval(%struct.S) nonnull align 4 @s, %struct.S* byval(%struct.S) nonnull align 4 @s)
  %add = add nsw i32 %call2, %call
  ret i32 %add
}
; CHECK-LABEL: test_local_moving_sp
; Locals area
; CHECK-NOFP: sub sp, #36
; CHECK-FP-ATPCS: sub sp, #44
; CHECK-FP-AAPCS: sub sp, #40
; Setup BP
; CHECK:      mov r6, sp
; Outoging arguments
; CHECK:      sub sp, #508
; CHECK-NEXT: sub sp, #508
; CHECK-NEXT: sub sp, #8
; Argument addresses computed relative to BP
; CHECK-NOFP:      adds r4, r6, #7
; CHECK-NOFP-NEXT: adds r4, #13
; CHECK-NOFP:      adds r1, r6, #7
; CHECK-NOFP-NEXT: adds r1, #9
; CHECK-NOFP:      adds r5, r6, #7
; CHECK-NOFP-NEXT: adds r5, #5
; CHECK-FP-ATPCS:      adds r0, r6, #7
; CHECK-FP-ATPCS-NEXT: adds r0, #21
; CHECK-FP-ATPCS:      adds r1, r6, #7
; CHECK-FP-ATPCS-NEXT: adds r1, #17
; CHECK-FP-ATPCS:      adds r5, r6, #7
; CHECK-FP-ATPCS-NEXT: adds r5, #13
; CHECK-FP-AAPCS:      adds r4, r6, #7
; CHECK-FP-AAPCS-NEXT: adds r4, #17
; CHECK-FP-AAPCS:      adds r1, r6, #7
; CHECK-FP-AAPCS-NEXT: adds r1, #13
; CHECK-FP-AAPCS:      adds r5, r6, #7
; CHECK-FP-AAPCS-NEXT: adds r5, #9
; CHECK:      bl   u
; Stack restored before next call
; CHECK:      add  sp, #508
; CHECK-NEXT: add  sp, #508
; CHECK-NEXT: add  sp, #8
; CHECK:      bl   u
