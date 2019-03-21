; RUN: llc -mtriple=thumbv6m-eabi -frame-pointer=none %s -o - | FileCheck %s

; struct S { int x[128]; } s;
; int f(int *, int, int, int, struct S);
; int g(int *, int, int, int, int, int);
; int h(int *, int *, int *);
; int u(int *, int *, int *, struct S, struct S);

%struct.S = type { [128 x i32] }
%struct.__va_list = type { i8* }

@s = common dso_local global %struct.S zeroinitializer, align 4

declare void @llvm.va_start(i8*)
declare dso_local i32 @g(i32*, i32, i32, i32, i32, i32) local_unnamed_addr
declare dso_local i32 @f(i32*, i32, i32, i32, %struct.S* byval align 4) local_unnamed_addr
declare dso_local i32 @h(i32*, i32*, i32*) local_unnamed_addr
declare dso_local i32 @u(i32*, i32*, i32*, %struct.S* byval align 4, %struct.S* byval align 4) local_unnamed_addr

;
; Test access to arguments, passed on stack (including varargs)
;

; Usual case, access via SP
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
; CHECK:       ldr    r0, [sp, #32]
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
; Incoming arguments area is accessed via SP
; CHECK:       add r0, sp, #36
; CHECK:       stm r0!, {r1, r2, r3}

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
; CHECK:       add r7, sp, #8
; Align stack
; CHECK:       mov  r4, sp
; CHECK-NEXT:  lsrs r4, r4, #4
; CHECK-NEXT:  lsls r4, r4, #4
; CHECK-NEXT:  mov  sp, r4
; Load `e` via FP
; CHECK:       ldr r0, [r7, #8]
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
; CHECK:       add r7, sp, #8
; Align stack
; CHECK:       mov  r4, sp
; CHECK-NEXT:  lsrs r4, r4, #4
; CHECK-NEXT:  lsls r4, r4, #4
; CHECK-NEXT:  mov  sp, r4
; Incoming register varargs stored via FP
; CHECK: mov	r0, r7
; CHECK-NEXT: adds r0, #8
; CHECK-NEXT: stm r0!, {r1, r2, r3}
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
; CHECK:       add r7, sp, #12
; Allocate outgoing stack arguments space
; CHECK:       sub sp, #4
; Load `e` via FP
; CHECK:       ldr r5, [r7, #8]
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
; CHECK:       add r7, sp, #8
; Register varargs stored via FP
; CHECK-DAG:  str r3, [r7, #16]
; CHECK-DAG:  str r2, [r7, #12]
; CHECK-DAG:  str r1, [r7, #8]

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
  %call = call i32 @f(i32* nonnull %arraydecay, i32 %a, i32 %add1, i32 %e, %struct.S* byval nonnull align 4 @s)
  %add.ptr = getelementptr inbounds [4 x i32], [4 x i32]* %v, i32 0, i32 1
  %add.ptr5 = getelementptr inbounds [4 x i32], [4 x i32]* %v, i32 0, i32 2
  %call6 = call i32 @h(i32* nonnull %arraydecay, i32* nonnull %add.ptr, i32* nonnull %add.ptr5)
  %add7 = add nsw i32 %call6, %call
  ret i32 %add7
}
; CHECK-LABEL: test_args_moving_sp
; 20 bytes callee-saved area
; CHECK:       push {r4, r5, r6, r7, lr}
; 20 bytes locals
; CHECK:       sub sp, #20
; Allocate outgoing arguments space
; CHECK:       sub sp, #508
; CHECK:       sub sp, #4
; Load `e` via SP, 552 = 512 + 20 + 20
; CHECK:       ldr r3, [sp, #552]
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
  %call = call i32 @f(i32* nonnull %arraydecay, i32 %a, i32 0, i32 0, %struct.S* byval nonnull align 4 @s)
  %add.ptr = getelementptr inbounds [4 x i32], [4 x i32]* %v, i32 0, i32 1
  %add.ptr5 = getelementptr inbounds [4 x i32], [4 x i32]* %v, i32 0, i32 2
  %call6 = call i32 @h(i32* nonnull %arraydecay, i32* nonnull %add.ptr, i32* nonnull %add.ptr5)
  %add = add nsw i32 %call6, %call
  ret i32 %add
}
; CHECK-LABEL: test_varargs_moving_sp
; Three incoming register varargs
; CHECK:       sub sp, #12
; 16 bytes callee-saves
; CHECK:       push {r4, r5, r7, lr}
; 20 bytes locals
; CHECK:       sub sp, #20
; Incoming varargs stored via SP, 36 = 20 + 16
; CHECK:       add r0, sp, #36
; CHECK-NEXT:  stm r0!, {r1, r2, r3}

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
; CHECK:       add r7, sp, #8
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
; CHECK:       add  r7, sp, #12
; Setup base pointer
; CHECK:       mov  r6, sp
; CHECK:       mov  r5, r6
; Arguments to `h` compute relative to BP
; CHECK:       adds r0, r6, #7
; CHECK-NEXT:  adds r0, #1
; CHECK-NEXT:  adds r1, r6, #4
; CHECK-NEXT:  mov  r2, r6
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
  %call = call i32 @u(i32* nonnull %arraydecay, i32* nonnull %x, i32* nonnull %y, %struct.S* byval nonnull align 4 @s, %struct.S* byval nonnull align 4 @s)
  %call2 = call i32 @u(i32* nonnull %arraydecay, i32* nonnull %y, i32* nonnull %z, %struct.S* byval nonnull align 4 @s, %struct.S* byval nonnull align 4 @s)
  %add = add nsw i32 %call2, %call
  ret i32 %add
}
; CHECK-LABEL: test_local_moving_sp
; Locals area
; CHECK:      sub sp, #36
; Outoging arguments
; CHECK:      sub sp, #508
; CHECK-NEXT: sub sp, #508
; CHECK-NEXT: sub sp, #8
; Argument addresses computed relative to SP
; CHECK:      add  r4, sp, #1020
; CHECK-NEXT: adds r4, #24
; CHECK:      add  r1, sp, #1020
; CHECK-NEXT: adds r1, #20
; CHECK:      add  r5, sp, #1020
; CHECK-NEXT: adds r5, #16
; CHECK:      bl   u
; Stack restored before next call
; CHECK:      add  sp, #508
; CHECK-NEXT: add  sp, #508
; CHECK-NEXT: add  sp, #8
; CHECK:      bl   u
