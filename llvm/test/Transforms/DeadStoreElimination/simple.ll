; RUN: opt < %s -basicaa -dse -S | FileCheck %s
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind
declare void @llvm.init.trampoline(i8*, i8*, i8*)

define void @test1(i32* %Q, i32* %P) {
        %DEAD = load i32, i32* %Q
        store i32 %DEAD, i32* %P
        store i32 0, i32* %P
        ret void
; CHECK-LABEL: @test1(
; CHECK-NEXT: store i32 0, i32* %P
; CHECK-NEXT: ret void
}

; PR8576 - Should delete store of 10 even though p/q are may aliases.
define void @test2(i32 *%p, i32 *%q) {
  store i32 10, i32* %p, align 4
  store i32 20, i32* %q, align 4
  store i32 30, i32* %p, align 4
  ret void
; CHECK-LABEL: @test2(
; CHECK-NEXT: store i32 20
}


; PR8677
@g = global i32 1

define i32 @test3(i32* %g_addr) nounwind {
; CHECK-LABEL: @test3(
; CHECK: load i32, i32* %g_addr
  %g_value = load i32, i32* %g_addr, align 4
  store i32 -1, i32* @g, align 4
  store i32 %g_value, i32* %g_addr, align 4
  %tmp3 = load i32, i32* @g, align 4
  ret i32 %tmp3
}


define void @test4(i32* %Q) {
        %a = load i32, i32* %Q
        store volatile i32 %a, i32* %Q
        ret void
; CHECK-LABEL: @test4(
; CHECK-NEXT: load i32
; CHECK-NEXT: store volatile
; CHECK-NEXT: ret void
}

define void @test5(i32* %Q) {
        %a = load volatile i32, i32* %Q
        store i32 %a, i32* %Q
        ret void
; CHECK-LABEL: @test5(
; CHECK-NEXT: load volatile
; CHECK-NEXT: ret void
}

; Should delete store of 10 even though memset is a may-store to P (P and Q may
; alias).
define void @test6(i32 *%p, i8 *%q) {
  store i32 10, i32* %p, align 4       ;; dead.
  call void @llvm.memset.p0i8.i64(i8* %q, i8 42, i64 900, i32 1, i1 false)
  store i32 30, i32* %p, align 4
  ret void
; CHECK-LABEL: @test6(
; CHECK-NEXT: call void @llvm.memset
}

; Should delete store of 10 even though memcpy is a may-store to P (P and Q may
; alias).
define void @test7(i32 *%p, i8 *%q, i8* noalias %r) {
  store i32 10, i32* %p, align 4       ;; dead.
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %q, i8* %r, i64 900, i32 1, i1 false)
  store i32 30, i32* %p, align 4
  ret void
; CHECK-LABEL: @test7(
; CHECK-NEXT: call void @llvm.memcpy
}

; Do not delete stores that are only partially killed.
define i32 @test8() {
        %V = alloca i32
        store i32 1234567, i32* %V
        %V2 = bitcast i32* %V to i8*
        store i8 0, i8* %V2
        %X = load i32, i32* %V
        ret i32 %X
        
; CHECK-LABEL: @test8(
; CHECK: store i32 1234567
}


; Test for byval handling.
%struct.x = type { i32, i32, i32, i32 }
define void @test9(%struct.x* byval  %a) nounwind  {
	%tmp2 = getelementptr %struct.x, %struct.x* %a, i32 0, i32 0
	store i32 1, i32* %tmp2, align 4
	ret void
; CHECK-LABEL: @test9(
; CHECK-NEXT: ret void
}

; Test for inalloca handling.
define void @test9_2(%struct.x* inalloca  %a) nounwind  {
	%tmp2 = getelementptr %struct.x, %struct.x* %a, i32 0, i32 0
	store i32 1, i32* %tmp2, align 4
	ret void
; CHECK-LABEL: @test9_2(
; CHECK-NEXT: ret void
}

; va_arg has fuzzy dependence, the store shouldn't be zapped.
define double @test10(i8* %X) {
        %X_addr = alloca i8*
        store i8* %X, i8** %X_addr
        %tmp.0 = va_arg i8** %X_addr, double
        ret double %tmp.0
; CHECK-LABEL: @test10(
; CHECK: store
}


; DSE should delete the dead trampoline.
declare void @test11f()
define void @test11() {
; CHECK-LABEL: @test11(
	%storage = alloca [10 x i8], align 16		; <[10 x i8]*> [#uses=1]
; CHECK-NOT: alloca
	%cast = getelementptr [10 x i8], [10 x i8]* %storage, i32 0, i32 0		; <i8*> [#uses=1]
	call void @llvm.init.trampoline( i8* %cast, i8* bitcast (void ()* @test11f to i8*), i8* null )		; <i8*> [#uses=1]
; CHECK-NOT: trampoline
	ret void
; CHECK: ret void
}


; PR2599 - load -> store to same address.
define void @test12({ i32, i32 }* %x) nounwind  {
	%tmp4 = getelementptr { i32, i32 }, { i32, i32 }* %x, i32 0, i32 0
	%tmp5 = load i32, i32* %tmp4, align 4
	%tmp7 = getelementptr { i32, i32 }, { i32, i32 }* %x, i32 0, i32 1
	%tmp8 = load i32, i32* %tmp7, align 4
	%tmp17 = sub i32 0, %tmp8
	store i32 %tmp5, i32* %tmp4, align 4
	store i32 %tmp17, i32* %tmp7, align 4
	ret void
; CHECK-LABEL: @test12(
; CHECK-NOT: tmp5
; CHECK: ret void
}


; %P doesn't escape, the DEAD instructions should be removed.
declare void @test13f()
define i32* @test13() {
        %p = tail call i8* @malloc(i32 4)
        %P = bitcast i8* %p to i32*
        %DEAD = load i32, i32* %P
        %DEAD2 = add i32 %DEAD, 1
        store i32 %DEAD2, i32* %P
        call void @test13f( )
        store i32 0, i32* %P
        ret i32* %P
; CHECK: @test13()
; CHECK-NEXT: malloc
; CHECK-NEXT: bitcast
; CHECK-NEXT: call void
}

define i32 addrspace(1)* @test13_addrspacecast() {
  %p = tail call i8* @malloc(i32 4)
  %p.bc = bitcast i8* %p to i32*
  %P = addrspacecast i32* %p.bc to i32 addrspace(1)*
  %DEAD = load i32, i32 addrspace(1)* %P
  %DEAD2 = add i32 %DEAD, 1
  store i32 %DEAD2, i32 addrspace(1)* %P
  call void @test13f( )
  store i32 0, i32 addrspace(1)* %P
  ret i32 addrspace(1)* %P
; CHECK: @test13_addrspacecast()
; CHECK-NEXT: malloc
; CHECK-NEXT: bitcast
; CHECK-NEXT: addrspacecast
; CHECK-NEXT: call void
}

declare noalias i8* @malloc(i32)
declare noalias i8* @calloc(i32, i32)


define void @test14(i32* %Q) {
        %P = alloca i32
        %DEAD = load i32, i32* %Q
        store i32 %DEAD, i32* %P
        ret void

; CHECK-LABEL: @test14(
; CHECK-NEXT: ret void
}


; PR8701

;; Fully dead overwrite of memcpy.
define void @test15(i8* %P, i8* %Q) nounwind ssp {
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
  ret void
; CHECK-LABEL: @test15(
; CHECK-NEXT: call void @llvm.memcpy
; CHECK-NEXT: ret
}

;; Full overwrite of smaller memcpy.
define void @test16(i8* %P, i8* %Q) nounwind ssp {
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 8, i32 1, i1 false)
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
  ret void
; CHECK-LABEL: @test16(
; CHECK-NEXT: call void @llvm.memcpy
; CHECK-NEXT: ret
}

;; Overwrite of memset by memcpy.
define void @test17(i8* %P, i8* noalias %Q) nounwind ssp {
  tail call void @llvm.memset.p0i8.i64(i8* %P, i8 42, i64 8, i32 1, i1 false)
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
  ret void
; CHECK-LABEL: @test17(
; CHECK-NEXT: call void @llvm.memcpy
; CHECK-NEXT: ret
}

; Should not delete the volatile memset.
define void @test17v(i8* %P, i8* %Q) nounwind ssp {
  tail call void @llvm.memset.p0i8.i64(i8* %P, i8 42, i64 8, i32 1, i1 true)
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
  ret void
; CHECK-LABEL: @test17v(
; CHECK-NEXT: call void @llvm.memset
; CHECK-NEXT: call void @llvm.memcpy
; CHECK-NEXT: ret
}

; PR8728
; Do not delete instruction where possible situation is:
; A = B
; A = A
define void @test18(i8* %P, i8* %Q, i8* %R) nounwind ssp {
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %R, i64 12, i32 1, i1 false)
  ret void
; CHECK-LABEL: @test18(
; CHECK-NEXT: call void @llvm.memcpy
; CHECK-NEXT: call void @llvm.memcpy
; CHECK-NEXT: ret
}


; The store here is not dead because the byval call reads it.
declare void @test19f({i32}* byval align 4 %P)

define void @test19({i32} * nocapture byval align 4 %arg5) nounwind ssp {
bb:
  %tmp7 = getelementptr inbounds {i32}, {i32}* %arg5, i32 0, i32 0
  store i32 912, i32* %tmp7
  call void @test19f({i32}* byval align 4 %arg5)
  ret void

; CHECK-LABEL: @test19(
; CHECK: store i32 912
; CHECK: call void @test19f
}

define void @test20() {
  %m = call i8* @malloc(i32 24)
  store i8 0, i8* %m
  ret void
}
; CHECK-LABEL: @test20(
; CHECK-NEXT: ret void

; CHECK-LABEL: @test21(
define void @test21() {
  %m = call i8* @calloc(i32 9, i32 7)
  store i8 0, i8* %m
; CHECK-NEXT: ret void
  ret void
}

; CHECK-LABEL: @test22(
define void @test22(i1 %i, i32 %k, i32 %m) nounwind {
  %k.addr = alloca i32
  %m.addr = alloca i32
  %k.addr.m.addr = select i1 %i, i32* %k.addr, i32* %m.addr
  store i32 0, i32* %k.addr.m.addr, align 4
; CHECK-NEXT: ret void
  ret void
}

; PR13547
; CHECK-LABEL: @test23(
; CHECK: store i8 97
; CHECK: store i8 0
declare noalias i8* @strdup(i8* nocapture) nounwind
define noalias i8* @test23() nounwind uwtable ssp {
  %x = alloca [2 x i8], align 1
  %arrayidx = getelementptr inbounds [2 x i8], [2 x i8]* %x, i64 0, i64 0
  store i8 97, i8* %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds [2 x i8], [2 x i8]* %x, i64 0, i64 1
  store i8 0, i8* %arrayidx1, align 1
  %call = call i8* @strdup(i8* %arrayidx) nounwind
  ret i8* %call
}

; Make sure same sized store to later element is deleted
; CHECK-LABEL: @test24(
; CHECK-NOT: store i32 0
; CHECK-NOT: store i32 0
; CHECK: store i32 %b
; CHECK: store i32 %c
; CHECK: ret void
define void @test24([2 x i32]* %a, i32 %b, i32 %c) nounwind {
  %1 = getelementptr inbounds [2 x i32], [2 x i32]* %a, i64 0, i64 0
  store i32 0, i32* %1, align 4
  %2 = getelementptr inbounds [2 x i32], [2 x i32]* %a, i64 0, i64 1
  store i32 0, i32* %2, align 4
  %3 = getelementptr inbounds [2 x i32], [2 x i32]* %a, i64 0, i64 0
  store i32 %b, i32* %3, align 4
  %4 = getelementptr inbounds [2 x i32], [2 x i32]* %a, i64 0, i64 1
  store i32 %c, i32* %4, align 4
  ret void
}

; Check another case like PR13547 where strdup is not like malloc.
; CHECK-LABEL: @test25(
; CHECK: load i8
; CHECK: store i8 0
; CHECK: store i8 %tmp
define i8* @test25(i8* %p) nounwind {
  %p.4 = getelementptr i8, i8* %p, i64 4
  %tmp = load i8, i8* %p.4, align 1
  store i8 0, i8* %p.4, align 1
  %q = call i8* @strdup(i8* %p) nounwind optsize
  store i8 %tmp, i8* %p.4, align 1
  ret i8* %q
}

; Remove redundant store if loaded value is in another block.
; CHECK-LABEL: @test26(
; CHECK-NOT: store
; CHECK: ret
define i32 @test26(i1 %c, i32* %p) {
entry:
  %v = load i32, i32* %p, align 4
  br i1 %c, label %bb1, label %bb2
bb1:
  br label %bb3
bb2:
  store i32 %v, i32* %p, align 4
  br label %bb3
bb3:
  ret i32 0
}

; Remove redundant store if loaded value is in another block.
; CHECK-LABEL: @test27(
; CHECK-NOT: store
; CHECK: ret
define i32 @test27(i1 %c, i32* %p) {
entry:
  %v = load i32, i32* %p, align 4
  br i1 %c, label %bb1, label %bb2
bb1:
  br label %bb3
bb2:
  br label %bb3
bb3:
  store i32 %v, i32* %p, align 4
  ret i32 0
}

; Don't remove redundant store because of may-aliased store.
; CHECK-LABEL: @test28(
; CHECK: bb3:
; CHECK-NEXT: store i32 %v
define i32 @test28(i1 %c, i32* %p, i32* %p2, i32 %i) {
entry:
  %v = load i32, i32* %p, align 4

  ; Might overwrite value at %p
  store i32 %i, i32* %p2, align 4
  br i1 %c, label %bb1, label %bb2
bb1:
  br label %bb3
bb2:
  br label %bb3
bb3:
  store i32 %v, i32* %p, align 4
  ret i32 0
}

; Don't remove redundant store because of may-aliased store.
; CHECK-LABEL: @test29(
; CHECK: bb3:
; CHECK-NEXT: store i32 %v
define i32 @test29(i1 %c, i32* %p, i32* %p2, i32 %i) {
entry:
  %v = load i32, i32* %p, align 4
  br i1 %c, label %bb1, label %bb2
bb1:
  br label %bb3
bb2:
  ; Might overwrite value at %p
  store i32 %i, i32* %p2, align 4
  br label %bb3
bb3:
  store i32 %v, i32* %p, align 4
  ret i32 0
}

declare void @unknown_func()

; Don't remove redundant store because of unknown call.
; CHECK-LABEL: @test30(
; CHECK: bb3:
; CHECK-NEXT: store i32 %v
define i32 @test30(i1 %c, i32* %p, i32 %i) {
entry:
  %v = load i32, i32* %p, align 4
  br i1 %c, label %bb1, label %bb2
bb1:
  br label %bb3
bb2:
  ; Might overwrite value at %p
  call void @unknown_func()
  br label %bb3
bb3:
  store i32 %v, i32* %p, align 4
  ret i32 0
}

; Remove redundant store if loaded value is in another block inside a loop.
; CHECK-LABEL: @test31(
; CHECK-NOT: store
; CHECK: ret
define i32 @test31(i1 %c, i32* %p, i32 %i) {
entry:
  %v = load i32, i32* %p, align 4
  br label %bb1
bb1:
  store i32 %v, i32* %p, align 4
  br i1 undef, label %bb1, label %bb2
bb2:
  ret i32 0
}

; Don't remove redundant store in a loop with a may-alias store.
; CHECK-LABEL: @test32(
; CHECK: bb1:
; CHECK-NEXT: store i32 %v
; CHECK-NEXT: call void @unknown_func
define i32 @test32(i1 %c, i32* %p, i32 %i) {
entry:
  %v = load i32, i32* %p, align 4
  br label %bb1
bb1:
  store i32 %v, i32* %p, align 4
  ; Might read and overwrite value at %p
  call void @unknown_func()
  br i1 undef, label %bb1, label %bb2
bb2:
  ret i32 0
}

; Remove redundant store, which is in the lame loop as the load.
; CHECK-LABEL: @test33(
; CHECK-NOT: store
; CHECK: ret
define i32 @test33(i1 %c, i32* %p, i32 %i) {
entry:
  br label %bb1
bb1:
  %v = load i32, i32* %p, align 4
  br label %bb2
bb2:
  store i32 %v, i32* %p, align 4
  ; Might read and overwrite value at %p, but doesn't matter.
  call void @unknown_func()
  br i1 undef, label %bb1, label %bb3
bb3:
  ret i32 0
}

