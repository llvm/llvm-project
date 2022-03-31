; RUN: opt < %s -passes=instcombine -S | FileCheck %s

@glob = global i32 0

declare i8* @llvm.stacksave()
declare void @llvm.stackrestore(i8*)

;; Test that llvm.stackrestore is removed when possible.
define i32* @test1(i32 %P) {
	%tmp = call i8* @llvm.stacksave( )
	call void @llvm.stackrestore( i8* %tmp ) ;; not restoring anything
	%A = alloca i32, i32 %P
	ret i32* %A
}

; CHECK-LABEL: define i32* @test1(
; CHECK-NOT: call void @llvm.stackrestore
; CHECK: ret i32*

define void @test2(i8* %X) {
	call void @llvm.stackrestore( i8* %X )  ;; no allocas before return.
	ret void
}

; CHECK-LABEL: define void @test2(
; CHECK-NOT: call void @llvm.stackrestore
; CHECK: ret void

define void @foo(i32 %size) nounwind  {
entry:
	%tmp118124 = icmp sgt i32 %size, 0		; <i1> [#uses=1]
	br i1 %tmp118124, label %bb.preheader, label %return

bb.preheader:		; preds = %entry
	%tmp25 = add i32 %size, -1		; <i32> [#uses=1]
	%tmp125 = icmp slt i32 %size, 1		; <i1> [#uses=1]
	%smax = select i1 %tmp125, i32 1, i32 %size		; <i32> [#uses=1]
	br label %bb

bb:		; preds = %bb, %bb.preheader
	%i.0.reg2mem.0 = phi i32 [ 0, %bb.preheader ], [ %indvar.next, %bb ]		; <i32> [#uses=2]
	%tmp = call i8* @llvm.stacksave( )		; <i8*> [#uses=1]
	%tmp23 = alloca i8, i32 %size		; <i8*> [#uses=2]
	%tmp27 = getelementptr i8, i8* %tmp23, i32 %tmp25		; <i8*> [#uses=1]
	store i8 0, i8* %tmp27, align 1
	%tmp28 = call i8* @llvm.stacksave( )		; <i8*> [#uses=1]
	%tmp52 = alloca i8, i32 %size		; <i8*> [#uses=1]
	%tmp53 = call i8* @llvm.stacksave( )		; <i8*> [#uses=1]
	%tmp77 = alloca i8, i32 %size		; <i8*> [#uses=1]
	%tmp78 = call i8* @llvm.stacksave( )		; <i8*> [#uses=1]
	%tmp102 = alloca i8, i32 %size		; <i8*> [#uses=1]
	call void @bar( i32 %i.0.reg2mem.0, i8* %tmp23, i8* %tmp52, i8* %tmp77, i8* %tmp102, i32 %size ) nounwind
	call void @llvm.stackrestore( i8* %tmp78 )
	call void @llvm.stackrestore( i8* %tmp53 )
	call void @llvm.stackrestore( i8* %tmp28 )
	call void @llvm.stackrestore( i8* %tmp )
	%indvar.next = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, %smax		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %bb

return:		; preds = %bb, %entry
	ret void
}

; CHECK-LABEL: define void @foo(
; CHECK: %tmp = call i8* @llvm.stacksave()
; CHECK: alloca i8
; CHECK-NOT: stacksave
; CHECK: call void @bar(
; CHECK-NEXT: call void @llvm.stackrestore(i8* %tmp)
; CHECK: ret void

declare void @bar(i32, i8*, i8*, i8*, i8*, i32)

declare void @inalloca_callee(i32* inalloca(i32))

define void @test3(i32 %c) {
entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%i1, %loop]
  %save1 = call i8* @llvm.stacksave()
  %argmem = alloca inalloca i32
  store i32 0, i32* %argmem
  call void @inalloca_callee(i32* inalloca(i32) %argmem)

  ; This restore cannot be deleted, the restore below does not make it dead.
  call void @llvm.stackrestore(i8* %save1)

  ; FIXME: We should be able to remove this save/restore pair, but we don't.
  %save2 = call i8* @llvm.stacksave()
  store i32 0, i32* @glob
  call void @llvm.stackrestore(i8* %save2)
  %i1 = add i32 1, %i
  %done = icmp eq i32 %i1, %c
  br i1 %done, label %loop, label %return

return:
  ret void
}

; CHECK-LABEL: define void @test3(
; CHECK: loop:
; CHECK: %i = phi i32 [ 0, %entry ], [ %i1, %loop ]
; CHECK: %save1 = call i8* @llvm.stacksave()
; CHECK: %argmem = alloca inalloca i32
; CHECK: store i32 0, i32* %argmem
; CHECK: call void @inalloca_callee(i32* {{.*}} inalloca(i32) %argmem)
; CHECK: call void @llvm.stackrestore(i8* %save1)
; CHECK: br i1 %done, label %loop, label %return
; CHECK: ret void

define i32 @test4(i32 %m, i32* %a, i32* %b) {
entry:
  br label %for.body

for.body:
  %x.012 = phi i32 [ 0, %entry ], [ %add2, %for.body ]
  %i.011 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %0 = call i8* @llvm.stacksave()
  %load1 = load i32, i32* %a, align 4
  %mul1 = mul nsw i32 %load1, %m
  %add1 = add nsw i32 %mul1, %x.012
  call void @llvm.stackrestore(i8* %0)
  %load2 = load i32, i32* %b, align 4
  %mul2 = mul nsw i32 %load2, %m
  %add2 = add nsw i32 %mul2, %add1
  call void @llvm.stackrestore(i8* %0)
  %inc = add nuw nsw i32 %i.011, 1
  %exitcond.not = icmp eq i32 %inc, 100
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i32 %add2
}

; CHECK-LABEL: define i32 @test4(
; CHECK-NOT: call void @llvm.stackrestore
; CHECK: ret i32 %add2
