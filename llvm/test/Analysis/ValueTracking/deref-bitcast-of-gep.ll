; RUN: opt -S -licm < %s | FileCheck %s

; Note: the !invariant.load is there just solely to let us call @use()
; to add a fake use, and still have the aliasing work out.  The call
; to @use(0) is just to provide a may-unwind exit out of the loop, so
; that LICM cannot hoist out the load simply because it is guaranteed
; to execute.

declare void @use(i32)

define void @f_0(i8* align 4 dereferenceable(1024) %ptr) {
; CHECK-LABEL: @f_0(
; CHECK: entry:
; CHECK:  %val = load i32, i32* %ptr.i32
; CHECK:  br label %loop
; CHECK: loop:
; CHECK:  call void @use(i32 0)
; CHECK-NEXT:  call void @use(i32 %val)


entry:
  %ptr.gep = getelementptr i8, i8* %ptr, i32 32
  %ptr.i32 = bitcast i8* %ptr.gep to i32*
  br label %loop

loop:
  call void @use(i32 0)
  %val = load i32, i32* %ptr.i32, !invariant.load !{}
  call void @use(i32 %val)
  br label %loop
}

define void @f_1(i8* align 4 dereferenceable_or_null(1024) %ptr) {
; CHECK-LABEL: @f_1(
entry:
  %ptr.gep = getelementptr i8, i8* %ptr, i32 32
  %ptr.i32 = bitcast i8* %ptr.gep to i32*
  %ptr_is_null = icmp eq i8* %ptr, null
  br i1 %ptr_is_null, label %leave, label %loop

; CHECK: loop.preheader:
; CHECK:   %val = load i32, i32* %ptr.i32
; CHECK:   br label %loop
; CHECK: loop:
; CHECK:  call void @use(i32 0)
; CHECK-NEXT:  call void @use(i32 %val)

loop:
  call void @use(i32 0)
  %val = load i32, i32* %ptr.i32, !invariant.load !{}
  call void @use(i32 %val)
  br label %loop

leave:
  ret void
}

define void @f_2(i8* align 4 dereferenceable_or_null(1024) %ptr) {
; CHECK-LABEL: @f_2(
; CHECK-NOT: load
; CHECK:  call void @use(i32 0)
; CHECK-NEXT:  %val = load i32, i32* %ptr.i32, !invariant.load !0
; CHECK-NEXT:  call void @use(i32 %val)

entry:
  ;; Can't hoist, since the alignment does not work out -- (<4 byte
  ;; aligned> + 30) is not necessarily 4 byte aligned.

  %ptr.gep = getelementptr i8, i8* %ptr, i32 30
  %ptr.i32 = bitcast i8* %ptr.gep to i32*
  %ptr_is_null = icmp eq i8* %ptr, null
  br i1 %ptr_is_null, label %leave, label %loop

loop:
  call void @use(i32 0)
  %val = load i32, i32* %ptr.i32, !invariant.load !{}
  call void @use(i32 %val)
  br label %loop

leave:
  ret void
}
