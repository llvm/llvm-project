; RUN: opt -safe-stack -safe-stack-coloring=1 -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -safe-stack-coloring=1 -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

; x and y share the stack slot.
define void @f() safestack {
; CHECK-LABEL: define void @f
entry:
; CHECK:  %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -16

  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %z = alloca i32, align 4
  %x0 = bitcast i32* %x to i8*
  %y0 = bitcast i32* %y to i8*
  %z0 = bitcast i32* %z to i8*

  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %z0)
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %x0)

; CHECK:   getelementptr i8, i8* %[[USP]], i32 -4
  call void @capture32(i32* %x)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %x0)
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %y0)

; CHECK:   getelementptr i8, i8* %[[USP]], i32 -4
  call void @capture32(i32* %y)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %y0)

; CHECK:   getelementptr i8, i8* %[[USP]], i32 -8
  call void @capture32(i32* %z)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %z0)

  ret void
}

define void @no_markers() safestack {
; CHECK-LABEL: define void @no_markers(
entry:
; CHECK:  %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -16

  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %x0 = bitcast i32* %x to i8*

  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %x0)

; CHECK:   getelementptr i8, i8* %[[USP]], i32 -4
  call void @capture32(i32* %x)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %x0)

; CHECK:   getelementptr i8, i8* %[[USP]], i32 -8
  call void @capture32(i32* %y)

  ret void
}

; x and y can't share memory, but they can split z's storage.
define void @g() safestack {
; CHECK-LABEL: define void @g
entry:
; CHECK:  %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -16

  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %z = alloca i64, align 4
  %x0 = bitcast i32* %x to i8*
  %y0 = bitcast i32* %y to i8*
  %z0 = bitcast i64* %z to i8*

  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %x0)
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %y0)

; CHECK:   getelementptr i8, i8* %[[USP]], i32 -4
  call void @capture32(i32* %x)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %x0)

; CHECK:   getelementptr i8, i8* %[[USP]], i32 -8
  call void @capture32(i32* %y)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %y0)
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %z0)

; CHECK:   getelementptr i8, i8* %[[USP]], i32 -8
  call void @capture64(i64* %z)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %z0)

  ret void
}

; Both y and z fit in x's alignment gap.
define void @h() safestack {
; CHECK-LABEL: define void @h
entry:
; CHECK:  %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -16

  %x = alloca i32, align 16
  %z = alloca i64, align 4
  %y = alloca i32, align 4
  %x0 = bitcast i32* %x to i8*
  %y0 = bitcast i32* %y to i8*
  %z0 = bitcast i64* %z to i8*

  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %x0)
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %y0)
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %z0)

; CHECK:   getelementptr i8, i8* %[[USP]], i32 -16
  call void @capture32(i32* %x)

; CHECK:   getelementptr i8, i8* %[[USP]], i32 -12
  call void @capture32(i32* %y)

; CHECK:   getelementptr i8, i8* %[[USP]], i32 -8
  call void @capture64(i64* %z)

  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %x0)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %y0)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %z0)

  ret void
}

; void f(bool a, bool b) {
;   long x1, x2; capture64(&x1); capture64(&x2);
;   if (a) {
;     long y; capture64(&y);
;     if (b) {
;       long y1; capture64(&y1);
;     } else {
;       long y2; capture64(&y2);
;     }
;   } else {
;     long z; capture64(&z);
;     if (b) {
;       long z1; capture64(&z1);
;     } else {
;       long z2; capture64(&z2);
;     }
;   }
; }
; Everything fits in 4 x 64-bit slots.
define void @i(i1 zeroext %a, i1 zeroext %b) safestack {
; CHECK-LABEL: define void @i
entry:
; CHECK:        %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
; CHECK-NEXT:   getelementptr i8, i8* %[[USP]], i32 -32
  %x1 = alloca i64, align 8
  %x2 = alloca i64, align 8
  %y = alloca i64, align 8
  %y1 = alloca i64, align 8
  %y2 = alloca i64, align 8
  %z = alloca i64, align 8
  %z1 = alloca i64, align 8
  %z2 = alloca i64, align 8
  %0 = bitcast i64* %x1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %0)
  %1 = bitcast i64* %x2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %1)
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -8
; CHECK:   call void @capture64(
  call void @capture64(i64* nonnull %x1)
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -16
; CHECK:   call void @capture64(
  call void @capture64(i64* nonnull %x2)
  br i1 %a, label %if.then, label %if.else4

if.then:                                          ; preds = %entry
  %2 = bitcast i64* %y to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %2)
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -24
; CHECK:   call void @capture64(
  call void @capture64(i64* nonnull %y)
  br i1 %b, label %if.then3, label %if.else

if.then3:                                         ; preds = %if.then
  %3 = bitcast i64* %y1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %3)
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -32
; CHECK:   call void @capture64(
  call void @capture64(i64* nonnull %y1)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %3)
  br label %if.end

if.else:                                          ; preds = %if.then
  %4 = bitcast i64* %y2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %4)
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -32
; CHECK:   call void @capture64(
  call void @capture64(i64* nonnull %y2)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %4)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then3
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %2)
  br label %if.end9

if.else4:                                         ; preds = %entry
  %5 = bitcast i64* %z to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %5)
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -24
; CHECK:   call void @capture64(
  call void @capture64(i64* nonnull %z)
  br i1 %b, label %if.then6, label %if.else7

if.then6:                                         ; preds = %if.else4
  %6 = bitcast i64* %z1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %6)
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -32
; CHECK:   call void @capture64(
  call void @capture64(i64* nonnull %z1)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %6)
  br label %if.end8

if.else7:                                         ; preds = %if.else4
  %7 = bitcast i64* %z2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %7)
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -32
; CHECK:   call void @capture64(
  call void @capture64(i64* nonnull %z2)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %7)
  br label %if.end8

if.end8:                                          ; preds = %if.else7, %if.then6
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %5)
  br label %if.end9

if.end9:                                          ; preds = %if.end8, %if.end
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %1)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %0)
  ret void
}

; lifetime for x ends in 2 different BBs
define void @no_merge1(i1 %d) safestack {
; CHECK-LABEL: define void @no_merge1(
entry:
; CHECK:        %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
; CHECK-NEXT:   getelementptr i8, i8* %[[USP]], i32 -16
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %x0 = bitcast i32* %x to i8*
  %y0 = bitcast i32* %y to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %x0)
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -4
; CHECK:   call void @capture32(
  call void @capture32(i32* %x)
  br i1 %d, label %bb2, label %bb3
bb2:
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %y0)
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -8
; CHECK:   call void @capture32(
  call void @capture32(i32* %y)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %y0)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %x0)
  ret void
bb3:
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %x0)
  ret void
}

define void @merge1(i1 %d) safestack {
; CHECK-LABEL: define void @merge1(
entry:
; CHECK:        %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
; CHECK-NEXT:   getelementptr i8, i8* %[[USP]], i32 -16
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %x0 = bitcast i32* %x to i8*
  %y0 = bitcast i32* %y to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %x0)
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -4
; CHECK:   call void @capture32(
  call void @capture32(i32* %x)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %x0)
  br i1 %d, label %bb2, label %bb3
bb2:
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %y0)
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -4
; CHECK:   call void @capture32(
  call void @capture32(i32* %y)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %y0)
  ret void
bb3:
  ret void
}

; Missing lifetime.end
define void @merge2_noend(i1 %d) safestack {
; CHECK-LABEL: define void @merge2_noend(
entry:
; CHECK:        %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
; CHECK-NEXT:   getelementptr i8, i8* %[[USP]], i32 -16
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %x0 = bitcast i32* %x to i8*
  %y0 = bitcast i32* %y to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %x0)
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -4
; CHECK:   call void @capture32(
  call void @capture32(i32* %x)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %x0)
  br i1 %d, label %bb2, label %bb3
bb2:
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %y0)
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -4
; CHECK:   call void @capture32(
  call void @capture32(i32* %y)
  ret void
bb3:
  ret void
}

; Missing lifetime.end
define void @merge3_noend(i1 %d) safestack {
; CHECK-LABEL: define void @merge3_noend(
entry:
; CHECK:        %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
; CHECK-NEXT:   getelementptr i8, i8* %[[USP]], i32 -16
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %x0 = bitcast i32* %x to i8*
  %y0 = bitcast i32* %y to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %x0)
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -4
; CHECK:   call void @capture32(
  call void @capture32(i32* %x)
  br i1 %d, label %bb2, label %bb3
bb2:
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %x0)
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %y0)
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -4
; CHECK:   call void @capture32(
  call void @capture32(i32* %y)
  ret void
bb3:
  ret void
}

; Missing lifetime.start
define void @nomerge4_nostart(i1 %d) safestack {
; CHECK-LABEL: define void @nomerge4_nostart(
entry:
; CHECK:        %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
; CHECK-NEXT:   getelementptr i8, i8* %[[USP]], i32 -16
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %x0 = bitcast i32* %x to i8*
  %y0 = bitcast i32* %y to i8*
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -4
; CHECK:   call void @capture32(
  call void @capture32(i32* %x)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %x0)
  br i1 %d, label %bb2, label %bb3
bb2:
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %y0)
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -8
; CHECK:   call void @capture32(
  call void @capture32(i32* %y)
  ret void
bb3:
  ret void
}

define void @array_merge() safestack {
; CHECK-LABEL: define void @array_merge(
entry:
; CHECK:        %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
; CHECK-NEXT:   getelementptr i8, i8* %[[USP]], i32 -800
  %A.i1 = alloca [100 x i32], align 4
  %B.i2 = alloca [100 x i32], align 4
  %A.i = alloca [100 x i32], align 4
  %B.i = alloca [100 x i32], align 4
  %0 = bitcast [100 x i32]* %A.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %0)
  %1 = bitcast [100 x i32]* %B.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %1)
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -400
; CHECK:   call void @capture100x32(
  call void @capture100x32([100 x i32]* %A.i)
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -800
; CHECK:   call void @capture100x32(
  call void @capture100x32([100 x i32]* %B.i)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %0)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %1)
  %2 = bitcast [100 x i32]* %A.i1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %2)
  %3 = bitcast [100 x i32]* %B.i2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %3)
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -400
; CHECK:   call void @capture100x32(
  call void @capture100x32([100 x i32]* %A.i1)
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -800
; CHECK:   call void @capture100x32(
  call void @capture100x32([100 x i32]* %B.i2)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %2)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %3)
  ret void
}

define void @myCall_pr15707() safestack {
; CHECK-LABEL: define void @myCall_pr15707(
entry:
; CHECK:        %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
; CHECK-NEXT:   getelementptr i8, i8* %[[USP]], i32 -200000
  %buf1 = alloca i8, i32 100000, align 16
  %buf2 = alloca i8, i32 100000, align 16

  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %buf1)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %buf1)

  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %buf1)
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %buf2)
  call void @capture8(i8* %buf1)
  call void @capture8(i8* %buf2)
  ret void
}

; Check that we don't assert and crash even when there are allocas
; outside the declared lifetime regions.
define void @bad_range() safestack {
; CHECK-LABEL: define void @bad_range(
entry:
; CHECK:        %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
; A.i and B.i unsafe, not merged
; CHECK-NEXT:   getelementptr i8, i8* %[[USP]], i32 -800
; A.i1 and B.i2 safe
; CHECK: = alloca [100 x i32], align 4
; CHECK: = alloca [100 x i32], align 4

  %A.i1 = alloca [100 x i32], align 4
  %B.i2 = alloca [100 x i32], align 4
  %A.i = alloca [100 x i32], align 4
  %B.i = alloca [100 x i32], align 4
  %0 = bitcast [100 x i32]* %A.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %0) nounwind
  %1 = bitcast [100 x i32]* %B.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %1) nounwind
  call void @capture100x32([100 x i32]* %A.i)
  call void @capture100x32([100 x i32]* %B.i)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %0) nounwind
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %1) nounwind
  br label %block2

block2:
  ; I am used outside the marked lifetime.
  call void @capture100x32([100 x i32]* %A.i)
  call void @capture100x32([100 x i32]* %B.i)
  ret void
}

%struct.Klass = type { i32, i32 }

define i32 @shady_range(i32 %argc, i8** nocapture %argv) safestack {
; CHECK-LABEL: define i32 @shady_range(
entry:
; CHECK:        %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
; CHECK-NEXT:   getelementptr i8, i8* %[[USP]], i32 -64
  %a.i = alloca [4 x %struct.Klass], align 16
  %b.i = alloca [4 x %struct.Klass], align 16
  %a8 = bitcast [4 x %struct.Klass]* %a.i to i8*
  %b8 = bitcast [4 x %struct.Klass]* %b.i to i8*
  ; I am used outside the lifetime zone below:
  %z2 = getelementptr inbounds [4 x %struct.Klass], [4 x %struct.Klass]* %a.i, i64 0, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %a8)
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %b8)
  call void @capture8(i8* %a8)
  call void @capture8(i8* %b8)
  %z3 = load i32, i32* %z2, align 16
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %a8)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %b8)
  ret i32 %z3
}

define void @end_loop() safestack {
; CHECK-LABEL: define void @end_loop()
entry:
; CHECK:        %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
; CHECK-NEXT:   getelementptr i8, i8* %[[USP]], i32 -16
  %x = alloca i8, align 4
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %x) nounwind
  br label %l2

l2:
  call void @capture8(i8* %x)
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %x) nounwind
  br label %l2
}

; Check that @x and @y get distinct stack slots => @x lifetime does not break
; when control re-enters l2.
define void @start_loop() safestack {
; CHECK-LABEL: define void @start_loop()
entry:
; CHECK:        %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
; CHECK-NEXT:   getelementptr i8, i8* %[[USP]], i32 -16
  %x = alloca i8, align 4
  %y = alloca i8, align 4
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %x) nounwind
  br label %l2

l2:
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -8
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %y) nounwind
  call void @capture8(i8* %y)
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %y) nounwind

; CHECK:   getelementptr i8, i8* %[[USP]], i32 -4
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %x) nounwind
  call void @capture8(i8* %x)
  br label %l2
}

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)
declare void @capture8(i8*)
declare void @capture32(i32*)
declare void @capture64(i64*)
declare void @capture100x32([100 x i32]*)
