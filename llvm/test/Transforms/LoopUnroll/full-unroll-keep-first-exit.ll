; RUN: opt -S -loop-unroll < %s | FileCheck %s
; RUN: opt -S -passes='require<opt-remark-emit>,loop(loop-unroll-full)' < %s | FileCheck %s

; Unroll twice, with first loop exit kept
; CHECK-LABEL: @s32_max1
; CHECK: do.body:
; CHECK:  store
; CHECK:  br i1 %cmp, label %do.body.1, label %do.end
; CHECK: do.end:
; CHECK:  ret void
; CHECK: do.body.1:
; CHECK:  store
; CHECK:  br label %do.end
define void @s32_max1(i32 %n, i32* %p) {
entry:
  %add = add i32 %n, 1
  br label %do.body

do.body:
  %i.0 = phi i32 [ %n, %entry ], [ %inc, %do.body ]
  %arrayidx = getelementptr i32, i32* %p, i32 %i.0
  store i32 %i.0, i32* %arrayidx, align 4
  %inc = add i32 %i.0, 1
  %cmp = icmp slt i32 %i.0, %add
  br i1 %cmp, label %do.body, label %do.end ; taken either 0 or 1 times

do.end:
  ret void
}

; Unroll thrice, with first loop exit kept
; CHECK-LABEL: @s32_max2
; CHECK: do.body:
; CHECK:  store
; CHECK:  br i1 %cmp, label %do.body.1, label %do.end
; CHECK: do.end:
; CHECK:  ret void
; CHECK: do.body.1:
; CHECK:  store
; CHECK:  store
; CHECK:  br label %do.end
define void @s32_max2(i32 %n, i32* %p) {
entry:
  %add = add i32 %n, 2
  br label %do.body

do.body:
  %i.0 = phi i32 [ %n, %entry ], [ %inc, %do.body ]
  %arrayidx = getelementptr i32, i32* %p, i32 %i.0
  store i32 %i.0, i32* %arrayidx, align 4
  %inc = add i32 %i.0, 1
  %cmp = icmp slt i32 %i.0, %add
  br i1 %cmp, label %do.body, label %do.end ; taken either 0 or 2 times

do.end:
  ret void
}

; Should not be unrolled
; CHECK-LABEL: @s32_maxx
; CHECK: do.body:
; CHECK: do.end:
; CHECK-NOT: do.body.1:
define void @s32_maxx(i32 %n, i32 %x, i32* %p) {
entry:
  %add = add i32 %x, %n
  br label %do.body

do.body:
  %i.0 = phi i32 [ %n, %entry ], [ %inc, %do.body ]
  %arrayidx = getelementptr i32, i32* %p, i32 %i.0
  store i32 %i.0, i32* %arrayidx, align 4
  %inc = add i32 %i.0, 1
  %cmp = icmp slt i32 %i.0, %add
  br i1 %cmp, label %do.body, label %do.end ; taken either 0 or x times

do.end:
  ret void
}

; Should not be unrolled
; CHECK-LABEL: @s32_max2_unpredictable_exit
; CHECK: do.body:
; CHECK: do.end:
; CHECK-NOT: do.body.1:
define void @s32_max2_unpredictable_exit(i32 %n, i32 %x, i32* %p) {
entry:
  %add = add i32 %n, 2
  br label %do.body

do.body:
  %i.0 = phi i32 [ %n, %entry ], [ %inc, %if.end ]
  %cmp = icmp eq i32 %i.0, %x
  br i1 %cmp, label %do.end, label %if.end ; unpredictable

if.end:
  %arrayidx = getelementptr i32, i32* %p, i32 %i.0
  store i32 %i.0, i32* %arrayidx, align 4
  %inc = add i32 %i.0, 1
  %cmp1 = icmp slt i32 %i.0, %add
  br i1 %cmp1, label %do.body, label %do.end ; taken either 0 or 2 times

do.end:
  ret void
}

; Unroll twice, with first loop exit kept
; CHECK-LABEL: @u32_max1
; CHECK: do.body:
; CHECK:  store
; CHECK:  br i1 %cmp, label %do.body.1, label %do.end
; CHECK: do.end:
; CHECK:  ret void
; CHECK: do.body.1:
; CHECK:  store
; CHECK:  br label %do.end
define void @u32_max1(i32 %n, i32* %p) {
entry:
  %add = add i32 %n, 1
  br label %do.body

do.body:
  %i.0 = phi i32 [ %n, %entry ], [ %inc, %do.body ]
  %arrayidx = getelementptr i32, i32* %p, i32 %i.0
  store i32 %i.0, i32* %arrayidx, align 4
  %inc = add i32 %i.0, 1
  %cmp = icmp ult i32 %i.0, %add
  br i1 %cmp, label %do.body, label %do.end ; taken either 0 or 1 times

do.end:
  ret void
}

; Unroll thrice, with first loop exit kept
; CHECK-LABEL: @u32_max2
; CHECK: do.body:
; CHECK:  store
; CHECK:  br i1 %cmp, label %do.body.1, label %do.end
; CHECK: do.end:
; CHECK:  ret void
; CHECK: do.body.1:
; CHECK:  store
; CHECK:  store
; CHECK:  br label %do.end
define void @u32_max2(i32 %n, i32* %p) {
entry:
  %add = add i32 %n, 2
  br label %do.body

do.body:
  %i.0 = phi i32 [ %n, %entry ], [ %inc, %do.body ]
  %arrayidx = getelementptr i32, i32* %p, i32 %i.0
  store i32 %i.0, i32* %arrayidx, align 4
  %inc = add i32 %i.0, 1
  %cmp = icmp ult i32 %i.0, %add
  br i1 %cmp, label %do.body, label %do.end ; taken either 0 or 2 times

do.end:
  ret void
}

; Should not be unrolled
; CHECK-LABEL: @u32_maxx
; CHECK: do.body:
; CHECK: do.end:
; CHECK-NOT: do.body.1:
define void @u32_maxx(i32 %n, i32 %x, i32* %p) {
entry:
  %add = add i32 %x, %n
  br label %do.body

do.body:
  %i.0 = phi i32 [ %n, %entry ], [ %inc, %do.body ]
  %arrayidx = getelementptr i32, i32* %p, i32 %i.0
  store i32 %i.0, i32* %arrayidx, align 4
  %inc = add i32 %i.0, 1
  %cmp = icmp ult i32 %i.0, %add
  br i1 %cmp, label %do.body, label %do.end ; taken either 0 or x times

do.end:
  ret void
}

; Should not be unrolled
; CHECK-LABEL: @u32_max2_unpredictable_exit
; CHECK: do.body:
; CHECK: do.end:
; CHECK-NOT: do.body.1:
define void @u32_max2_unpredictable_exit(i32 %n, i32 %x, i32* %p) {
entry:
  %add = add i32 %n, 2
  br label %do.body

do.body:
  %i.0 = phi i32 [ %n, %entry ], [ %inc, %if.end ]
  %cmp = icmp eq i32 %i.0, %x
  br i1 %cmp, label %do.end, label %if.end ; unpredictable

if.end:
  %arrayidx = getelementptr i32, i32* %p, i32 %i.0
  store i32 %i.0, i32* %arrayidx, align 4
  %inc = add i32 %i.0, 1
  %cmp1 = icmp ult i32 %i.0, %add
  br i1 %cmp1, label %do.body, label %do.end ; taken either 0 or 2 times

do.end:
  ret void
}
