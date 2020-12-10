; RUN: opt < %s -simplifycfg -S | FileCheck %s
; RUN: opt < %s -passes=simplifycfg -S | FileCheck %s

define i32 @foo(i32 %x) optforfuzzing {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  %0 = load i32, i32* %x.addr, align 4
  %cmp = icmp sgt i32 %0, 16
  br i1 %cmp, label %land.rhs, label %land.end

land.rhs:
  %1 = load i32, i32* %x.addr, align 4
  %cmp1 = icmp slt i32 %1, 32
  br label %land.end

land.end:
  %2 = phi i1 [ false, %entry ], [ %cmp1, %land.rhs ]
  %conv = zext i1 %2 to i32
  ret i32 %conv

; CHECK-LABEL: define i32 @foo(i32 %x)
; CHECK: br i1 %cmp, label %land.rhs, label %land.end
; CHECK-LABEL: land.rhs:
; CHECK: br label %land.end
; CHECK-LABEL: land.end:
; CHECK: phi {{.*}} %entry {{.*}} %land.rhs
}

define i32 @bar(i32 %x) {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  %0 = load i32, i32* %x.addr, align 4
  %cmp = icmp sgt i32 %0, 16
  br i1 %cmp, label %land.rhs, label %land.end

land.rhs:
  %1 = load i32, i32* %x.addr, align 4
  %cmp1 = icmp slt i32 %1, 32
  br label %land.end

land.end:
  %2 = phi i1 [ false, %entry ], [ %cmp1, %land.rhs ]
  %conv = zext i1 %2 to i32
  ret i32 %conv

; CHECK-LABEL: define i32 @bar(i32 %x)
; CHECK-NOT: br
}
