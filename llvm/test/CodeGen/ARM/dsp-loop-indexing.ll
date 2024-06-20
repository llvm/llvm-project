; RUN: llc -mtriple=thumbv7em -mattr=+fp-armv8 %s -o - | \
; RUN:     FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-DEFAULT

; RUN: llc -mtriple=thumbv8m.main -mattr=+fp-armv8,+dsp %s -o - | \
; RUN:     FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-DEFAULT

; -lsr-backedge-indexing=false

; RUN: llc -mtriple=thumbv8m.main -mattr=+fp-armv8,+dsp -lsr-preferred-addressing-mode=postindexed %s -o - | \
; RUN:     FileCheck %s --check-prefix=CHECK --check-prefix=DISABLED

; RUN: llc -mtriple=thumbv8 %s -o - | \
; RUN:     FileCheck %s --check-prefix=CHECK --check-prefix=DISABLED

; RUN: llc -mtriple=thumbv8m.main -mattr=+fp-armv8,+dsp -lsr-complexity-limit=2147483647 %s -o - | \
; RUN:     FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-COMPLEX

; CHECK-LABEL: test_qadd_2
; CHECK: @ %loop

; CHECK-DEFAULT: ldr{{.*}}, #4]
; CHECK-DEFAULT: ldr{{.*}}, #4]
; CHECK-DEFAULT: str{{.*}}, #4]
; CHECK-DEFAULT: ldr{{.*}}, #8]!
; CHECK-DEAFULT: ldr{{.*}}, #8]!
; CHECK-DEFAULT: str{{.*}}, #8]!

; CHECK-COMPLEX: ldr{{.*}}, #8]!
; CHECK-COMPLEX: ldr{{.*}}, #8]!
; CHECK-COMPLEX: str{{.*}}, #8]!
; CHECK-COMPLEX: ldr{{.*}}, #4]
; CHECK-COMPLEX: ldr{{.*}}, #4]
; CHECK-COMPLEX: str{{.*}}, #4]

; DISABLED-NOT: ldr{{.*}}]!
; DISABLED-NOT: str{{.*}}]!

define void @test_qadd_2(ptr %a.array, ptr %b.array, ptr %out.array, i32 %N) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %idx.1 = phi i32 [ 0, %entry ], [ %idx.next, %loop ]
  %gep.a.1 = getelementptr inbounds i32, ptr %a.array, i32 %idx.1
  %a.1 = load i32, ptr %gep.a.1
  %gep.b.1 = getelementptr inbounds i32, ptr %b.array, i32 %idx.1
  %b.1 = load i32, ptr %gep.b.1
  %qadd.1 = call i32 @llvm.arm.qadd(i32 %a.1, i32 %b.1)
  %addr.1 = getelementptr inbounds i32, ptr %out.array, i32 %idx.1
  store i32 %qadd.1, ptr %addr.1
  %idx.2 = or disjoint i32 %idx.1, 1
  %gep.a.2 = getelementptr inbounds i32, ptr %a.array, i32 %idx.2
  %a.2 = load i32, ptr %gep.a.2
  %gep.b.2 = getelementptr inbounds i32, ptr %b.array, i32 %idx.2
  %b.2 = load i32, ptr %gep.b.2
  %qadd.2 = call i32 @llvm.arm.qadd(i32 %a.2, i32 %b.2)
  %addr.2 = getelementptr inbounds i32, ptr %out.array, i32 %idx.2
  store i32 %qadd.2, ptr %addr.2
  %i.next = add nsw nuw i32 %i, -2
  %idx.next = add nsw nuw i32 %idx.1, 2
  %cmp = icmp ult i32 %i.next, %N
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; CHECK-LABEL: test_qadd_2_backwards
; TODO: Indexes should be generated.

; CHECK: @ %loop

; CHECK-DEFAULT: ldr{{.*}},
; CHECK-DEFAULT: ldr{{.*}},
; CHECK-DEFAULT: str{{.*}},
; CHECK-DEFAULT: ldr{{.*}}, #-4]
; CHECK-DEFAULT: ldr{{.*}}, #-4]
; CHECK-DEFAULT: sub{{.*}}, #8
; CHECK-DEFAULT: str{{.*}}, #-4]
; CHECK-DEFAULT: sub{{.*}}, #8

; CHECK-COMPLEX: ldr{{.*}} lsl #2]
; CHECK-COMPLEX: ldr{{.*}} lsl #2]
; CHECK-COMPLEX: str{{.*}} lsl #2]
; CHECK-COMPLEX: ldr{{.*}} lsl #2]
; CHECK-COMPLEX: ldr{{.*}} lsl #2]
; CHECK-COMPLEX: str{{.*}} lsl #2]

; DISABLED-NOT: ldr{{.*}}]!
; DISABLED-NOT: str{{.*}}]!

define void @test_qadd_2_backwards(ptr %a.array, ptr %b.array, ptr %out.array, i32 %N) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %idx.1 = phi i32 [ %N, %entry ], [ %idx.next, %loop ]
  %gep.a.1 = getelementptr inbounds i32, ptr %a.array, i32 %idx.1
  %a.1 = load i32, ptr %gep.a.1
  %gep.b.1 = getelementptr inbounds i32, ptr %b.array, i32 %idx.1
  %b.1 = load i32, ptr %gep.b.1
  %qadd.1 = call i32 @llvm.arm.qadd(i32 %a.1, i32 %b.1)
  %addr.1 = getelementptr inbounds i32, ptr %out.array, i32 %idx.1
  store i32 %qadd.1, ptr %addr.1
  %idx.2 = sub nsw nuw i32 %idx.1, 1
  %gep.a.2 = getelementptr inbounds i32, ptr %a.array, i32 %idx.2
  %a.2 = load i32, ptr %gep.a.2
  %gep.b.2 = getelementptr inbounds i32, ptr %b.array, i32 %idx.2
  %b.2 = load i32, ptr %gep.b.2
  %qadd.2 = call i32 @llvm.arm.qadd(i32 %a.2, i32 %b.2)
  %addr.2 = getelementptr inbounds i32, ptr %out.array, i32 %idx.2
  store i32 %qadd.2, ptr %addr.2
  %i.next = add nsw nuw i32 %i, -2
  %idx.next = sub nsw nuw i32 %idx.1, 2
  %cmp = icmp ult i32 %i.next, %N
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; CHECK-LABEL: test_qadd_3
; CHECK: @ %loop

; CHECK-DEFAULT: ldr{{.*}}, #8]
; CHECK-DEFAULT: ldr{{.*}}, #8]
; CHECK-DEFAULT: str{{.*}}, #8]
; CHECK-DEFAULT: ldr{{.*}}, #12]!
; CHECK-DEFAULT: ldr{{.*}}, #12]!
; CHECK-DEFAULT: str{{.*}}, #12]!

; CHECK-COMPLEX: ldr{{.*}}, #12]!
; CHECK-COMPLEX: ldr{{.*}}, #12]!
; CHECK-COMPLEX: str{{.*}}, #12]!
; CHECK-COMPLEX: ldr{{.*}}, #4]
; CHECK-COMPLEX: ldr{{.*}}, #4]
; CHECK-COMPLEX: str{{.*}}, #4]
; CHECK-COMPLEX: ldr{{.*}}, #8]
; CHECK-COMPLEX: ldr{{.*}}, #8]
; CHECK-COMPLEX: str{{.*}}, #8]

; DISABLED-NOT: ldr{{.*}}]!
; DISABLED-NOT: str{{.*}}]!

define void @test_qadd_3(ptr %a.array, ptr %b.array, ptr %out.array, i32 %N) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %idx.1 = phi i32 [ 0, %entry ], [ %idx.next, %loop ]
  %gep.a.1 = getelementptr inbounds i32, ptr %a.array, i32 %idx.1
  %a.1 = load i32, ptr %gep.a.1
  %gep.b.1 = getelementptr inbounds i32, ptr %b.array, i32 %idx.1
  %b.1 = load i32, ptr %gep.b.1
  %qadd.1 = call i32 @llvm.arm.qadd(i32 %a.1, i32 %b.1)
  %addr.1 = getelementptr inbounds i32, ptr %out.array, i32 %idx.1
  store i32 %qadd.1, ptr %addr.1
  %idx.2 = add nuw nsw i32 %idx.1, 1
  %gep.a.2 = getelementptr inbounds i32, ptr %a.array, i32 %idx.2
  %a.2 = load i32, ptr %gep.a.2
  %gep.b.2 = getelementptr inbounds i32, ptr %b.array, i32 %idx.2
  %b.2 = load i32, ptr %gep.b.2
  %qadd.2 = call i32 @llvm.arm.qadd(i32 %a.2, i32 %b.2)
  %addr.2 = getelementptr inbounds i32, ptr %out.array, i32 %idx.2
  store i32 %qadd.2, ptr %addr.2
  %idx.3 = add nuw nsw i32 %idx.1, 2
  %gep.a.3 = getelementptr inbounds i32, ptr %a.array, i32 %idx.3
  %a.3 = load i32, ptr %gep.a.3
  %gep.b.3 = getelementptr inbounds i32, ptr %b.array, i32 %idx.3
  %b.3 = load i32, ptr %gep.b.3
  %qadd.3 = call i32 @llvm.arm.qadd(i32 %a.3, i32 %b.3)
  %addr.3 = getelementptr inbounds i32, ptr %out.array, i32 %idx.3
  store i32 %qadd.3, ptr %addr.3
  %i.next = add nsw nuw i32 %i, -3
  %idx.next = add nsw nuw i32 %idx.1, 3
  %cmp = icmp ult i32 %i.next, %N
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; CHECK-LABEL: test_qadd_4
; CHECK: @ %loop

; TODO: pre-inc store

; CHECK-DEFAULT: ldr{{.*}}, #4]
; CHECK-DEFAULT: ldr{{.*}}, #4]
; CHECK-DEFAULT: str{{.*}}, #4]
; CHECK-DEFAULT: ldr{{.*}}, #8]
; CHECK-DEFAULT: ldr{{.*}}, #8]
; CHECK-DEFAULT: str{{.*}}, #8]
; CHECK-DEFAULT: ldr{{.*}}, #12]
; CHECK-DEFAULT: ldr{{.*}}, #12]
; CHECK-DEFAULT: str{{.*}}, #12]

; CHECK-COMPLEX: ldr{{.*}}, #16]!
; CHECK-COMPLEX: ldr{{.*}}, #16]!
; CHECK-COMPLEX: str{{.*}}, #16]!
; CHECK-COMPLEX: ldr{{.*}}, #4]
; CHECK-COMPLEX: ldr{{.*}}, #4]
; CHECK-COMPLEX: str{{.*}}, #4]
; CHECK-COMPLEX: ldr{{.*}}, #8]
; CHECK-COMPLEX: ldr{{.*}}, #8]
; CHECK-COMPLEX: str{{.*}}, #8]
; CHECK-COMPLEX: ldr{{.*}}, #12]
; CHECK-COMPLEX: ldr{{.*}}, #12]
; CHECK-COMPLEX: str{{.*}}, #12]

; DISABLED-NOT: ldr{{.*}}]!
; DISABLED-NOT: str{{.*}}]!

define void @test_qadd_4(ptr %a.array, ptr %b.array, ptr %out.array, i32 %N) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %idx.1 = phi i32 [ 0, %entry ], [ %idx.next, %loop ]
  %gep.a.1 = getelementptr inbounds i32, ptr %a.array, i32 %idx.1
  %a.1 = load i32, ptr %gep.a.1
  %gep.b.1 = getelementptr inbounds i32, ptr %b.array, i32 %idx.1
  %b.1 = load i32, ptr %gep.b.1
  %qadd.1 = call i32 @llvm.arm.qadd(i32 %a.1, i32 %b.1)
  %addr.1 = getelementptr inbounds i32, ptr %out.array, i32 %idx.1
  store i32 %qadd.1, ptr %addr.1
  %idx.2 = or disjoint i32 %idx.1, 1
  %gep.a.2 = getelementptr inbounds i32, ptr %a.array, i32 %idx.2
  %a.2 = load i32, ptr %gep.a.2
  %gep.b.2 = getelementptr inbounds i32, ptr %b.array, i32 %idx.2
  %b.2 = load i32, ptr %gep.b.2
  %qadd.2 = call i32 @llvm.arm.qadd(i32 %a.2, i32 %b.2)
  %addr.2 = getelementptr inbounds i32, ptr %out.array, i32 %idx.2
  store i32 %qadd.2, ptr %addr.2
  %idx.3 = or disjoint i32 %idx.1, 2
  %gep.a.3 = getelementptr inbounds i32, ptr %a.array, i32 %idx.3
  %a.3 = load i32, ptr %gep.a.3
  %gep.b.3 = getelementptr inbounds i32, ptr %b.array, i32 %idx.3
  %b.3 = load i32, ptr %gep.b.3
  %qadd.3 = call i32 @llvm.arm.qadd(i32 %a.3, i32 %b.3)
  %addr.3 = getelementptr inbounds i32, ptr %out.array, i32 %idx.3
  store i32 %qadd.3, ptr %addr.3
  %idx.4 = or disjoint i32 %idx.1, 3
  %gep.a.4 = getelementptr inbounds i32, ptr %a.array, i32 %idx.4
  %a.4 = load i32, ptr %gep.a.4
  %gep.b.4 = getelementptr inbounds i32, ptr %b.array, i32 %idx.4
  %b.4 = load i32, ptr %gep.b.4
  %qadd.4 = call i32 @llvm.arm.qadd(i32 %a.4, i32 %b.4)
  %addr.4 = getelementptr inbounds i32, ptr %out.array, i32 %idx.4
  store i32 %qadd.4, ptr %addr.4
  %i.next = add nsw nuw i32 %i, -4
  %idx.next = add nsw nuw i32 %idx.1, 4
  %cmp = icmp ult i32 %i.next, %N
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; CHECK-LABEL: test_qadd16_2
; CHECK: @ %loop
; TODO: pre-inc store.

; CHECK-DEFAULT: ldr{{.*}}, #4]
; CHECK-DEFAULT: ldr{{.*}}, #4]
; CHECK-DEFAULT: str{{.*}}, #8]
; CHECK-DEFAULT: ldr{{.*}}, #8]!
; CHECK-DEFAULT: ldr{{.*}}, #8]!
; CHECK-DEFAULT: str{{.*}}, #16]!

; CHECK-COMPLEX: ldr{{.*}}, #8]!
; CHECK-COMPLEX: ldr{{.*}}, #8]!
; CHECK-COMPLEX: str{{.*}}, #16]!
; CHECK-COMPLEX: ldr{{.*}}, #4]
; CHECK-COMPLEX: ldr{{.*}}, #4]
; CHECK-COMPLEX: str{{.*}}, #8]

; DISABLED-NOT: ldr{{.*}}]!
; DISABLED-NOT: str{{.*}}]!

define void @test_qadd16_2(ptr %a.array, ptr %b.array, ptr %out.array, i32 %N) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %idx.1 = phi i32 [ 0, %entry ], [ %idx.next, %loop ]
  %gep.a.1 = getelementptr inbounds i16, ptr %a.array, i32 %idx.1
  %a.1 = load i32, ptr %gep.a.1
  %gep.b.1 = getelementptr inbounds i16, ptr %b.array, i32 %idx.1
  %b.1 = load i32, ptr %gep.b.1
  %qadd.1 = call i32 @llvm.arm.qadd16(i32 %a.1, i32 %b.1)
  %addr.1 = getelementptr inbounds i32, ptr %out.array, i32 %idx.1
  store i32 %qadd.1, ptr %addr.1
  %idx.2 = add nsw nuw i32 %idx.1, 2
  %gep.a.2 = getelementptr inbounds i16, ptr %a.array, i32 %idx.2
  %a.2 = load i32, ptr %gep.a.2
  %gep.b.2 = getelementptr inbounds i16, ptr %b.array, i32 %idx.2
  %b.2 = load i32, ptr %gep.b.2
  %qadd.2 = call i32 @llvm.arm.qadd16(i32 %a.2, i32 %b.2)
  %addr.2 = getelementptr inbounds i32, ptr %out.array, i32 %idx.2
  store i32 %qadd.2, ptr %addr.2
  %i.next = add nsw nuw i32 %i, -2
  %idx.next = add nsw nuw i32 %idx.1, 4
  %cmp = icmp ult i32 %i.next, %N
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

declare i32 @llvm.arm.qadd(i32, i32)
declare i32 @llvm.arm.qadd16(i32, i32)
