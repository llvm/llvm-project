; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

%struct.S = type { i32, [2 x i32], i32 }
%struct.S2 = type { i32, [4 x i32], [4 x i32] }

; CHECK: Function: t1
; CHECK: NoAlias: i32* %gep1, i32* %gep2
define void @t1(%struct.S* %s) {
  %gep1 = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 1, i32 1
  %gep2 = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 1, i32 0
  load i32, i32* %gep1
  load i32, i32* %gep2
  ret void
}

; CHECK: Function: t2_fwd
; CHECK: MayAlias: i32* %gep1, i32* %gep2
define void @t2_fwd(%struct.S* %s, i32* %q) {
  %in_array = load i32, i32* %q, !range !0
  %gep1 = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 1, i32 %in_array
  %gep2 = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 1, i32 0
  load i32, i32* %gep1
  load i32, i32* %gep2
  ret void
}

; CHECK: Function: t2_rev
; CHECK: MayAlias: i32* %gep1, i32* %gep2
define void @t2_rev(%struct.S* %s, i32* %q) {
  %in_array = load i32, i32* %q, !range !0
  %gep1 = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 1, i32 0
  %gep2 = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 1, i32 %in_array
  load i32, i32* %gep1
  load i32, i32* %gep2
  ret void
}

; CHECK: Function: t3_fwd
; CHECK: NoAlias: i32* %gep1, i32* %gep2
define void @t3_fwd(%struct.S* %s, i32* %q) {
  %knownzero = load i32, i32* %q, !range !1
  %gep1 = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 1, i32 %knownzero
  %gep2 = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 1, i32 1
  load i32, i32* %gep1
  load i32, i32* %gep2
  ret void
}

; CHECK: Function: t3_rev
; CHECK: NoAlias: i32* %gep1, i32* %gep2
define void @t3_rev(%struct.S* %s, i32* %q) {
  %knownzero = load i32, i32* %q, !range !1
  %gep1 = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 1, i32 1
  %gep2 = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 1, i32 %knownzero
  load i32, i32* %gep1
  load i32, i32* %gep2
  ret void
}

; CHECK: Function: member_after
; CHECK: NoAlias: i32* %gep1, i32* %gep2
define void @member_after(%struct.S* %s, i32* %q) {
  %in_array = load i32, i32* %q, !range !0
  %gep1 = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 1, i32 %in_array
  %gep2 = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 2
  load i32, i32* %gep1
  load i32, i32* %gep2
  ret void
}

; CHECK: Function: member_after_rev
; CHECK: NoAlias: i32* %gep1, i32* %gep2
define void @member_after_rev(%struct.S* %s, i32* %q) {
  %in_array = load i32, i32* %q, !range !0
  %gep2 = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 2
  %gep1 = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 1, i32 %in_array
  load i32, i32* %gep1
  load i32, i32* %gep2
  ret void
}

; CHECK: Function: member_before
; CHECK: NoAlias: i32* %gep1, i32* %gep2
define void @member_before(%struct.S* %s, i32* %q) {
  %in_array = load i32, i32* %q, !range !0
  %gep1 = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 1, i32 %in_array
  %gep2 = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 0
  load i32, i32* %gep1
  load i32, i32* %gep2
  ret void
}

; CHECK: Function: member_before_rev
; CHECK: NoAlias: i32* %gep1, i32* %gep2
define void @member_before_rev(%struct.S* %s, i32* %q) {
  %in_array = load i32, i32* %q, !range !0
  %gep2 = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 0
  %gep1 = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 1, i32 %in_array
  load i32, i32* %gep1
  load i32, i32* %gep2
  ret void
}

; CHECK-LABEL: Function: t5
; CHECK: MayAlias: i32* %gep1, %struct.S2* %s
; CHECK: PartialAlias (off -4): i32* %gep2, %struct.S2* %s
; CHECK: NoAlias: i32* %gep1, i32* %gep2
define void @t5(%struct.S2* %s, i32* %q) {
  %in_array = load i32, i32* %q, !range !3
  %gep1 = getelementptr inbounds %struct.S2, %struct.S2* %s, i64 0, i32 2, i32 %in_array
  %gep2 = getelementptr inbounds %struct.S2, %struct.S2* %s, i64 0, i32 1, i32 0
  load %struct.S2, %struct.S2* %s
  load i32, i32* %gep1
  load i32, i32* %gep2
  ret void
}

; CHECK-LABEL: Function: t6
; CHECK: MayAlias: i32* %gep1, %struct.S2* %s
; CHECK: PartialAlias (off -16): i32* %gep2, %struct.S2* %s
; CHECK: MayAlias: i32* %gep1, i32* %gep2
define void @t6(%struct.S2* %s, i32* %q) {
  %in_array = load i32, i32* %q, !range !3
  %gep1 = getelementptr inbounds %struct.S2, %struct.S2* %s, i64 0, i32 2, i32 %in_array
  %gep2 = getelementptr inbounds %struct.S2, %struct.S2* %s, i64 0, i32 1, i32 3
  load %struct.S2, %struct.S2* %s
  load i32, i32* %gep1
  load i32, i32* %gep2
  ret void
}

; CHECK-LABEL: Function: t7
; CHECK: MayAlias: i32* %gep1, %struct.S2* %s
; CHECK: PartialAlias (off -20): i32* %gep2, %struct.S2* %s
; CHECK: NoAlias: i32* %gep1, i32* %gep2
define void @t7(%struct.S2* %s, i32* %q) {
  %in_array = load i32, i32* %q, !range !4
  %gep1 = getelementptr inbounds %struct.S2, %struct.S2* %s, i64 0, i32 2, i32 %in_array
  %gep2 = getelementptr inbounds %struct.S2, %struct.S2* %s, i64 0, i32 2, i32 0
  load %struct.S2, %struct.S2* %s
  load i32, i32* %gep1
  load i32, i32* %gep2
  ret void
}

; CHECK-LABEL: Function: t8
; CHECK: MayAlias: i32* %gep1, %struct.S2* %s
; CHECK: PartialAlias (off -24): i32* %gep2, %struct.S2* %s
; CHECK: MayAlias: i32* %gep1, i32* %gep2
define void @t8(%struct.S2* %s, i32* %q) {
  %in_array = load i32, i32* %q, !range !4
  %gep1 = getelementptr inbounds %struct.S2, %struct.S2* %s, i64 0, i32 2, i32 %in_array
  %gep2 = getelementptr inbounds %struct.S2, %struct.S2* %s, i64 0, i32 2, i32 1
  load %struct.S2, %struct.S2* %s
  load i32, i32* %q
  load i32, i32* %gep1
  load i32, i32* %gep2
  ret void
}

; CHECK-LABEL: Function: t9
; CHECK: MayAlias: i32* %gep1, %struct.S2* %s
; CHECK: PartialAlias (off -20): i32* %gep2, %struct.S2* %s
; CHECK: NoAlias: i32* %gep1, i32* %gep2
define void @t9(%struct.S2* %s, i32* %q) {
  %in_array = load i32, i32* %q, !range !5
  %gep1 = getelementptr inbounds %struct.S2, %struct.S2* %s, i64 0, i32 1, i32 %in_array
  %gep2 = getelementptr inbounds %struct.S2, %struct.S2* %s, i64 0, i32 2, i32 0
  load %struct.S2, %struct.S2* %s
  load i32, i32* %gep1
  load i32, i32* %gep2
  ret void
}

; CHECK-LABEL: Function: t10
; CHECK: MayAlias: i32* %gep1, %struct.S2* %s
; CHECK: PartialAlias (off -4): i32* %gep2, %struct.S2* %s
; CHECK: MayAlias: i32* %gep1, i32* %gep2
define void @t10(%struct.S2* %s, i32* %q) {
  %in_array = load i32, i32* %q, !range !5
  %gep1 = getelementptr inbounds %struct.S2, %struct.S2* %s, i64 0, i32 2, i32 %in_array
  %gep2 = getelementptr inbounds %struct.S2, %struct.S2* %s, i64 0, i32 1, i32 0
  load %struct.S2, %struct.S2* %s
  load i32, i32* %gep1
  load i32, i32* %gep2
  ret void
}

; CHECK-LABEL: Function: zeroext_index
; CHECK:  MayAlias:     i32* %gep, [256 x i32]* %s
define void @zeroext_index([256 x i32]* %s, i8* %q) {
  %a = load i8, i8* %q, !range !6
  %in_array = zext i8 %a to i32
  %gep = getelementptr inbounds [256 x i32], [256 x i32]* %s, i64 0, i32 %in_array
  load [256 x i32], [256 x i32]* %s
  load i32, i32* %gep
  ret void
}

; CHECK-LABEL: Function: multiple
; CHECK: MayAlias: i32* %p, i32* %p.01
; CHECK: MayAlias: i32* %p, i32* %p.02
; CHECK: MayAlias: i32* %p.01, i32* %p.02
; CHECK: NoAlias:  i32* %p.01, i32* %p.2
; CHECK: MayAlias: i32* %p.02, i32* %p.2
; CHECK: NoAlias:  i32* %p.01, i32* %p.3
; CHECK: NoAlias:  i32* %p.02, i32* %p.3
define void @multiple(i32* %p, i32* %o1_ptr, i32* %o2_ptr) {
  %o1 = load i32, i32* %o1_ptr, !range !0
  %o2 = load i32, i32* %o2_ptr, !range !0
  %p.01 = getelementptr i32, i32* %p, i32 %o1  ; p + [0, 1]
  %p.02 = getelementptr i32, i32* %p.01, i32 %o2 ; p + [0, 2]
  %p.2 = getelementptr i32, i32* %p, i32 2
  %p.3 = getelementptr i32, i32* %p, i32 3
  load i32, i32* %p
  load i32, i32* %p.01
  load i32, i32* %p.02
  load i32, i32* %p.2
  load i32, i32* %p.3
  ret void
}

; p.neg1 and p.o.1 don't alias, even though the addition o+1 may overflow.
; While it makes INT_MIN a possible offset, offset -1 is not possible.
; CHECK-LABEL: Function: benign_overflow
; CHECK: MayAlias: i8* %p, i8* %p.o
; CHECK: MayAlias: i8* %p.neg1, i8* %p.o
; CHECK: MayAlias: i8* %p, i8* %p.o.1
; CHECK: NoAlias: i8* %p.neg1, i8* %p.o.1
; CHECK: NoAlias:  i8* %p.o, i8* %p.o.1
define void @benign_overflow(i8* %p, i64 %o) {
  %c = icmp sge i64 %o, -1
  call void @llvm.assume(i1 %c)
  %p.neg1 = getelementptr i8, i8* %p, i64 -1
  %p.o = getelementptr i8, i8* %p, i64 %o
  %p.o.1 = getelementptr i8, i8* %p.o, i64 1
  load i8, i8* %p
  load i8, i8* %p.neg1
  load i8, i8* %p.o
  load i8, i8* %p.o.1
  ret void
}

declare void @llvm.assume(i1)


!0 = !{ i32 0, i32 2 }
!1 = !{ i32 0, i32 1 }
!2 = !{ i32 1, i32 2 }
!3 = !{ i32 -2, i32 0 }
!4 = !{ i32 1, i32 536870911 }
!5 = !{ i32 -536870911, i32 4 }
!6 = !{ i8 -2, i8 0 }
