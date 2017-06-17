; RUN: opt < %s -S -instcombine | FileCheck %s
@A = extern_weak global i32, align 4
@B = extern_weak global i32, align 4

define i32 @foo(i1 %which) {
entry:
  br i1 %which, label %final, label %delay

delay:
  br label %final

; CHECK-LABEL: @foo
; CHECK-LABEL: final:
; CHECK: phi i32 [ 1, %entry ], [ select (i1 icmp eq (i32* @A, i32* @B), i32 2, i32 1), %delay ]
final:
  %use2 = phi i1 [ false, %entry ], [ icmp eq (i32* @A, i32* @B), %delay ]
  %value = select i1 %use2, i32 2, i32 1
  ret i32 %value
}


; test folding of select into phi for vectors.
define <4 x i64> @vec1(i1 %which) {
entry:
  br i1 %which, label %final, label %delay

delay:
 br label %final

final:
; CHECK-LABEL: @vec1
; CHECK-LABEL: final:
; CHECK: %phinode = phi <4 x i64> [ zeroinitializer, %entry ], [ <i64 0, i64 0, i64 126, i64 127>, %delay ]
; CHECK-NOT: select
; CHECK: ret <4 x i64> %phinode
 %phinode =  phi <4 x i1> [ <i1 true, i1 true, i1 true, i1 true>, %entry ], [ <i1 true, i1 true, i1 false, i1 false>, %delay ]
 %sel = select <4 x i1> %phinode, <4 x i64> zeroinitializer, <4 x i64> <i64 124, i64 125, i64 126, i64 127>
 ret <4 x i64> %sel
}

define <4 x i64> @vec2(i1 %which) {
entry:
  br i1 %which, label %final, label %delay

delay:
 br label %final

final:
; CHECK-LABEL: @vec2
; CHECK-LABEL: final:
; CHECK: %phinode = phi <4 x i64> [ <i64 124, i64 125, i64 126, i64 127>, %entry ], [ <i64 0, i64 125, i64 0, i64 127>, %delay ]
; CHECK-NOT: select
; CHECK: ret <4 x i64> %phinode
 %phinode =  phi <4 x i1> [ <i1 false, i1 false, i1 false, i1 false>, %entry ], [ <i1 true, i1 false, i1 true, i1 false>, %delay ]
 %sel = select <4 x i1> %phinode, <4 x i64> zeroinitializer, <4 x i64> <i64 124, i64 125, i64 126, i64 127>
 ret <4 x i64> %sel
}

; Test PR33364
; Insert the generated select into the same block as the incoming phi value.
; phi has constant vectors along with a single non-constant vector as operands.
define <2 x i8> @vec3(i1 %cond1, i1 %cond2, <2 x i1> %x, <2 x i8> %y, <2 x i8> %z) {
; CHECK-LABEL: @vec3
; CHECK-LABEL: entry:
; CHECK-NEXT: [[PHITMP1:%.*]] = shufflevector <2 x i8> %y, <2 x i8> %z, <2 x i32> <i32 2, i32 1>
entry:
  br i1 %cond1, label %if1, label %else

; CHECK-LABEL: if1:
; CHECK-NEXT: [[PHITMP2:%.*]] = shufflevector <2 x i8> %y, <2 x i8> %z, <2 x i32> <i32 0, i32 3>
if1:
  br i1 %cond2, label %if2, label %else

; CHECK-LABEL: if2:
; CHECK-NEXT: [[PHITMP3:%.*]] = select <2 x i1> %x, <2 x i8> %y, <2 x i8> %z
if2:
  br label %else

; CHECK-LABEL: else:
; CHECK-NEXT: [[PHITMP4:%.*]] = phi <2 x i8> [ [[PHITMP3]], %if2 ], [ [[PHITMP1]], %entry ], [ [[PHITMP2]], %if1 ]
; CHECK-NEXT: ret <2 x i8> [[PHITMP4]]
else:
  %phi = phi <2 x i1> [ %x, %if2 ], [ <i1 0, i1 1>, %entry ], [ <i1 1, i1 0>, %if1 ]
  %sel = select <2 x i1> %phi, <2 x i8> %y, <2 x i8> %z
  ret <2 x i8> %sel
}
