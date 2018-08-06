; RUN: llc < %s -mtriple=armv4t--linux-androideabi -print-machineinstrs=if-converter -o /dev/null 2>&1 | FileCheck %s
; Fix a bug triggered in IfConverterTriangle when CvtBB has multiple
; predecessors.
; PR18752

%classK = type { i8, %classF }
%classF = type { i8 }
%classL = type { %classG, i32, i32 }
%classG = type { %classL* }
%classM2 = type { %classL }

define zeroext i1 @test(%classK* %this, %classM2* nocapture readnone %p1, %classM2* nocapture readnone %p2) align 2 {
entry:
  br i1 undef, label %for.end, label %for.body

; Before if conversion, we have
; for.body -> lor.lhs.false.i (50%)
;          -> for.cond.backedge (50%)
; lor.lhs.false.i -> for.cond.backedge (100%)
;                 -> cond.false.i (0%)
; Afer if conversion, we have
; for.body -> for.cond.backedge (100%)
;          -> cond.false.i (0%)
; CHECK: bb.1.for.body:
; CHECK: successors: %bb.2(0x80000000), %bb.4(0x00000001)
for.body:
  br i1 undef, label %for.cond.backedge, label %lor.lhs.false.i, !prof !1

for.cond.backedge:
  %tobool = icmp eq %classL* undef, null
  br i1 %tobool, label %for.end, label %for.body

lor.lhs.false.i:
  %tobool.i.i7 = icmp eq i32 undef, 0
  br i1 %tobool.i.i7, label %for.cond.backedge, label %cond.false.i

cond.false.i:
  call void @_Z3fn1v()
  unreachable

for.end:
  br i1 undef, label %if.else.i.i, label %if.then.i.i

if.then.i.i:
  store %classL* null, %classL** undef, align 4
  br label %_ZN1M6spliceEv.exit

if.else.i.i:
  store %classL* null, %classL** null, align 4
  br label %_ZN1M6spliceEv.exit

_ZN1M6spliceEv.exit:
  %LIS = getelementptr inbounds %classK, %classK* %this, i32 0, i32 1
  call void @_ZN1F10handleMoveEb(%classF* %LIS, i1 zeroext false)
  unreachable
}

declare %classL* @_ZN1M1JI1LS1_EcvPS1_Ev(%classM2*)
declare void @_ZN1F10handleMoveEb(%classF*, i1 zeroext)
declare void @_Z3fn1v()

!0 = !{!"clang version 3.5"}
!1 = !{!"branch_weights", i32 62, i32 62}
