; RUN: llc -mtriple=hexagon -mcpu=hexagonv5 -hexagon-eif=0 -stop-after=if-converter < %s | FileCheck %s
; Check that the edge weights are updated correctly after if-conversion.

; CHECK: bb.3.if{{[0-9a-zA-Z.]*}}:
; CHECK: successors: %bb.2(0x0ccccccd), %bb.1(0x73333333)
@a = external global i32
@d = external global i32

; In the following CFG, A,B,C,D will be if-converted into a single block.
; Check if the edge weights on edges to E and F are maintained correctly.
;
;    A
;   / \
;  B   C
;   \ /
;    D
;   / \
;  E   F
;
define void @test1(i8 zeroext %la, i8 zeroext %lb) {
entry:
  %cmp0 = call i1 @pred()
  br i1 %cmp0, label %if.else2, label %if.then0, !prof !1

if.else2:
  call void @bar(i32 2)
  br label %if.end2

if.end2:
  call void @foo(i32 2)
  br label %return

if.end:
  %storemerge = phi i32 [ %and, %if.else ], [ %shl, %if.then ]
  store i32 %storemerge, ptr @a, align 4
  %0 = load i32, ptr @d, align 4
  %cmp2 = call i1 @pred()
  br i1 %cmp2, label %if.end2, label %if.else2, !prof !2

if.then0:
  %cmp = icmp eq i8 %la, %lb
  br i1 %cmp, label %if.then, label %if.else, !prof !1

if.then:
  %conv1 = zext i8 %la to i32
  %shl = shl nuw nsw i32 %conv1, 16
  br label %if.end

if.else:
  %and8 = and i8 %lb, %la
  %and = zext i8 %and8 to i32
  br label %if.end

return:
  call void @foo(i32 2)
  ret void
}

declare void @foo(i32)
declare void @bar(i32)
declare i1 @pred()

!1 = !{!"branch_weights", i32 80, i32 20}
!2 = !{!"branch_weights", i32 10, i32 90}
