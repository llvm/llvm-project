; RUN: llc < %s -mtriple=i686-linux -show-mc-encoding | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-linux -show-mc-encoding | FileCheck %s

declare void @foo()
declare void @bar()

define void @f(i32 %x, i32 %y) optsize {
entry:
	%p = icmp eq i32 %x, %y
  br i1 %p, label %bb1, label %bb2
bb1:
  tail call void @foo()
  ret void
bb2:
  tail call void @bar()
  ret void

; CHECK-LABEL: f:
; CHECK: cmp
; CHECK: jne bar
; Check that the asm doesn't just look good, but uses the correct encoding.
; CHECK: encoding: [0x75,A]
; CHECK: jmp foo
}


declare x86_thiscallcc zeroext i1 @baz(i8*, i32)
define x86_thiscallcc zeroext i1 @BlockPlacementTest(i8* %this, i32 %x) optsize {
entry:
  %and = and i32 %x, 42
  %tobool = icmp eq i32 %and, 0
  br i1 %tobool, label %land.end, label %land.rhs

land.rhs:
  %and6 = and i32 %x, 44
  %tobool7 = icmp eq i32 %and6, 0
  br i1 %tobool7, label %lor.rhs, label %land.end

lor.rhs:
  %call = tail call x86_thiscallcc zeroext i1 @baz(i8* %this, i32 %x) #2
  br label %land.end

land.end:
  %0 = phi i1 [ false, %entry ], [ true, %land.rhs ], [ %call, %lor.rhs ]
  ret i1 %0

; Make sure machine block placement isn't confused by the conditional tail call,
; but sees that it can fall through to the next block.
; CHECK-LABEL: BlockPlacementTest
; CHECK: je baz
; CHECK-NOT: xor
; CHECK: ret
}
