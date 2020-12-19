; RUN: opt -S -simplifycfg < %s | FileCheck %s

; SimplifyCFG should eliminate redundant indirectbr edges.

; CHECK: indbrtest0
; CHECK: indirectbr i8* %t, [label %BB0, label %BB1, label %BB2]
; CHECK: %x = phi i32 [ 0, %BB0 ], [ 1, %entry ]

declare void @foo()
declare void @A()
declare void @B(i32)
declare void @C()

define void @indbrtest0(i8** %P, i8** %Q) {
entry:
  store i8* blockaddress(@indbrtest0, %BB0), i8** %P
  store i8* blockaddress(@indbrtest0, %BB1), i8** %P
  store i8* blockaddress(@indbrtest0, %BB2), i8** %P
  call void @foo()
  %t = load i8*, i8** %Q
  indirectbr i8* %t, [label %BB0, label %BB1, label %BB2, label %BB0, label %BB1, label %BB2]
BB0:
  call void @A()
  br label %BB1
BB1:
  %x = phi i32 [ 0, %BB0 ], [ 1, %entry ], [ 1, %entry ]
  call void @B(i32 %x)
  ret void
BB2:
  call void @C()
  ret void
}

; SimplifyCFG should convert the indirectbr into a directbr. It would be even
; better if it removed the branch altogether, but simplifycfdg currently misses
; that because the predecessor is the entry block.

; CHECK: indbrtest1
; CHECK: br label %BB0

define void @indbrtest1(i8** %P, i8** %Q) {
entry:
  store i8* blockaddress(@indbrtest1, %BB0), i8** %P
  call void @foo()
  %t = load i8*, i8** %Q
  indirectbr i8* %t, [label %BB0, label %BB0]
BB0:
  call void @A()
  ret void
}

; SimplifyCFG should notice that BB0 does not have its address taken and
; remove it from entry's successor list.

; CHECK: indbrtest2
; CHECK: entry:
; CHECK-NEXT: unreachable

define void @indbrtest2(i8* %t) {
entry:
  indirectbr i8* %t, [label %BB0, label %BB0]
BB0:
  ret void
}


; Make sure the blocks in the next few tests aren't trivially removable as
; successors by taking their addresses.

@anchor = constant [13 x i8*] [
  i8* blockaddress(@indbrtest3, %L1), i8* blockaddress(@indbrtest3, %L2), i8* blockaddress(@indbrtest3, %L3),
  i8* blockaddress(@indbrtest4, %L1), i8* blockaddress(@indbrtest4, %L2), i8* blockaddress(@indbrtest4, %L3),
  i8* blockaddress(@indbrtest5, %L1), i8* blockaddress(@indbrtest5, %L2), i8* blockaddress(@indbrtest5, %L3), i8* blockaddress(@indbrtest5, %L4),
  i8* blockaddress(@indbrtest6, %L1), i8* blockaddress(@indbrtest6, %L2), i8* blockaddress(@indbrtest6, %L3)
]

; SimplifyCFG should turn the indirectbr into a conditional branch on the
; condition of the select.

; CHECK-LABEL: @indbrtest3(
; CHECK-NEXT: entry:
; CHECK-NEXT: br i1 %cond, label %L1, label %L2
; CHECK-NOT: indirectbr
; CHECK-NOT: br
; CHECK-NOT: L3:
define void @indbrtest3(i1 %cond, i8* %address) nounwind {
entry:
  %indirect.goto.dest = select i1 %cond, i8* blockaddress(@indbrtest3, %L1), i8* blockaddress(@indbrtest3, %L2)
  indirectbr i8* %indirect.goto.dest, [label %L1, label %L2, label %L3]

L1:
  call void @A()
  ret void
L2:
  call void @C()
  ret void
L3:
  call void @foo()
  ret void
}

; SimplifyCFG should turn the indirectbr into an unconditional branch to the
; only possible destination.
; As in @indbrtest1, it should really remove the branch entirely, but it doesn't
; because it's in the entry block.

; CHECK-LABEL: @indbrtest4(
; CHECK-NEXT: entry:
; CHECK-NEXT: br label %L1
define void @indbrtest4(i1 %cond) nounwind {
entry:
  %indirect.goto.dest = select i1 %cond, i8* blockaddress(@indbrtest4, %L1), i8* blockaddress(@indbrtest4, %L1)
  indirectbr i8* %indirect.goto.dest, [label %L1, label %L2, label %L3]

L1:
  call void @A()
  ret void
L2:
  call void @C()
  ret void
L3:
  call void @foo()
  ret void
}

; SimplifyCFG should turn the indirectbr into an unreachable because neither
; destination is listed as a successor.

; CHECK-LABEL: @indbrtest5(
; CHECK-NEXT: entry:
; CHECK-NEXT: unreachable
; CHECK-NEXT: }
define void @indbrtest5(i1 %cond, i8* %anchor) nounwind {
entry:
  %indirect.goto.dest = select i1 %cond, i8* blockaddress(@indbrtest5, %L1), i8* blockaddress(@indbrtest5, %L2)
; This needs to have more than one successor for this test, otherwise it gets
; replaced with an unconditional branch to the single successor.
  indirectbr i8* %indirect.goto.dest, [label %L3, label %L4]

L1:
  call void @A()
  ret void
L2:
  call void @C()
  ret void
L3:
  call void @foo()
  ret void
L4:
  call void @foo()

; This keeps blockaddresses not otherwise listed as successors from being zapped
; before SimplifyCFG even looks at the indirectbr.
  indirectbr i8* %anchor, [label %L1, label %L2]
}

; The same as above, except the selected addresses are equal.

; CHECK-LABEL: @indbrtest6(
; CHECK-NEXT: entry:
; CHECK-NEXT: unreachable
; CHECK-NEXT: }
define void @indbrtest6(i1 %cond, i8* %anchor) nounwind {
entry:
  %indirect.goto.dest = select i1 %cond, i8* blockaddress(@indbrtest6, %L1), i8* blockaddress(@indbrtest6, %L1)
; This needs to have more than one successor for this test, otherwise it gets
; replaced with an unconditional branch to the single successor.
  indirectbr i8* %indirect.goto.dest, [label %L2, label %L3]

L1:
  call void @A()
  ret void
L2:
  call void @C()
  ret void
L3:
  call void @foo()

; This keeps blockaddresses not otherwise listed as successors from being zapped
; before SimplifyCFG even looks at the indirectbr.
  indirectbr i8* %anchor, [label %L1, label %L2]
}

; PR10072

@xblkx.bbs = internal unnamed_addr constant [9 x i8*] [i8* blockaddress(@indbrtest7, %xblkx.begin), i8* blockaddress(@indbrtest7, %xblkx.begin3), i8* blockaddress(@indbrtest7, %xblkx.begin4), i8* blockaddress(@indbrtest7, %xblkx.begin5), i8* blockaddress(@indbrtest7, %xblkx.begin6), i8* blockaddress(@indbrtest7, %xblkx.begin7), i8* blockaddress(@indbrtest7, %xblkx.begin8), i8* blockaddress(@indbrtest7, %xblkx.begin9), i8* blockaddress(@indbrtest7, %xblkx.end)]

define void @indbrtest7() {
escape-string.top:
  %xval202x = call i32 @xfunc5x()
  br label %xlab5x

xlab8x:                                           ; preds = %xlab5x
  %xvaluex = call i32 @xselectorx()
  %xblkx.x = getelementptr [9 x i8*], [9 x i8*]* @xblkx.bbs, i32 0, i32 %xvaluex
  %xblkx.load = load i8*, i8** %xblkx.x
  indirectbr i8* %xblkx.load, [label %xblkx.begin, label %xblkx.begin3, label %xblkx.begin4, label %xblkx.begin5, label %xblkx.begin6, label %xblkx.begin7, label %xblkx.begin8, label %xblkx.begin9, label %xblkx.end]

xblkx.begin:
  br label %xblkx.end

xblkx.begin3:
  br label %xblkx.end

xblkx.begin4:
  br label %xblkx.end

xblkx.begin5:
  br label %xblkx.end

xblkx.begin6:
  br label %xblkx.end

xblkx.begin7:
  br label %xblkx.end

xblkx.begin8:
  br label %xblkx.end

xblkx.begin9:
  br label %xblkx.end

xblkx.end:
  %yes.0 = phi i1 [ false, %xblkx.begin ], [ true, %xlab8x ], [ false, %xblkx.begin9 ], [ false, %xblkx.begin8 ], [ false, %xblkx.begin7 ], [ false, %xblkx.begin6 ], [ false, %xblkx.begin5 ], [ true, %xblkx.begin4 ], [ false, %xblkx.begin3 ]
  br i1 %yes.0, label %v2j, label %xlab17x

v2j:
; CHECK: %xunusedx = call i32 @xactionx()
  %xunusedx = call i32 @xactionx()
  br label %xlab4x

xlab17x:
  br label %xlab4x

xlab4x:
  %incr19 = add i32 %xval704x.0, 1
  br label %xlab5x

xlab5x:
  %xval704x.0 = phi i32 [ 0, %escape-string.top ], [ %incr19, %xlab4x ]
  %xval10x = icmp ult i32 %xval704x.0, %xval202x
  br i1 %xval10x, label %xlab8x, label %xlab9x

xlab9x:
  ret void
}

declare i32 @xfunc5x()
declare i8 @xfunc7x()
declare i32 @xselectorx()
declare i32 @xactionx()
