; RUN: opt < %s -internalize -ipsccp -S | FileCheck %s
; PR5569

; IPSCCP should prove that the blocks are dead and delete them, and
; properly handle the dangling blockaddress constants.

; CHECK: @bar.l = internal constant [2 x i8*] [i8* inttoptr (i32 1 to i8*), i8* inttoptr (i32 1 to i8*)]

@code = global [5 x i32] [i32 0, i32 0, i32 0, i32 0, i32 1], align 4 ; <[5 x i32]*> [#uses=0]
@bar.l = internal constant [2 x i8*] [i8* blockaddress(@bar, %lab0), i8* blockaddress(@bar, %end)] ; <[2 x i8*]*> [#uses=1]

define void @foo(i32 %x) nounwind readnone {
entry:
  %b = alloca i32, align 4                        ; <i32*> [#uses=1]
  store volatile i32 -1, i32* %b
  ret void
}

define void @bar(i32* nocapture %pc) nounwind readonly {
entry:
  br label %indirectgoto

lab0:                                             ; preds = %indirectgoto
  %indvar.next = add i32 %indvar, 1               ; <i32> [#uses=1]
  br label %indirectgoto

end:                                              ; preds = %indirectgoto
  ret void

indirectgoto:                                     ; preds = %lab0, %entry
  %indvar = phi i32 [ %indvar.next, %lab0 ], [ 0, %entry ] ; <i32> [#uses=2]
  %pc.addr.0 = getelementptr i32* %pc, i32 %indvar ; <i32*> [#uses=1]
  %tmp1.pn = load i32* %pc.addr.0                 ; <i32> [#uses=1]
  %indirect.goto.dest.in = getelementptr inbounds [2 x i8*]* @bar.l, i32 0, i32 %tmp1.pn ; <i8**> [#uses=1]
  %indirect.goto.dest = load i8** %indirect.goto.dest.in ; <i8*> [#uses=1]
  indirectbr i8* %indirect.goto.dest, [label %lab0, label %end]
}

define i32 @main() nounwind readnone {
entry:
  ret i32 0
}
