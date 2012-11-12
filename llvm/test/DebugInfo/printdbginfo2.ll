; RUN: opt < %s -print-dbginfo -disable-output 2>&1 | FileCheck %s
;  grep {%b is variable b of type x declared at x.c:7} %t1
;  grep {%2 is variable b of type x declared at x.c:7} %t1
;  grep {@c.1442 is variable c of type int declared at x.c:4} %t1

%struct.foo = type { i32 }

@main.c = internal global i32 5                   ; <i32*> [#uses=1]

define i32 @main() nounwind {
entry:
  %retval = alloca i32                            ; <i32*> [#uses=3]
  %b = alloca %struct.foo, align 4                ; <%struct.foo*> [#uses=2]
; CHECK:; %b is variable b of type foo declared at x.c:7
  %a = alloca [4 x i32], align 4                  ; <[4 x i32]*> [#uses=1]
; CHECK:; %a is variable a of type  declared at x.c:8
  call void @llvm.dbg.func.start(metadata !3)
  store i32 0, i32* %retval
  call void @llvm.dbg.stoppoint(i32 6, i32 3, metadata !1)
  call void @llvm.dbg.stoppoint(i32 7, i32 3, metadata !1)
  %0 = bitcast %struct.foo* %b to { }*            ; <{ }*> [#uses=1]
  call void @llvm.dbg.declare(metadata !{%struct.foo* %b}, metadata !4)
; CHECK:; %0 is variable b of type foo declared at x.c:7
  call void @llvm.dbg.stoppoint(i32 8, i32 3, metadata !1)
  %1 = bitcast [4 x i32]* %a to { }*              ; <{ }*> [#uses=1]
  call void @llvm.dbg.declare(metadata !{[4 x i32]* %a}, metadata !8)
; CHECK:; %1 is variable a of type  declared at x.c:8
  call void @llvm.dbg.stoppoint(i32 9, i32 3, metadata !1)
  %tmp = getelementptr inbounds %struct.foo* %b, i32 0, i32 0 ; <i32*> [#uses=1]
; CHECK:; %tmp is variable b of type foo declared at x.c:7
  store i32 5, i32* %tmp
  call void @llvm.dbg.stoppoint(i32 10, i32 3, metadata !1)
  %tmp1 = load i32* @main.c                       ; <i32> [#uses=1]
; CHECK:; @main.c is variable c of type int declared at x.c:6
  store i32 %tmp1, i32* %retval
  br label %2

; <label>:2                                       ; preds = %entry
  call void @llvm.dbg.stoppoint(i32 11, i32 1, metadata !1)
  call void @llvm.dbg.region.end(metadata !3)
  %3 = load i32* %retval                          ; <i32> [#uses=1]
  ret i32 %3
}

declare void @llvm.dbg.func.start(metadata) nounwind readnone

declare void @llvm.dbg.stoppoint(i32, i32, metadata) nounwind readnone

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare void @llvm.dbg.region.end(metadata) nounwind readnone

!llvm.dbg.gv = !{!0}

!0 = metadata !{i32 458804, i32 0, metadata !1, metadata !"c", metadata !"c", metadata !"", metadata !1, i32 6, metadata !2, i1 true, i1 true, i32* @main.c}
!1 = metadata !{i32 458769, i32 0, i32 12, metadata !"x.c", metadata !"/home/edwin/llvm-git/llvm/test/DebugInfo", metadata !"clang 1.0", i1 true, i1 false, metadata !"", i32 0}
!2 = metadata !{i32 458788, metadata !1, metadata !"int", metadata !1, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5}
!3 = metadata !{i32 458798, i32 0, metadata !1, metadata !"main", metadata !"main", metadata !"main", metadata !1, i32 5, metadata !2, i1 false, i1 true}
!4 = metadata !{i32 459008, metadata !3, metadata !"b", metadata !1, i32 7, metadata !5}
!5 = metadata !{i32 458771, metadata !1, metadata !"foo", metadata !1, i32 1, i64 32, i64 32, i64 0, i32 0, null, metadata !6, i32 0}
!6 = metadata !{metadata !7}
!7 = metadata !{i32 458765, metadata !1, metadata !"a", metadata !1, i32 2, i64 32, i64 32, i64 0, i32 0, metadata !2}
!8 = metadata !{i32 459008, metadata !3, metadata !"a", metadata !1, i32 8, metadata !9}
!9 = metadata !{i32 458753, metadata !1, metadata !"", null, i32 0, i64 128, i64 32, i64 0, i32 0, metadata !2, metadata !10, i32 0}
!10 = metadata !{metadata !11}
!11 = metadata !{i32 458785, i64 0, i64 3}
