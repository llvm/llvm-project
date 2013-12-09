; RUN: llvm-as < %s | llvm-dis | FileCheck %s

define void @Foo(i32 %a, i32 %b) {
entry:
  call void @llvm.dbg.value(metadata !{ i32* %1 }, i64 16, metadata !2)
; CHECK: call void @llvm.dbg.value(metadata !{i32* %1}, i64 16, metadata ![[ID2:[0-9]+]])
  %0 = add i32 %a, 1                              ; <i32> [#uses=1]
  %two = add i32 %b, %0                           ; <i32> [#uses=0]
  %1 = alloca i32                                 ; <i32*> [#uses=1]

  call void @llvm.dbg.declare(metadata !{i32* %1}, metadata !{i32* %1})
; CHECK: metadata !{i32* %1}, metadata !{i32* %1}
  call void @llvm.dbg.declare(metadata !{i32 %two}, metadata !{i32 %0})
; CHECK: metadata !{i32 %two}, metadata !{i32 %0}
  call void @llvm.dbg.declare(metadata !{i32 %0}, metadata !{i32* %1, i32 %0})
; CHECK: metadata !{i32 %0}, metadata !{i32* %1, i32 %0}
  call void @llvm.dbg.declare(metadata !{i32* %1}, metadata !{i32 %b, i32 %0})
; CHECK: metadata !{i32* %1}, metadata !{i32 %b, i32 %0}
  call void @llvm.dbg.declare(metadata !{i32 %a}, metadata !{i32 %a, metadata !"foo"})
; CHECK: metadata !{i32 %a}, metadata !{i32 %a, metadata !"foo"}
  call void @llvm.dbg.declare(metadata !{i32 %b}, metadata !{metadata !0, i32 %two})
; CHECK: metadata !{i32 %b}, metadata !{metadata ![[ID0:[0-9]+]], i32 %two}

  call void @llvm.dbg.value(metadata !{ i32 %a }, i64 0, metadata !1)
; CHECK: metadata !{i32 %a}, i64 0, metadata ![[ID1:[0-9]+]]
  call void @llvm.dbg.value(metadata !{ i32 %0 }, i64 25, metadata !0)
; CHECK: metadata !{i32 %0}, i64 25, metadata ![[ID0]]
  call void @llvm.dbg.value(metadata !{ i32* %1 }, i64 16, metadata !3)
; CHECK: call void @llvm.dbg.value(metadata !{i32* %1}, i64 16, metadata ![[ID3:[0-9]+]])
  call void @llvm.dbg.value(metadata !3, i64 12, metadata !2)
; CHECK: metadata ![[ID3]], i64 12, metadata ![[ID2]]

  ret void, !foo !0, !bar !1
; CHECK: ret void, !foo ![[FOO:[0-9]+]], !bar ![[BAR:[0-9]+]]
}

!llvm.module.flags = !{!4}

!0 = metadata !{i32 662302, i32 26, metadata !1, null}
!1 = metadata !{i32 4, metadata !"foo"}
!2 = metadata !{metadata !"bar"}
!3 = metadata !{metadata !"foo"}
!4 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!foo = !{ !0 }
!bar = !{ !1 }

; CHECK: !foo = !{![[FOO]]}
; CHECK: !bar = !{![[BAR]]}
; CHECK: ![[ID0]] = metadata !{i32 662302, i32 26, metadata ![[ID1]], null}
; CHECK: ![[ID1]] = metadata !{i32 4, metadata !"foo"}
; CHECK: ![[ID2]] = metadata !{metadata !"bar"}
; CHECK; ![[ID3]] = metadata !{metadata !"foo"}
