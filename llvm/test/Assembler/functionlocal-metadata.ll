; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

define void @Foo(i32 %a, i32 %b) {
entry:
  call void @llvm.dbg.value(metadata i32* %1, i64 16, metadata !2, metadata !{!"0x102"})
; CHECK: call void @llvm.dbg.value(metadata i32* %1, i64 16, metadata ![[ID2:[0-9]+]], metadata {{.*}})
  %0 = add i32 %a, 1                              ; <i32> [#uses=1]
  %two = add i32 %b, %0                           ; <i32> [#uses=0]
  %1 = alloca i32                                 ; <i32*> [#uses=1]

  call void @llvm.dbg.declare(metadata i32* %1, metadata i32* %1, metadata !{!"0x102"})
; CHECK: call void @llvm.dbg.declare(metadata i32* %1, metadata i32* %1, metadata {{.*}})
  call void @llvm.dbg.declare(metadata i32 %two, metadata i32 %0, metadata !{!"0x102"})
; CHECK: call void @llvm.dbg.declare(metadata i32 %two, metadata i32 %0, metadata {{.*}})
  call void @llvm.dbg.declare(metadata i32* %1, metadata i32 %b, metadata !{!"0x102"})
; CHECK: call void @llvm.dbg.declare(metadata i32* %1, metadata i32 %b, metadata {{.*}})
  call void @llvm.dbg.declare(metadata i32 %a, metadata i32 %a, metadata !{!"0x102"})
; CHECK: call void @llvm.dbg.declare(metadata i32 %a, metadata i32 %a, metadata {{.*}})
  call void @llvm.dbg.declare(metadata i32 %b, metadata i32 %two, metadata !{!"0x102"})
; CHECK: call void @llvm.dbg.declare(metadata i32 %b, metadata i32 %two, metadata {{.*}})

  call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata !1, metadata !{!"0x102"})
; CHECK: call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata ![[ID1:[0-9]+]], metadata {{.*}})
  call void @llvm.dbg.value(metadata i32 %0, i64 25, metadata !0, metadata !{!"0x102"})
; CHECK: call void @llvm.dbg.value(metadata i32 %0, i64 25, metadata ![[ID0:[0-9]+]], metadata {{.*}})
  call void @llvm.dbg.value(metadata i32* %1, i64 16, metadata !3, metadata !{!"0x102"})
; CHECK: call void @llvm.dbg.value(metadata i32* %1, i64 16, metadata ![[ID3:[0-9]+]], metadata {{.*}})
  call void @llvm.dbg.value(metadata !3, i64 12, metadata !2, metadata !{!"0x102"})
; CHECK: call void @llvm.dbg.value(metadata ![[ID3]], i64 12, metadata ![[ID2]], metadata {{.*}})

  ret void, !foo !0, !bar !1
; CHECK: ret void, !foo ![[FOO:[0-9]+]], !bar ![[BAR:[0-9]+]]
}

!llvm.module.flags = !{!4}

!0 = !MDLocation(line: 662302, column: 26, scope: !1)
!1 = !{i32 4, !"foo"}
!2 = !{!"bar"}
!3 = !{!"foo"}
!4 = !{i32 1, !"Debug Info Version", i32 2}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!foo = !{ !0 }
!bar = !{ !1 }

; CHECK: !foo = !{![[FOO]]}
; CHECK: !bar = !{![[BAR]]}
; CHECK: ![[ID0]] = !MDLocation(line: 662302, column: 26, scope: ![[ID1]])
; CHECK: ![[ID1]] = !{i32 4, !"foo"}
; CHECK: ![[ID2]] = !{!"bar"}
; CHECK: ![[ID3]] = !{!"foo"}
