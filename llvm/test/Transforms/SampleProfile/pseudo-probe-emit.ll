; REQUIRES: x86-registered-target
; RUN: opt < %s -passes=pseudo-probe -function-sections -S -o %t
; RUN: FileCheck %s < %t --check-prefix=CHECK-IL
; RUN: llc %t -stop-after=instruction-select -o - | FileCheck %s --check-prefix=CHECK-MIR
;
;; Check the generation of pseudoprobe intrinsic call.

define void @foo(i32 %x) !dbg !3 {
bb0:
  %cmp = icmp eq i32 %x, 0
; CHECK-IL: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 1, i32 0), !dbg ![[#FAKELINE:]]
; CHECK-MIR: PSEUDO_PROBE [[#GUID:]], 1, 0, 0
  br i1 %cmp, label %bb1, label %bb2

bb1:
; CHECK-IL: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 2, i32 0), !dbg ![[#FAKELINE]]
; CHECK-MIR: PSEUDO_PROBE [[#GUID]], 3, 0, 0
; CHECK-MIR: PSEUDO_PROBE [[#GUID]], 4, 0, 0
  br label %bb3

bb2:
; CHECK-IL: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 3, i32 0), !dbg ![[#FAKELINE]]
; CHECK-MIR: PSEUDO_PROBE [[#GUID]], 2, 0, 0
; CHECK-MIR: PSEUDO_PROBE [[#GUID]], 4, 0, 0
  br label %bb3

bb3:
; CHECK-IL: call void @llvm.pseudoprobe(i64 [[#GUID]], i64 4, i32 0), !dbg ![[#REALLINE:]]
  ret void, !dbg !12
}

; CHECK-IL: ![[#FOO:]] = distinct !DISubprogram(name: "foo"
; CHECK-IL: ![[#FAKELINE]] = !DILocation(line: 0, scope: ![[#FOO]])
; CHECK-IL: ![[#REALLINE]] = !DILocation(line: 2, scope: ![[#FOO]])

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1)
!1 = !DIFile(filename: "test.c", directory: "")
!2 = !{}
!3 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !5, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{!"clang version 3.9.0"}
!12 = !DILocation(line: 2, scope: !3)