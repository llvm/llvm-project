; RUN: opt %s -S --passes="loop-mssa(simple-loop-unswitch<nontrivial>)" -o - \
; RUN: | FileCheck %s

;; Unswitch from:
;;                    +- noclobber -+
;;                    |             v
;; entry -> loop.header             loop.latch -> exit.
;;          ^         |             ^        |
;;          |         +- clobber ---+        |
;;          +--------------------------------+
;;
;; To (unswitched loop unconditionally branches to noclobber.us):
;;                          +---------------------------------------------+
;;                          V                                             |
;;     +- entry.split.us -> loop.header.us -> noclobber.us -> loop.latch.us -> exit.split.us -+
;;     |                                                                                      |
;; entry                                                                                      |
;;     |                                                                                      v
;;     +- entry.split -> loop.header -> noclobber -> loop.latch -> exit.split --------------> exit.
;;                       ^         |                 ^        |
;;                       |         +--> clobber -----+        |
;;                       +------------------------------------+
;;
;; Check the duplicated instructions get remapped source atoms. Note some
;; instructions get duplicated from loop.header into entry too.

; CHECK-LABEL: define i32 @partial_unswitch_true_successor_hoist_invariant(
; CHECK-SAME: ptr [[PTR:%.*]], i32 [[N:%.*]]) !dbg [[DBG5:![0-9]+]] {

;; Instructions duplicated from loop.header need remapped atoms.
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr i32, ptr [[PTR]], i64 1, !dbg [[DBG8:![0-9]+]]
; CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr [[TMP0]], align 4, !dbg [[DBG9:![0-9]+]]
; CHECK-NEXT:    [[TMP2:%.*]] = icmp eq i32 [[TMP1]], 100, !dbg [[DBG10:![0-9]+]]
; CHECK-NEXT:    br i1 [[TMP2]], label %[[ENTRY_SPLIT_US:.*]], label %[[ENTRY_SPLIT:.*]]

;; Same source location as entry's original br, with remapped atom group.
; CHECK:       [[ENTRY_SPLIT_US]]:
; CHECK-NEXT:    br label %[[LOOP_HEADER_US:.*]], !dbg [[DBG11:![0-9]+]]

;; Instructions duplicated from loop.header need remapped atoms.
; CHECK:       [[LOOP_HEADER_US]]:
; CHECK-NEXT:    [[IV_US:%.*]] = phi i32 [ 0, %[[ENTRY_SPLIT_US]] ], [ [[IV_NEXT_US:%.*]], %[[LOOP_LATCH_US:.*]] ], !dbg [[DBG12:![0-9]+]]
; CHECK-NEXT:    br label %[[NOCLOBBER_US:.*]], !dbg [[DBG13:![0-9]+]]

;; Instructions duplicated from noclobber need remapped atoms.
; CHECK:       [[NOCLOBBER_US]]:
; CHECK-NEXT:    br label %[[LOOP_LATCH_US]], !dbg [[DBG14:![0-9]+]]

; CHECK:       [[LOOP_LATCH_US]]:
; CHECK-NEXT:    [[C_US:%.*]] = icmp ult i32 [[IV_US]], [[N]], !dbg [[DBG15:![0-9]+]]
; CHECK-NEXT:    [[IV_NEXT_US]] = add i32 [[IV_US]], 1, !dbg [[DBG16:![0-9]+]]
; CHECK-NEXT:    br i1 [[C_US]], label %[[LOOP_HEADER_US]], label %[[EXIT_SPLIT_US:.*]], !dbg [[DBG17:![0-9]+]]

;; Split from exit, this DILocation shouldn't have source atom info.
; CHECK:       [[EXIT_SPLIT_US]]:
; CHECK-NEXT:    br label %[[EXIT:.*]], !dbg [[DBG18:![0-9]+]]

;; Same source location as entry's original br, with remapped atom group.
; CHECK:       [[ENTRY_SPLIT]]:
; CHECK-NEXT:    br label %[[LOOP_HEADER:.*]], !dbg [[DBG19:![0-9]+]]

;; Original loop blocks - the atoms groups should be distinct from those
;; on duplicated instructions in the blocks above.
; CHECK:       [[LOOP_HEADER]]:
; CHECK-NEXT:    [[IV:%.*]] = phi i32 [ 0, %[[ENTRY_SPLIT]] ], [ [[IV_NEXT:%.*]], %[[LOOP_LATCH:.*]] ], !dbg [[DBG20:![0-9]+]]
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr i32, ptr [[PTR]], i64 1, !dbg [[DBG21:![0-9]+]]
; CHECK-NEXT:    [[LV:%.*]] = load i32, ptr [[GEP]], align 4, !dbg [[DBG22:![0-9]+]]
; CHECK-NEXT:    [[SC:%.*]] = icmp eq i32 [[LV]], 100, !dbg [[DBG23:![0-9]+]]
; CHECK-NEXT:    br i1 [[SC]], label %[[NOCLOBBER:.*]], label %[[CLOBBER:.*]], !dbg [[DBG24:![0-9]+]]
; CHECK:       [[NOCLOBBER]]:
; CHECK-NEXT:    br label %[[LOOP_LATCH]], !dbg [[DBG25:![0-9]+]]
; CHECK:       [[CLOBBER]]:
; CHECK-NEXT:    call void @clobber(), !dbg [[DBG26:![0-9]+]]
; CHECK-NEXT:    br label %[[LOOP_LATCH]], !dbg [[DBG27:![0-9]+]]
; CHECK:       [[LOOP_LATCH]]:
; CHECK-NEXT:    [[C:%.*]] = icmp ult i32 [[IV]], [[N]], !dbg [[DBG28:![0-9]+]]
; CHECK-NEXT:    [[IV_NEXT]] = add i32 [[IV]], 1, !dbg [[DBG29:![0-9]+]]
; CHECK-NEXT:    br i1 [[C]], label %[[LOOP_HEADER]], label %[[EXIT_SPLIT:.*]], !dbg [[DBG30:![0-9]+]]

;; Split from exit, this DILocation shouldn't have source atom info.
; CHECK:       [[EXIT_SPLIT]]:
; CHECK-NEXT:    br label %[[EXIT]], !dbg [[DBG18]]

;; exit.split and exit.split.us take the source location from here but drop its
;; source atom info.
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    ret i32 10, !dbg [[DBG33:![0-9]+]]

define i32 @partial_unswitch_true_successor_hoist_invariant(ptr %ptr, i32 %N) !dbg !5 {
entry:
  br label %loop.header, !dbg !8

loop.header:                                      ; preds = %loop.latch, %entry
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ], !dbg !9
  %gep = getelementptr i32, ptr %ptr, i64 1, !dbg !10
  %lv = load i32, ptr %gep, align 4, !dbg !11
  %sc = icmp eq i32 %lv, 100, !dbg !12
  br i1 %sc, label %noclobber, label %clobber, !dbg !13

noclobber:                                        ; preds = %loop.header
  br label %loop.latch, !dbg !14

clobber:                                          ; preds = %loop.header
  call void @clobber(), !dbg !15
  br label %loop.latch, !dbg !16

loop.latch:                                       ; preds = %clobber, %noclobber
  %c = icmp ult i32 %iv, %N, !dbg !17
  %iv.next = add i32 %iv, 1, !dbg !18
  br i1 %c, label %loop.header, label %exit, !dbg !19

exit:                                             ; preds = %loop.latch
  ret i32 10, !dbg !20
}

declare void @clobber()

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.ll", directory: "/")
!2 = !{i32 13}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "partial_unswitch_true_successor_hoist_invariant", linkageName: "partial_unswitch_true_successor_hoist_invariant", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, keyInstructions: true)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 1, scope: !5, atomGroup: 1, atomRank: 1)
!9 = !DILocation(line: 2, scope: !5, atomGroup: 2, atomRank: 1)
!10 = !DILocation(line: 3, scope: !5, atomGroup: 3, atomRank: 1)
!11 = !DILocation(line: 4, scope: !5, atomGroup: 4, atomRank: 1)
!12 = !DILocation(line: 5, scope: !5, atomGroup: 5, atomRank: 1)
!13 = !DILocation(line: 6, scope: !5, atomGroup: 6, atomRank: 1)
!14 = !DILocation(line: 7, scope: !5, atomGroup: 7, atomRank: 1)
!15 = !DILocation(line: 8, scope: !5, atomGroup: 8, atomRank: 1)
!16 = !DILocation(line: 9, scope: !5, atomGroup: 9, atomRank: 1)
!17 = !DILocation(line: 10, scope: !5, atomGroup: 10, atomRank: 1)
!18 = !DILocation(line: 11, scope: !5, atomGroup: 11, atomRank: 1)
!19 = !DILocation(line: 12, scope: !5, atomGroup: 12, atomRank: 1)
!20 = !DILocation(line: 13, scope: !5, atomGroup: 13, atomRank: 1)
;.
; CHECK: distinct !DISubprogram(name: "partial_unswitch_true_successor_hoist_invariant", {{.*}}keyInstructions: true)
; CHECK: [[DBG8]] = !DILocation(line: 3{{.*}}, atomGroup: 24, atomRank: 1)
; CHECK: [[DBG9]] = !DILocation(line: 4{{.*}}, atomGroup: 25, atomRank: 1)
; CHECK: [[DBG10]] = !DILocation(line: 5{{.*}}, atomGroup: 26, atomRank: 1)
; CHECK: [[DBG11]] = !DILocation(line: 1{{.*}}, atomGroup: 14, atomRank: 1)
; CHECK: [[DBG12]] = !DILocation(line: 2{{.*}}, atomGroup: 15, atomRank: 1)
; CHECK: [[DBG13]] = !DILocation(line: 6{{.*}}, atomGroup: 19, atomRank: 1)
; CHECK: [[DBG14]] = !DILocation(line: 7{{.*}}, atomGroup: 20, atomRank: 1)
; CHECK: [[DBG15]] = !DILocation(line: 10{{.*}}, atomGroup: 21, atomRank: 1)
; CHECK: [[DBG16]] = !DILocation(line: 11{{.*}}, atomGroup: 22, atomRank: 1)
; CHECK: [[DBG17]] = !DILocation(line: 12{{.*}}, atomGroup: 23, atomRank: 1)
; CHECK: [[DBG18]] = !DILocation(line: 13, scope: ![[#]])
; CHECK: [[DBG19]] = !DILocation(line: 1{{.*}}, atomGroup: 1, atomRank: 1)
; CHECK: [[DBG20]] = !DILocation(line: 2{{.*}}, atomGroup: 2, atomRank: 1)
; CHECK: [[DBG21]] = !DILocation(line: 3{{.*}}, atomGroup: 3, atomRank: 1)
; CHECK: [[DBG22]] = !DILocation(line: 4{{.*}}, atomGroup: 4, atomRank: 1)
; CHECK: [[DBG23]] = !DILocation(line: 5{{.*}}, atomGroup: 5, atomRank: 1)
; CHECK: [[DBG24]] = !DILocation(line: 6{{.*}}, atomGroup: 6, atomRank: 1)
; CHECK: [[DBG25]] = !DILocation(line: 7{{.*}}, atomGroup: 7, atomRank: 1)
; CHECK: [[DBG26]] = !DILocation(line: 8{{.*}}, atomGroup: 8, atomRank: 1)
; CHECK: [[DBG27]] = !DILocation(line: 9{{.*}}, atomGroup: 9, atomRank: 1)
; CHECK: [[DBG28]] = !DILocation(line: 10{{.*}}, atomGroup: 10, atomRank: 1)
; CHECK: [[DBG29]] = !DILocation(line: 11{{.*}}, atomGroup: 11, atomRank: 1)
; CHECK: [[DBG30]] = !DILocation(line: 12{{.*}}, atomGroup: 12, atomRank: 1)
; CHECK: [[DBG33]] = !DILocation(line: 13{{.*}}, atomGroup: 13, atomRank: 1)
;.
