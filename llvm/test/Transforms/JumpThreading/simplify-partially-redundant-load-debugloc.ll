; RUN: opt -S -passes=jump-threading < %s | FileCheck %s

;; Check that when we simplify the load of %p by replacing it with 0 when coming
;; from %left, the inttoptr cast instruction is assigned the DILocation from the
;; load instruction.

; CHECK-LABEL: define void @foo(

; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[P:%.*]] = alloca i32, align 8
; CHECK-NEXT:    br i1 {{.*}}, label %[[LEFT:.*]], label %[[RIGHT:.*]]

;; Cast in "left" should have the load's debug location.
; CHECK:       [[LEFT]]:
; CHECK-NEXT:    call void @llvm.memset.p0.i64(ptr [[P]], i8 0, i64 8, i1 false)
; CHECK-NEXT:    [[TMP0:%.*]] = inttoptr i64 0 to ptr, !dbg [[LOAD_LOC:![0-9]+]]
; CHECK-NEXT:    br label %[[END:.*]]

; CHECK:       [[RIGHT]]:
; CHECK-NEXT:    br i1 {{.*}}, label %[[RIGHT_LEFT:.*]], label %[[ENDTHREAD_PRE_SPLIT:.*]]

;; Cast in "right.left" should not have a debug location, as we are not
;; guaranteed to reach the load's original position.
; CHECK:       [[RIGHT_LEFT]]:
; CHECK-NEXT:    call void @llvm.memset.p0.i64(ptr [[P]], i8 0, i64 8, i1 false)
; CHECK-NEXT:    [[TMP1:%.*]] = inttoptr i64 0 to ptr
; CHECK-NEXT:    br i1 {{.*}}, label %[[END]], label %[[EXIT:.*]]

;; Load in "right.right" should have the load's debug location.
; CHECK:       [[ENDTHREAD_PRE_SPLIT]]:
; CHECK-NEXT:    [[DOTPR:%.*]] = load ptr, ptr [[P]], align 8, !dbg [[LOAD_LOC]]
; CHECK-NEXT:    br label %[[END]], !dbg [[LOAD_LOC]]

;; Finally, the PHI node should also have the load's debug location.
; CHECK:       [[END]]:
; CHECK-NEXT:    [[TMP2:%.*]] = phi ptr [ [[DOTPR]], %[[ENDTHREAD_PRE_SPLIT]] ], [ [[TMP1]], %[[RIGHT_LEFT]] ], [ [[TMP0]], %[[LEFT]] ], !dbg [[LOAD_LOC]]

; CHECK: [[LOAD_LOC]] = !DILocation(line: 1, column: 1,

define void @foo(i1 %b, i1 %c, i1 %d) !dbg !5 {
entry:
  %p = alloca i32, align 8
  br i1 %b, label %left, label %right

left:
  call void @llvm.memset.p0.i64(ptr %p, i8 0, i64 8, i1 false)
  br label %end

right:
  br i1 %c, label %right.left, label %right.right

right.left:
  call void @llvm.memset.p0.i64(ptr %p, i8 0, i64 8, i1 false)
  br i1 %d, label %end, label %exit

right.right:
  br label %end

end:
  %0 = load ptr, ptr %p, align 8, !dbg !8
  %isnull = icmp eq ptr %0, null
  br i1 %isnull, label %exit, label %notnull

notnull:
  br label %exit

exit:
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #0

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: write) }

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "simplify-partially-redundant-load-debugloc.ll", directory: "/")
!2 = !{i32 13}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !0, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 1, column: 1, scope: !5)
