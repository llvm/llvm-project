; RUN: opt -S -passes=callsite-splitting -o - < %s | FileCheck %s --check-prefixes=CHECK,CHECK-DEBUG
; RUN: opt -S -strip-debug -passes=callsite-splitting -o - < %s | FileCheck %s

define internal i16 @bar(i16 %p1, i16 %p2) {
  %_tmp3 = mul i16 %p2, %p1
  ret i16 %_tmp3
}

define i16 @foo(i16 %in) {
bb0:
  %a = alloca i16, align 4, !DIAssignID !12
  call void @llvm.dbg.assign(metadata i1 undef, metadata !11, metadata !DIExpression(), metadata !12, metadata ptr %a, metadata !DIExpression()), !dbg !8
  store i16 7, ptr %a, align 4, !DIAssignID !13
  br label %bb1

bb1:
  %0 = icmp ne i16 %in, 0
  br i1 %0, label %bb2, label %CallsiteBB

bb2:
  br label %CallsiteBB

CallsiteBB:
  %1 = phi i16 [ 0, %bb1 ], [ 1, %bb2 ]
  %c = phi i16 [ 2, %bb1 ], [ 3, %bb2 ]
  %p = phi ptr [ %a, %bb1 ], [ %a, %bb2 ]
  call void @llvm.dbg.value(metadata i16 %1, metadata !7, metadata !DIExpression()), !dbg !8
  call void @llvm.dbg.value(metadata i16 %c, metadata !7, metadata !DIExpression()), !dbg !8
  call void @llvm.dbg.value(metadata !DIArgList(i16 %1, i16 %c), metadata !7, metadata !DIExpression()), !dbg !8
  call void @llvm.dbg.value(metadata !DIArgList(i16 %c, i16 %c), metadata !7, metadata !DIExpression()), !dbg !8
  call void @llvm.dbg.assign(metadata i16 %1, metadata !11, metadata !DIExpression(), metadata !13, metadata ptr %a, metadata !DIExpression()), !dbg !8
  call void @llvm.dbg.assign(metadata i16 %c, metadata !11, metadata !DIExpression(), metadata !13, metadata ptr %a, metadata !DIExpression()), !dbg !8
  call void @llvm.dbg.assign(metadata i16 %1, metadata !11, metadata !DIExpression(), metadata !13, metadata ptr %p, metadata !DIExpression()), !dbg !8
  %2 = call i16 @bar(i16 %1, i16 5)
  ret i16 %2
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #0
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

attributes #0 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "My Compiler")
!1 = !DIFile(filename: "foo.c", directory: "/bar")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{!"My Compiler"}
!5 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 4, unit: !0)
!7 = !DILocalVariable(name: "c", scope: !6, line: 5, type: !5)
!8 = !DILocation(line: 5, column: 7, scope: !6)
!11 = !DILocalVariable(name: "a", scope: !6, line: 6, type: !5)
!12 = distinct !DIAssignID()
!13 = distinct !DIAssignID()

; The optimization should trigger even in the presence of the dbg.value in
; CallSiteBB.

; CHECK-LABEL: @foo
; CHECK-LABEL: bb1.split:
; CHECK-DEBUG: #dbg_value(i16 0, ![[DBG_1:[0-9]+]], {{.*}}
; CHECK-DEBUG: #dbg_value(i16 2, ![[DBG_1]], {{.*}}
; CHECK-DEBUG: #dbg_value(!DIArgList(i16 0, i16 2), {{.*}}
; CHECK-DEBUG: #dbg_value(!DIArgList(i16 2, i16 2), {{.*}}
; CHECK-DEBUG: #dbg_assign(i16 0, ![[DBG_2:[0-9]+]], {{.*}}
; CHECK-DEBUG: #dbg_assign(i16 2, ![[DBG_2]], {{.*}}
; CHECK-DEBUG: #dbg_assign(i16 0, ![[DBG_2]], !DIExpression(), ![[ID_1:[0-9]+]], ptr %a, {{.*}}
; CHECK: [[TMP1:%[0-9]+]] = call i16 @bar(i16 0, i16 5)

; CHECK-LABEL: bb2.split:
; CHECK-DEBUG: #dbg_value(i16 1, ![[DBG_1]], {{.*}}
; CHECK-DEBUG: #dbg_value(i16 3, ![[DBG_1]], {{.*}}
; CHECK-DEBUG: #dbg_value(!DIArgList(i16 1, i16 3), {{.*}}
; CHECK-DEBUG: #dbg_value(!DIArgList(i16 3, i16 3), {{.*}}
; CHECK-DEBUG: #dbg_assign(i16 1, ![[DBG_2]], {{.*}}
; CHECK-DEBUG: #dbg_assign(i16 3, ![[DBG_2]], {{.*}}
; CHECK-DEBUG: #dbg_assign(i16 1, ![[DBG_2]], !DIExpression(), ![[ID_1:[0-9]+]], ptr %a, {{.*}}
; CHECK: [[TMP2:%[0-9]+]] = call i16 @bar(i16 1, i16 5)

; CHECK-LABEL: CallsiteBB
; CHECK: %phi.call = phi i16 [ [[TMP2]], %bb2.split ], [ [[TMP1]], %bb1.split

; CHECK-DEBUG-DAG: ![[DBG_1]] = !DILocalVariable(name: "c"{{.*}})
; CHECK-DEBUG-DAG: ![[DBG_2]] = !DILocalVariable(name: "a"{{.*}})
; CHECK-DEBUG-DAG: ![[ID_1]] = distinct !DIAssignID()
