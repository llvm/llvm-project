; RUN: opt -S -callsite-splitting -o - < %s | FileCheck %s
; RUN: opt -S -strip-debug -callsite-splitting -o - < %s | FileCheck %s

define internal i16 @bar(i16 %p1, i16 %p2) {
  %_tmp3 = mul i16 %p2, %p1
  ret i16 %_tmp3
}

define i16 @foo(i16 %in) {
bb0:
  br label %bb1

bb1:
  %0 = icmp ne i16 %in, 0
  br i1 %0, label %bb2, label %CallsiteBB

bb2:
  br label %CallsiteBB

CallsiteBB:
  %1 = phi i16 [ 0, %bb1 ], [ 1, %bb2 ]
  %c = phi i16 [ 2, %bb1 ], [ 3, %bb2 ]
  call void @llvm.dbg.value(metadata i16 %c, metadata !7, metadata !DIExpression()), !dbg !8
  %2 = call i16 @bar(i16 %1, i16 5)
  ret i16 %2
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

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

; The optimization should trigger even in the presence of the dbg.value in
; CallSiteBB.

; CHECK-LABEL: @foo
; CHECK-LABEL: CallsiteBB.predBB1.split:
; CHECK: [[TMP1:%[0-9]+]] = call i16 @bar(i16 1, i16 5)
; CHECK-LABEL: CallsiteBB.predBB2.split:
; CHECK: [[TMP2:%[0-9]+]] = call i16 @bar(i16 0, i16 5)
; CHECK-LABEL: CallsiteBB
; CHECK: %phi.call = phi i16 [ [[TMP1]], %CallsiteBB.predBB1.split ], [ [[TMP2]], %CallsiteBB.predBB2.split

