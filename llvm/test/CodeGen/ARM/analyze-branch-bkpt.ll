; RUN: llc -o - %s -mtriple thumbv4-unknown-linux-android | FileCheck --check-prefix=V4 %s
; RUN: llc -o - %s -mtriple thumbv5-unknown-linux-android | FileCheck --check-prefix=V5 %s

; V4: udf #254
; V5: bkpt #0

define i1 @a(i32 %b) !dbg !3 {
  br i1 undef, label %c, label %d, !dbg !4

d:                                                ; preds = %0
  call void @llvm.debugtrap()
  br label %ah, !dbg !4

c:                                                ; preds = %0
  %aj = icmp ne i20 undef, 5
  br label %ah, !dbg !4

ah:                                               ; preds = %c, %d
  %ak = phi i1 [ false, %d ], [ %aj, %c ]
  call void @llvm.dbg.value(metadata i1 %ak, metadata !7, metadata !DIExpression()), !dbg !9
  switch i32 %b, label %al [
    i32 0, label %am
    i32 10, label %an
  ]

an:                                               ; preds = %ah
  %ch = select i1 %ak, i32 0, i32 5
  br label %am, !dbg !10

al:                                               ; preds = %ah
  br label %am, !dbg !9

am:                                               ; preds = %al, %an, %ah
  %1 = phi i32 [ 0, %al ], [ %ch, %an ], [ %b, %ah ]
  unreachable
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

; Function Attrs: nounwind
declare void @llvm.debugtrap() #1

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug)
!1 = !DIFile(filename: "a", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(scope: null, isLocal: false, isDefinition: true, isOptimized: false, unit: !0)
!4 = !DILocation(line: 0, scope: !5, inlinedAt: !6)
!5 = distinct !DISubprogram(scope: null, isLocal: false, isDefinition: true, isOptimized: false, unit: !0)
!6 = !DILocation(line: 0, scope: !3)
!7 = !DILocalVariable(scope: !8)
!8 = distinct !DISubprogram(scope: null, isLocal: false, isDefinition: true, isOptimized: false, unit: !0)
!9 = !DILocation(line: 0, scope: !8, inlinedAt: !6)
!10 = !DILocation(line: 0, scope: !11, inlinedAt: !6)
!11 = !DILexicalBlock(scope: !8)
