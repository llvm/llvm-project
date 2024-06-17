; RUN: opt -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -hoist-common-insts=true -S < %s | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -hoist-common-insts=true -S < %s | FileCheck %s
; Verify that we don't crash due an invalid !dbg location on the hoisted llvm.dbg.value

define i64 @caller(ptr %ptr, i64 %flag) !dbg !10 {
init:
  %v9 = icmp eq i64 %flag, 0
  br i1 %v9, label %a, label %b

; CHECK:  %vala = load i64, ptr %ptr
; CHECK-NEXT:  #dbg_value(i64 %vala, [[MD:![0-9]*]]
; CHECK-NEXT:  #dbg_value(i64 %vala, [[MD]]
; CHECK-NEXT:  %valbmasked = and i64 %vala, 1

a:                                              ; preds = %init
  %vala = load i64, ptr %ptr, align 8
  call void @llvm.dbg.value(metadata i64 %vala, metadata !8, metadata !DIExpression()), !dbg !12
  br label %test.exit

b:                                              ; preds = %init
  %valb = load i64, ptr %ptr, align 8
  call void @llvm.dbg.value(metadata i64 %valb, metadata !8, metadata !DIExpression()), !dbg !13
  %valbmasked = and i64 %valb, 1
  br label %test.exit

test.exit:                                      ; preds = %a, %b
  %retv = phi i64 [ %vala, %a ], [ %valbmasked, %b ]
  ret i64 %retv
}

; CHECK: [[MD]] = !DILocalVariable(name: "var"

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !3)
!2 = !DIFile(filename: "optbug", directory: "")
!3 = !{}
!4 = distinct !DISubprogram(name: "callee", scope: !2, file: !2, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !1, retainedNodes: !7)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{!8}
!8 = !DILocalVariable(name: "var", scope: !4, file: !2, type: !9)
!9 = !DIBasicType(name: "var_t", size: 64, encoding: DW_ATE_unsigned)
!10 = distinct !DISubprogram(name: "caller", scope: !2, file: !2, line: 5, type: !5, isLocal: false, isDefinition: true, scopeLine: 5, isOptimized: false, unit: !1, retainedNodes: !3)
!11 = distinct !DILocation(line: 6, scope: !10)
!12 = !DILocation(line: 2, scope: !4, inlinedAt: !11)
!13 = !DILocation(line: 3, scope: !4, inlinedAt: !11)
