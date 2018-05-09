; Test bitcode writer/reader for DILabel metadata.
; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s
;
; CHECK: top:
; CHECK: call void @llvm.dbg.label(metadata [[LABEL_METADATA:![0-9]+]])
; CHECK: distinct !DISubprogram(name: "foo", {{.*}}, retainedNodes: [[ELEMENTS:![0-9]+]])
; CHECK: [[ELEMENTS]] = !{[[LABEL_METADATA]]}
; CHECK: [[LABEL_METADATA]] = !DILabel({{.*}}, name: "top", {{.*}}, line: 4)

source_filename = "debug-label-bitcode.c"

; Function Attrs: noinline nounwind optnone
define i32 @foo(i32 signext %a, i32 signext %b) !dbg !4 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %sum = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 %b, i32* %b.addr, align 4
  br label %top

top:                                              ; preds = %entry
  call void @llvm.dbg.label(metadata !9), !dbg !10
  %0 = load i32, i32* %a.addr, align 4
  %1 = load i32, i32* %b.addr, align 4
  %add = add nsw i32 %0, %1
  store i32 %add, i32* %sum, align 4
  br label %done

done:                                             ; preds = %top
  call void @llvm.dbg.label(metadata !11), !dbg !12
  %2 = load i32, i32* %sum, align 4
  ret i32 %2
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.label(metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang 6.0.0", isOptimized: false, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "debug-label-bitcode.c", directory: "./")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, unit: !0, retainedNodes: !5)
!5 = !{!9}
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8, !8}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DILabel(scope: !4, name: "top", file: !1, line: 4)
!10 = !DILocation(line: 4, column: 1, scope: !4)
!11 = !DILabel(scope: !4, name: "done", file: !1, line: 7)
!12 = !DILocation(line: 7, column: 1, scope: !4)
