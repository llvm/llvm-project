; Test DBG_LABEL MachineInstr for label debugging.
; REQUIRES: asserts
; RUN: llc -debug-only=isel %s -o /dev/null 2> %t.debug
; RUN: cat %t.debug | FileCheck %s --check-prefix=CHECKMI
;
; CHECKMI: DBG_LABEL "top", debug-location !9
; CHECKMI: DBG_LABEL "done", debug-location !11
;
; RUN: llc %s -o - | FileCheck %s --check-prefix=CHECKASM
;
; CHECKASM: DEBUG_LABEL: foo:top
; CHECKASM: DEBUG_LABEL: foo:done

source_filename = "debug-label-mi.c"

; Function Attrs: noinline nounwind optnone
define i32 @foo(i32 signext %a, i32 signext %b) #0 !dbg !4 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %sum = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 %b, i32* %b.addr, align 4
  br label %top

top:                                              ; preds = %entry
  call void @llvm.dbg.label(metadata !8), !dbg !9
  %0 = load i32, i32* %a.addr, align 4
  %1 = load i32, i32* %b.addr, align 4
  %add = add nsw i32 %0, %1
  store i32 %add, i32* %sum, align 4
  br label %done

done:                                             ; preds = %top
  call void @llvm.dbg.label(metadata !10), !dbg !11
  %2 = load i32, i32* %sum, align 4
  ret i32 %2
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.label(metadata)

attributes #0 = { noinline nounwind optnone uwtable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: false, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "debug-label-mi.c", directory: "./")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DILabel(scope: !4, name: "top", file: !1, line: 4)
!9 = !DILocation(line: 4, column: 1, scope: !4)
!10 = !DILabel(scope: !4, name: "done", file: !1, line: 7)
!11 = !DILocation(line: 7, column: 1, scope: !4)
