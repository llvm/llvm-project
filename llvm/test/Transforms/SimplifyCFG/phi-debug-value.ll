; RUN: opt -passes=simplifycfg -S < %s | FileCheck %s

; The PHI value is specialized independently in each predecessor when the PHI
; block is duplicated. Its debug value must be specialized in the same way.

; CHECK-LABEL: define void @phi_extra_use(
; CHECK:       cond.true:
; CHECK-NEXT:    call void @foo()
; CHECK-NEXT:      #dbg_value(i1 true, ![[VAR:[0-9]+]], !DIExpression(), ![[LOC:[0-9]+]])
; CHECK-NEXT:    call void @use_bool(i1 true)
; CHECK:       cond.false:
; CHECK-NEXT:    call void @bar()
; CHECK-NEXT:      #dbg_value(i1 false, ![[VAR]], !DIExpression(), ![[LOC]])
; CHECK-NEXT:    call void @use_bool(i1 false)

declare void @foo()
declare void @bar()
declare void @use_bool(i1)

define void @phi_extra_use(i1 %c1) !dbg !5 {
entry:
  br i1 %c1, label %cond.true, label %cond.false

cond.true:
  call void @foo()
  br label %cond.end

cond.false:
  call void @bar()
  br label %cond.end

cond.end:
  %cond = phi i1 [ true, %cond.true ], [ false, %cond.false ]
    #dbg_value(i1 %cond, !7, !DIExpression(), !6)
  call void @use_bool(i1 %cond)
  br i1 %cond, label %if.then, label %if.else

if.then:
  call void @foo()
  br label %if.end

if.else:
  call void @bar()
  br label %if.end

if.end:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}
!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "llvm", emissionKind: FullDebug)
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !DIFile(filename: "phi-debug-value.c", directory: "/tmp")
!4 = !DISubroutineType(types: !{})
!5 = distinct !DISubprogram(name: "phi_extra_use", scope: !3, file: !3, line: 1, type: !4, unit: !0, spFlags: DISPFlagDefinition)
!6 = !DILocation(line: 2, scope: !5)
!7 = !DILocalVariable(name: "c2", scope: !5, file: !3, line: 2, type: !8)
!8 = !DIBasicType(name: "bool", size: 1, encoding: DW_ATE_boolean)
