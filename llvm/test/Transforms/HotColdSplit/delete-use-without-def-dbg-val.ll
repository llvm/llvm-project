; RUN: opt -hotcoldsplit -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; CHECK-LABEL: define {{.*}}@foo(
; CHECK-NOT: call {{.*}}llvm.dbg.value

; CHECK-LABEL: define {{.*}}@foo.cold
; CHECK-NOT: call {{.*}}llvm.dbg.value

define void @foo() !dbg !6 {
entry:
  br i1 undef, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  br label %cleanup

if.end:                                           ; preds = %entry
  ; We expect this block to be outlined. That kills the definition of %var.
  %var = add i32 0, 0, !dbg !11
  call void @sink()
  call void @sink()
  call void @sink()
  br label %cleanup

cleanup:
  ; This dbg.value should be deleted after outlining, otherwise the verifier
  ; complains about function-local metadata being used outside of a function.
  call void @llvm.dbg.value(metadata i32 %var, metadata !9, metadata !DIExpression()), !dbg !11
  ret void
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

declare void @sink() cold

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !4}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{}
!3 = !{i32 7}
!4 = !{i32 1}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, retainedNodes: !8)
!7 = !DISubroutineType(types: !2)
!8 = !{!9}
!9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
!10 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!11 = !DILocation(line: 1, column: 1, scope: !6)
