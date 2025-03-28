; RUN: opt -passes='require<profile-summary>,function(codegenprepare)' -S -mtriple=x86_64 < %s \
; RUN: | FileCheck %s

;; Check debug locations are propagated onto new PHIs.

; CHECK: .split:
; CHECK-NEXT: %merge = phi i32 [ poison, %while.body ], [ %dest.sroa.clone, %while.body.clone ], !dbg [[DBG:!.*]]

; CHECK: while.body.clone:
; CHECK-NEXT: %dest.sroa.clone = phi i32 [ %1, %.split ], [ poison, %if.else ], !dbg [[DBG]]

define void @test(i1 %c) !dbg !5 {
entry:
  br label %if.else

if.else:                                          ; preds = %if.else1, %entry
  br i1 %c, label %while.body, label %preheader

preheader:                                        ; preds = %if.else
  br label %if.else1

if.then:                                          ; preds = %if.else1
  unreachable

while.body:                                       ; preds = %if.else1, %while.body, %if.else
  %dest.sroa = phi i32 [ %1, %while.body ], [ poison, %if.else1 ], [ poison, %if.else ], !dbg !8
  %0 = inttoptr i32 %dest.sroa to ptr
  %incdec.ptr = getelementptr inbounds i8, ptr %0, i32 -1
  %1 = ptrtoint ptr %incdec.ptr to i32
  store i8 0, ptr %incdec.ptr, align 1
  br label %while.body

if.else1:                                         ; preds = %if.else1, %preheader
  indirectbr ptr poison, [label %if.then, label %while.body, label %if.else, label %if.else1]
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.ll", directory: "/")
!2 = !{i32 11}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "test", linkageName: "test", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 1, column: 1, scope: !5)
