; RUN: opt -passes=hotcoldsplit -hotcoldsplit-threshold=0 -S < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %a, i32 %b) !dbg !2 {
entry:
   %1 = alloca i32, i64 1, align 4
   %2 = alloca i32, i64 1, align 4
   store i32 %a, ptr %1, align 4
   #dbg_declare(ptr %1, !8, !DIExpression(), !1)
   #dbg_value(i32 %b, !9, !DIExpression(), !1)
   %tobool = icmp eq i32 %a, 0
   br i1 %tobool, label %if.then, label %if.end
if.then:                                          ; preds = %entry
  ret void

if.end:                                           ; preds = %entry
   store i32 10, ptr %1, align 4
   %3 = add i32 %b, 1
   store i32 1, ptr %2, align 4
   call void @sink(i32 %3)
   #dbg_declare(ptr %2, !10, !DIExpression(), !1)
   ret void
}

declare void @sink(i32) cold

!llvm.dbg.cu = !{!6}
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !DILocation(line: 11, column: 7, scope: !2)
!2 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, type: !4, spFlags: DISPFlagDefinition, unit: !6)
!3 = !DIFile(filename: "test.c", directory: "")
!4 = !DISubroutineType(cc: DW_CC_program, types: !5)
!5 = !{}
!6 = distinct !DICompileUnit(language: DW_LANG_C, file: !3)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DILocalVariable(name: "a", scope: !2, file: !3, type: !7)
!9 = !DILocalVariable(name: "b", scope: !2, file: !3, type: !7)
!10 = !DILocalVariable(name: "c", scope: !2, file: !3, type: !7)

; CHECK: define {{.*}}@foo.cold.1(ptr %[[ARG0:.*]], i32 %[[ARG1:.*]], ptr %[[ARG2:.*]]){{.*}} !dbg ![[FN:.*]] {
; CHECK-NEXT: newFuncRoot:
; CHECK-NEXT: #dbg_declare(ptr %[[ARG0]], ![[V1:[0-9]+]], {{.*}})
; CHECK-NEXT: #dbg_value(i32 %[[ARG1]], ![[V2:[0-9]+]], {{.*}})
; CHECK-NEXT: br
; CHECK: if.end:
; CHECK:     #dbg_declare(ptr %[[ARG2]], ![[V3:[0-9]+]], {{.*}})
; CHECK: }

; CHECK: ![[V1]] = !DILocalVariable(name: "a", scope: ![[FN]]{{.*}})
; CHECK: ![[V2]] = !DILocalVariable(name: "b", scope: ![[FN]]{{.*}})
; CHECK: ![[V3]] = !DILocalVariable(name: "c", scope: ![[FN]]{{.*}})
