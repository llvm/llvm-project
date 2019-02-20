; RUN: opt %s -tailcallelim -verify-dom-info -pass-remarks=tailcallelim -o /dev/null 2>&1 | FileCheck %s
; RUN: opt %s -o /dev/null -passes='require<opt-remark-emit>,tailcallelim' -pass-remarks=tailcallelim 2>&1 | FileCheck %s

; CHECK: /home/davide/pat.c:2:20: transforming tail recursion into loop
define i32 @fib(i32 %n) nounwind ssp {
entry:
  %cmp = icmp slt i32 %n, 2
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  br label %return

if.end:                                           ; preds = %entry
  %sub = add nsw i32 %n, -2
  %call = call i32 @fib(i32 %sub)
  %sub3 = add nsw i32 %n, -1
  %call4 = call i32 @fib(i32 %sub3), !dbg !8
  %add = add nsw i32 %call, %call4
  br label %return

return:                                           ; preds = %if.end, %if.then
  %retval.0 = phi i32 [ 1, %if.then ], [ %add, %if.end ]
  ret i32 %retval.0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "/home/davide/pat.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"PIC Level", i32 2}
!5 = !{!"clang version 3.9.0 "}
!6 = distinct !DISubprogram(name: "success", scope: !1, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 2, column: 20, scope: !6)
