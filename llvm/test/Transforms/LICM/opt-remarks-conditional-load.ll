; RUN: opt < %s -passes=licm -pass-remarks-missed=licm -o /dev/null 2>&1 | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop-mssa(licm)' %s -o /dev/null -pass-remarks-missed=licm 2>&1 | FileCheck %s
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

; With the load from %p conditional, we can't optmize this and the remark
; should tell us about it.

define void @test(ptr %array, ptr noalias %p) {
Entry:
  br label %Loop

Loop:
  %j = phi i32 [ 0, %Entry ], [ %Next, %else]
  %addr = getelementptr i32, ptr %array, i32 %j
  %a = load i32, ptr %addr
  %c = icmp eq i32 %a, 0
  br i1 %c, label %then, label %else

then:
; CHECK: remark: /tmp/kk.c:2:20: failed to hoist load with loop-invariant address because load is conditionally executed
  %b = load i32, ptr %p, !dbg !8
  %a2 = add i32 %a, %b
  store i32 %a2, ptr %addr
  br label %else

else:
  %Next = add i32 %j, 1
  %cond = icmp eq i32 %Next, 0
  br i1 %cond, label %Out, label %Loop

Out:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "/tmp/kk.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"PIC Level", i32 2}
!5 = !{!"clang version 3.9.0 "}
!6 = distinct !DISubprogram(name: "success", scope: !1, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 2, column: 20, scope: !6)
