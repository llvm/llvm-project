; Check that debug intrinsics do not affect code generation.

; RUN: llc < %s -mtriple=aarch64-unknown-unknown -mattr=+avx | FileCheck --check-prefix=AARCH64-CHECK %s

define i64 @simulate(<2 x i32> %a) {
entry:
  %rand = tail call i64 @lrand48()
  br label %body

body:                                        ; preds = %body, %entry
  %0 = phi <2 x i32> [ %add, %body ], [ zeroinitializer, %entry ]
  %add = add <2 x i32> %0, %a
  %rand1 = tail call i64 @lrand48() #3
  %cmp = icmp eq i64 %rand1, 0
  br i1 %cmp, label %end, label %body

end:                                        ; preds = %body
  %c = bitcast <2 x i32> %add to i64
  %res = add i64 %rand, %c
  ret i64 %res
}

; AARCH64-CHECK: simulate:
; AARCH64-CHECK: movi v0.2d, #0000000000000000
; AARCH64-CHECK: bl lrand48
; AARCH64-CHECK: mov x19, x0
; AARCH64-CHECK: BB0_1:


define i64 @simulateWithDebugIntrinsic(<2 x i32> %a) local_unnamed_addr  {
entry:
  %rand = tail call i64 @lrand48() #3
  tail call void @llvm.dbg.value(metadata i64 %rand, i64 0, metadata !6, metadata !7), !dbg !8
  br label %body

body:                                        ; preds = %body, %entry
  %0 = phi <2 x i32> [ %add, %body ], [ zeroinitializer, %entry ]
  %add = add <2 x i32> %0, %a
  %rand1 = tail call i64 @lrand48() #3
  %cmp = icmp eq i64 %rand1, 0
  br i1 %cmp, label %end, label %body

end:                                        ; preds = %body
  %c = bitcast <2 x i32> %add to i64
  %res = add i64 %rand, %c
  ret i64 %res
}

; AARCH64-CHECK: simulateWithDebugIntrinsic
; AARCH64-CHECK: movi v0.2d, #0000000000000000
; AARCH64-CHECK: bl lrand48
; AARCH64-CHECK: mov x19, x0
; AARCH64-CHECK: BB1_1:


define i64 @simulateWithDbgDeclare(<2 x i32> %a) local_unnamed_addr  {
entry:
  %rand = tail call i64 @lrand48() #3
  tail call void @llvm.dbg.declare(metadata i64 %rand, metadata !6, metadata !7), !dbg !8
  br label %body

body:                                        ; preds = %body, %entry
  %0 = phi <2 x i32> [ %add, %body ], [ zeroinitializer, %entry ]
  %add = add <2 x i32> %0, %a
  %rand1 = tail call i64 @lrand48() #3
  %cmp = icmp eq i64 %rand1, 0
  br i1 %cmp, label %end, label %body

end:                                        ; preds = %body
  %c = bitcast <2 x i32> %add to i64
  %res = add i64 %rand, %c
  ret i64 %res
}

; AARCH64-CHECK: simulateWithDbgDeclare:
; AARCH64-CHECK: movi v0.2d, #0000000000000000
; AARCH64-CHECK: bl lrand48
; AARCH64-CHECK: mov x19, x0
; AARCH64-CHECK: BB2_1:

declare i64 @lrand48()

declare void @llvm.dbg.value(metadata, i64, metadata, metadata)
declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!3, !4}

!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, runtimeVersion: 0, emissionKind: FullDebug)
!2 = !DIFile(filename: "test.ll", directory: ".")
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "simulateWithDebugIntrinsic", scope: !2, file: !2, line: 64, isLocal: false, isDefinition: true, scopeLine: 65, unit: !1)
!6 = !DILocalVariable(name: "randv", scope: !5, file: !2, line: 69)
!7 = !DIExpression()
!8 = !DILocation(line: 132, column: 2, scope: !5)
