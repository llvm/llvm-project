; Check info on calls.

; RUN: opt -pass-remarks=kernel-info -passes=kernel-info \
; RUN:     -disable-output %s 2>&1 | \
; RUN:   FileCheck -match-full-lines %s

target datalayout = "e-i65:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare void @personality()

define void @h() personality ptr @personality !dbg !100 {
entry:
  ; CHECK: remark: test.c:16:5: in artificial function 'h', direct call, callee is 'f'
  call void @f(), !dbg !102
  ; CHECK: remark: test.c:17:5: in artificial function 'h', direct call to defined function, callee is 'g'
  call void @g(), !dbg !104
  ; CHECK: remark: test.c:18:5: in artificial function 'h', direct call to defined function, callee is artificial 'h'
  call void @h(), !dbg !105
  %0 = load ptr, ptr null, align 8
  ; CHECK: remark: test.c:19:5: in artificial function 'h', indirect call
  call void %0(), !dbg !106
  ; CHECK: remark: test.c:20:5: in artificial function 'h', direct invoke, callee is 'f'
  invoke void @f() to label %fcont unwind label %cleanup, !dbg !107
fcont:
  ; CHECK: remark: test.c:21:5: in artificial function 'h', direct invoke to defined function, callee is 'g'
  invoke void @g() to label %gcont unwind label %cleanup, !dbg !108
gcont:
  ; CHECK: remark: test.c:22:5: in artificial function 'h', direct invoke to defined function, callee is artificial 'h'
  invoke void @h() to label %hcont unwind label %cleanup, !dbg !109
hcont:
  ; CHECK: remark: test.c:23:5: in artificial function 'h', indirect invoke
  invoke void %0() to label %end unwind label %cleanup, !dbg !110
cleanup:
  %ll = landingpad { ptr, i32 }
  cleanup
  br label %end
end:
  ret void
}
; CHECK: remark: test.c:13:0: in artificial function 'h', DirectCalls = 6
; CHECK: remark: test.c:13:0: in artificial function 'h', IndirectCalls = 2
; CHECK: remark: test.c:13:0: in artificial function 'h', DirectCallsToDefinedFunctions = 4
; CHECK: remark: test.c:13:0: in artificial function 'h', Invokes = 4

declare void @f()

define void @g() personality ptr @personality !dbg !200 {
entry:
  ; CHECK: remark: test.c:6:3: in function 'g', direct call, callee is 'f'
  call void @f(), !dbg !202
  ; CHECK: remark: test.c:7:3: in function 'g', direct call to defined function, callee is 'g'
  call void @g(), !dbg !203
  ; CHECK: remark: test.c:8:3: in function 'g', direct call to defined function, callee is artificial 'h'
  call void @h(), !dbg !204
  %0 = load ptr, ptr null, align 8
  ; CHECK: remark: test.c:9:3: in function 'g', indirect call
  call void %0(), !dbg !205
  ; CHECK: remark: test.c:10:3: in function 'g', direct invoke, callee is 'f'
  invoke void @f() to label %fcont unwind label %cleanup, !dbg !206
fcont:
  ; CHECK: remark: test.c:11:3: in function 'g', direct invoke to defined function, callee is 'g'
  invoke void @g() to label %gcont unwind label %cleanup, !dbg !207
gcont:
  ; CHECK: remark: test.c:12:3: in function 'g', direct invoke to defined function, callee is artificial 'h'
  invoke void @h() to label %hcont unwind label %cleanup, !dbg !208
hcont:
  ; CHECK: remark: test.c:13:3: in function 'g', indirect invoke
  invoke void %0() to label %end unwind label %cleanup, !dbg !209
cleanup:
  %ll = landingpad { ptr, i32 }
  cleanup
  br label %end
end:
  ret void
}
; CHECK: remark: test.c:3:0: in function 'g', DirectCalls = 6
; CHECK: remark: test.c:3:0: in function 'g', IndirectCalls = 2
; CHECK: remark: test.c:3:0: in function 'g', DirectCallsToDefinedFunctions = 4
; CHECK: remark: test.c:3:0: in function 'g', Invokes = 4

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C11, file: !2, producer: "clang version 19.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!2 = !DIFile(filename: "test.c", directory: "/tmp")
!3 = !{null}
!4 = !{}

!100 = distinct !DISubprogram(name: "h", scope: !2, file: !2, line: 13, type: !101, scopeLine: 13, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !1, retainedNodes: !4)
!101 = distinct !DISubroutineType(types: !3)
!102 = !DILocation(line: 16, column: 5, scope: !103)
!103 = distinct !DILexicalBlock(scope: !100, file: !2, line: 13, column: 3)
!104 = !DILocation(line: 17, column: 5, scope: !103)
!105 = !DILocation(line: 18, column: 5, scope: !103)
!106 = !DILocation(line: 19, column: 5, scope: !103)
!107 = !DILocation(line: 20, column: 5, scope: !103)
!108 = !DILocation(line: 21, column: 5, scope: !103)
!109 = !DILocation(line: 22, column: 5, scope: !103)
!110 = !DILocation(line: 23, column: 5, scope: !103)

!200 = distinct !DISubprogram(name: "g", scope: !2, file: !2, line: 3, type: !201, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !4)
!201 = !DISubroutineType(types: !3)
!202 = !DILocation(line: 6, column: 3, scope: !200)
!203 = !DILocation(line: 7, column: 3, scope: !200)
!204 = !DILocation(line: 8, column: 3, scope: !200)
!205 = !DILocation(line: 9, column: 3, scope: !200)
!206 = !DILocation(line: 10, column: 3, scope: !200)
!207 = !DILocation(line: 11, column: 3, scope: !200)
!208 = !DILocation(line: 12, column: 3, scope: !200)
!209 = !DILocation(line: 13, column: 3, scope: !200)
