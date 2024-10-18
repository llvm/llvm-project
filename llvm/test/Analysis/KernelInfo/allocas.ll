; Check info on allocas.

; RUN: opt -pass-remarks=kernel-info -passes=kernel-info \
; RUN:     -disable-output %s 2>&1 | \
; RUN:   FileCheck -match-full-lines %s

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @h() !dbg !3 {
entry:
  ; CHECK: remark: test.c:0:0: in artificial function 'h', artificial alloca ('%dyn_ptr.addr') for 'dyn_ptr' with static size of 8 bytes
  %dyn_ptr.addr = alloca ptr, align 8
  ; CHECK: remark: test.c:14:9: in artificial function 'h', alloca ('%i') for 'i' with static size of 4 bytes
  %i = alloca i32, align 4
  ; CHECK: remark: test.c:15:9: in artificial function 'h', alloca ('%a') for 'a' with static size of 8 bytes
  %a = alloca [2 x i32], align 4
  ; CHECK: remark: <unknown>:0:0: in artificial function 'h', alloca ('%nodbg') without debug info with static size of 4 bytes
  %nodbg = alloca i32, align 4
  tail call void @llvm.dbg.declare(metadata ptr %dyn_ptr.addr, metadata !7, metadata !DIExpression()), !dbg !11
  tail call void @llvm.dbg.declare(metadata ptr %i, metadata !12, metadata !DIExpression()), !dbg !15
  tail call void @llvm.dbg.declare(metadata ptr %a, metadata !16, metadata !DIExpression()), !dbg !20
  ret void
}
; CHECK: remark: test.c:13:0: in artificial function 'h', Allocas = 4
; CHECK: remark: test.c:13:0: in artificial function 'h', AllocasStaticSizeSum = 24
; CHECK: remark: test.c:13:0: in artificial function 'h', AllocasDyn = 0

define void @g() !dbg !21 {
entry:
  ; CHECK: remark: test.c:4:7: in function 'g', alloca ('%i') for 'i' with static size of 4 bytes
  %i = alloca i32, align 4
  ; CHECK: remark: test.c:5:7: in function 'g', alloca ('%a') for 'a' with static size of 8 bytes
  %a = alloca [2 x i32], align 4
  tail call void @llvm.dbg.declare(metadata ptr %i, metadata !23, metadata !DIExpression()), !dbg !24
  tail call void @llvm.dbg.declare(metadata ptr %a, metadata !25, metadata !DIExpression()), !dbg !26
  ret void
}
; CHECK: remark: test.c:3:0: in function 'g', Allocas = 2
; CHECK: remark: test.c:3:0: in function 'g', AllocasStaticSizeSum = 12
; CHECK: remark: test.c:3:0: in function 'g', AllocasDyn = 0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

; uselistorder directives
uselistorder ptr @llvm.dbg.declare, { 4, 3, 2, 1, 0 }

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C11, file: !2, producer: "clang version 19.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!2 = !DIFile(filename: "test.c", directory: "/tmp")
!3 = distinct !DISubprogram(name: "h", scope: !2, file: !2, line: 13, type: !4, scopeLine: 13, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !1, retainedNodes: !6)
!4 = distinct !DISubroutineType(types: !5)
!5 = !{null}
!6 = !{}
!7 = !DILocalVariable(name: "dyn_ptr", arg: 1, scope: !3, type: !8, flags: DIFlagArtificial)
!8 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !9)
!9 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !10)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!11 = !DILocation(line: 0, scope: !3)
!12 = !DILocalVariable(name: "i", scope: !13, file: !2, line: 14, type: !14)
!13 = distinct !DILexicalBlock(scope: !3, file: !2, line: 13, column: 3)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DILocation(line: 14, column: 9, scope: !13)
!16 = !DILocalVariable(name: "a", scope: !13, file: !2, line: 15, type: !17)
!17 = !DICompositeType(tag: DW_TAG_array_type, baseType: !14, size: 64, elements: !18)
!18 = !{!19}
!19 = !DISubrange(count: 2)
!20 = !DILocation(line: 15, column: 9, scope: !13)
!21 = distinct !DISubprogram(name: "g", scope: !2, file: !2, line: 3, type: !22, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !6)
!22 = !DISubroutineType(types: !5)
!23 = !DILocalVariable(name: "i", scope: !21, file: !2, line: 4, type: !14)
!24 = !DILocation(line: 4, column: 7, scope: !21)
!25 = !DILocalVariable(name: "a", scope: !21, file: !2, line: 5, type: !17)
!26 = !DILocation(line: 5, column: 7, scope: !21)
