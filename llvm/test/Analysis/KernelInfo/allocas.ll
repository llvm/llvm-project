; Check info on allocas.

; RUN: opt -pass-remarks=kernel-info -passes=kernel-info \
; RUN:     -disable-output %s 2>&1 | \
; RUN:   FileCheck -match-full-lines %s

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @h() !dbg !100 {
entry:
  ; CHECK: remark: test.c:0:0: in artificial function 'h', artificial alloca ('%dyn_ptr.addr') for 'dyn_ptr' with static size of 8 bytes
  %dyn_ptr.addr = alloca ptr, align 8
  ; CHECK: remark: test.c:14:9: in artificial function 'h', alloca ('%i') for 'i' with static size of 4 bytes
  %i = alloca i32, align 4
  ; CHECK: remark: test.c:15:9: in artificial function 'h', alloca ('%a') for 'a' with static size of 8 bytes
  %a = alloca [2 x i32], align 4
  %size = load i32, ptr %i, align 4
  ; CHECK: remark: test.c:16:9: in artificial function 'h', alloca ('%adyn') for 'adyn' with dynamic size
  %adyn = alloca i32, i32 %size, align 4
  ; CHECK: remark: <unknown>:0:0: in artificial function 'h', alloca ('%nodbg') without debug info with static size of 4 bytes
  %nodbg = alloca i32, align 4
  tail call void @llvm.dbg.declare(metadata ptr %dyn_ptr.addr, metadata !110, metadata !DIExpression()), !dbg !114
  tail call void @llvm.dbg.declare(metadata ptr %i, metadata !120, metadata !DIExpression()), !dbg !121
  tail call void @llvm.dbg.declare(metadata ptr %a, metadata !130, metadata !DIExpression()), !dbg !131
  tail call void @llvm.dbg.declare(metadata ptr %adyn, metadata !140, metadata !DIExpression()), !dbg !141
  br label %non-entry

non-entry:
  ; CHECK: remark: test.c:17:9: in artificial function 'h', alloca ('%i2') for 'i2' with static size of 4 bytes
  %i2 = alloca i32, align 4
  %size2 = load i32, ptr %i2, align 4
  ; CHECK: remark: test.c:18:9: in artificial function 'h', alloca ('%adyn2') for 'adyn2' with dynamic size
  %adyn2 = alloca i32, i32 %size, align 4
  tail call void @llvm.dbg.declare(metadata ptr %i2, metadata !150, metadata !DIExpression()), !dbg !151
  tail call void @llvm.dbg.declare(metadata ptr %adyn2, metadata !160, metadata !DIExpression()), !dbg !161
  ret void
}
; CHECK: remark: test.c:13:0: in artificial function 'h', Allocas = 7
; CHECK: remark: test.c:13:0: in artificial function 'h', AllocasStaticSizeSum = 28
; CHECK: remark: test.c:13:0: in artificial function 'h', AllocasDyn = 2

define void @g() !dbg !200 {
entry:
  ; CHECK: remark: test.c:4:7: in function 'g', alloca ('%i') for 'i' with static size of 4 bytes
  %i = alloca i32, align 4
  ; CHECK: remark: test.c:5:7: in function 'g', alloca ('%a') for 'a' with static size of 8 bytes
  %a = alloca [2 x i32], align 4
  tail call void @llvm.dbg.declare(metadata ptr %i, metadata !210, metadata !DIExpression()), !dbg !211
  tail call void @llvm.dbg.declare(metadata ptr %a, metadata !220, metadata !DIExpression()), !dbg !221
  ret void
}
; CHECK: remark: test.c:3:0: in function 'g', Allocas = 2
; CHECK: remark: test.c:3:0: in function 'g', AllocasStaticSizeSum = 12
; CHECK: remark: test.c:3:0: in function 'g', AllocasDyn = 0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

; uselistorder directives
uselistorder ptr @llvm.dbg.declare, { 7, 6, 5, 4, 3, 2, 1, 0 }

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C11, file: !2, producer: "clang version 19.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!2 = !DIFile(filename: "test.c", directory: "/tmp")
!3 = !{null}
!4 = !{}

!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!20 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !21)
!21 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !22)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)

!30 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 64, elements: !31)
!31 = !{!32}
!32 = !DISubrange(count: 2)

!40 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, elements: !41)
!41 = !{!42}
!42 = !DISubrange(count: !43)
!43 = !DILocalVariable(name: "__vla_expr0", scope: !100, type: !10, flags: DIFlagArtificial)

!100 = distinct !DISubprogram(name: "h", scope: !2, file: !2, line: 13, type: !101, scopeLine: 13, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !1, retainedNodes: !4)
!101 = distinct !DISubroutineType(types: !3)

!110 = !DILocalVariable(name: "dyn_ptr", arg: 1, scope: !100, type: !20, flags: DIFlagArtificial)
!114 = !DILocation(line: 0, scope: !100)

!120 = !DILocalVariable(name: "i", scope: !100, file: !2, line: 14, type: !10)
!121 = !DILocation(line: 14, column: 9, scope: !100)

!130 = !DILocalVariable(name: "a", scope: !100, file: !2, line: 15, type: !30)
!131 = !DILocation(line: 15, column: 9, scope: !100)

!140 = !DILocalVariable(name: "adyn", scope: !100, file: !2, line: 16, type: !40)
!141 = !DILocation(line: 16, column: 9, scope: !100)

!150 = !DILocalVariable(name: "i2", scope: !100, file: !2, line: 17, type: !10)
!151 = !DILocation(line: 17, column: 9, scope: !100)

!160 = !DILocalVariable(name: "adyn2", scope: !100, file: !2, line: 18, type: !40)
!161 = !DILocation(line: 18, column: 9, scope: !100)

!200 = distinct !DISubprogram(name: "g", scope: !2, file: !2, line: 3, type: !201, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !4)
!201 = !DISubroutineType(types: !3)

!210 = !DILocalVariable(name: "i", scope: !200, file: !2, line: 4, type: !10)
!211 = !DILocation(line: 4, column: 7, scope: !200)

!220 = !DILocalVariable(name: "a", scope: !200, file: !2, line: 5, type: !30)
!221 = !DILocation(line: 5, column: 7, scope: !200)
