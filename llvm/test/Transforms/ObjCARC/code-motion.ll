; RUN: opt -objc-arc -S < %s | FileCheck %s

declare void @alterRefCount()
declare void @use(i8*)

@g0 = global i8* null, align 8

; Check that ARC optimizer doesn't reverse the order of the retain call and the
; release call when there are debug instructions.

; CHECK: call i8* @llvm.objc.retain(i8* %x)
; CHECK: call void @llvm.objc.release(i8* %x)

define i32 @test(i8* %x, i8* %y, i8 %z, i32 %i) {
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %v1 = tail call i8* @llvm.objc.retain(i8* %x)
  store i8 %z, i8* %x
  call void @llvm.dbg.declare(metadata i32* %i.addr, metadata !9, metadata !DIExpression()), !dbg !10
  call void @alterRefCount()
  tail call void @llvm.objc.release(i8* %x)
  ret i32 %i
}

; ARC optimizer shouldn't move the release call, which is a precise release call
; past the call to @alterRefCount.

; CHECK-LABEL: define void @test2(
; CHECK: call void @alterRefCount(
; CHECK: call void @llvm.objc.release(

define void @test2() {
  %v0 = load i8*, i8** @g0, align 8
  %v1 = tail call i8* @llvm.objc.retain(i8* %v0)
  tail call void @use(i8* %v0)
  tail call void @alterRefCount()
  tail call void @llvm.objc.release(i8* %v0)
  ret void
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare i8* @llvm.objc.retain(i8*) local_unnamed_addr
declare void @llvm.objc.release(i8*) local_unnamed_addr

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !DILocalVariable(name: "i", arg: 1, scope: !3, file: !4, line: 1, type: !7)
!3 = distinct !DISubprogram(name: "test", scope: !4, file: !4, line: 1, type: !5, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !8, retainedNodes: !9)
!4 = !DIFile(filename: "test.m", directory: "dir")
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = distinct !DICompileUnit(language: DW_LANG_ObjC, file: !4, isOptimized: false, runtimeVersion: 2, emissionKind: FullDebug, enums: !9, nameTableKind: None)
!9 = !{}
!10 = !DILocation(line: 1, column: 14, scope: !3)
