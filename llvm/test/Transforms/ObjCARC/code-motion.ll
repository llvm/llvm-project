; RUN: opt -objc-arc -S < %s | FileCheck %s

declare void @alterRefCount()
declare void @use(i8*)
declare void @readOnlyFunc(i8*, i8*)

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

; Check that code motion is disabled in @test3 and @test4.
; Previously, ARC optimizer would move the release past the retain.

; if.then:
;   call void @readOnlyFunc(i8* %obj, i8* null)
;   call void @llvm.objc.release(i8* %obj) #1, !clang.imprecise_release !2
;   %1 = add i32 1, 2
;   %2 = tail call i8* @llvm.objc.retain(i8* %obj)
;
; Ideally, the retain/release pairs in BB if.then should be removed.

define void @test3(i8* %obj, i1 %cond) {
; CHECK-LABEL: @test3(
; CHECK-NEXT:    [[TMP2:%.*]] = tail call i8* @llvm.objc.retain(i8* [[OBJ:%.*]])
; CHECK-NEXT:    br i1 [[COND:%.*]], label [[IF_THEN:%.*]], label [[IF_ELSE:%.*]]
; CHECK:       if.then:
; CHECK-NEXT:    call void @readOnlyFunc(i8* [[OBJ]], i8* null)
; CHECK-NEXT:    [[TMP1:%.*]] = add i32 1, 2
; CHECK-NEXT:    call void @alterRefCount()
; CHECK-NEXT:    br label [[JOIN:%.*]]
; CHECK:       if.else:
; CHECK-NEXT:    call void @alterRefCount()
; CHECK-NEXT:    call void @use(i8* [[OBJ]])
; CHECK-NEXT:    br label [[JOIN]]
; CHECK:       join:
; CHECK-NEXT:    call void @llvm.objc.release(i8* [[OBJ]]) {{.*}}, !clang.imprecise_release !2
; CHECK-NEXT:    ret void
;
  %v0 = call i8* @llvm.objc.retain(i8* %obj)
  br i1 %cond, label %if.then, label %if.else

if.then:
  call void @readOnlyFunc(i8* %obj, i8* null) #0
  add i32 1, 2
  call void @alterRefCount()
  br label %join

if.else:
  call void @alterRefCount()
  call void @use(i8* %obj)
  br label %join

join:
  call void @llvm.objc.release(i8* %obj), !clang.imprecise_release !9
  ret void
}

define void @test4(i8* %obj0, i8* %obj1, i1 %cond) {
; CHECK-LABEL: @test4(
; CHECK-NEXT:    [[TMP3:%.*]] = tail call i8* @llvm.objc.retain(i8* [[OBJ0:%.*]])
; CHECK-NEXT:    [[TMP2:%.*]] = tail call i8* @llvm.objc.retain(i8* [[OBJ1:%.*]])
; CHECK-NEXT:    br i1 [[COND:%.*]], label [[IF_THEN:%.*]], label [[IF_ELSE:%.*]]
; CHECK:       if.then:
; CHECK-NEXT:    call void @readOnlyFunc(i8* [[OBJ0]], i8* [[OBJ1]])
; CHECK-NEXT:    [[TMP1:%.*]] = add i32 1, 2
; CHECK-NEXT:    call void @alterRefCount()
; CHECK-NEXT:    br label [[JOIN:%.*]]
; CHECK:       if.else:
; CHECK-NEXT:    call void @alterRefCount()
; CHECK-NEXT:    call void @use(i8* [[OBJ0]])
; CHECK-NEXT:    call void @use(i8* [[OBJ1]])
; CHECK-NEXT:    br label [[JOIN]]
; CHECK:       join:
; CHECK-NEXT:    call void @llvm.objc.release(i8* [[OBJ0]]) {{.*}}, !clang.imprecise_release !2
; CHECK-NEXT:    call void @llvm.objc.release(i8* [[OBJ1]]) {{.*}}, !clang.imprecise_release !2
; CHECK-NEXT:    ret void
;
  %v0 = call i8* @llvm.objc.retain(i8* %obj0)
  %v1 = call i8* @llvm.objc.retain(i8* %obj1)
  br i1 %cond, label %if.then, label %if.else

if.then:
  call void @readOnlyFunc(i8* %obj0, i8* %obj1) #0
  add i32 1, 2
  call void @alterRefCount()
  br label %join

if.else:
  call void @alterRefCount()
  call void @use(i8* %obj0)
  call void @use(i8* %obj1)
  br label %join

join:
  call void @llvm.objc.release(i8* %obj0), !clang.imprecise_release !9
  call void @llvm.objc.release(i8* %obj1), !clang.imprecise_release !9
  ret void
}

; In this test, insertion points for the retain and release calls that could be
; eliminated are in different blocks (bb1 and if.then).

define void @test5(i8* %obj, i1 %cond0, i1 %cond1) {
; CHECK-LABEL: @test5(
; CHECK-NEXT:    [[V0:%.*]] = tail call i8* @llvm.objc.retain(i8* [[OBJ:%.*]])
; CHECK-NEXT:    br i1 [[COND0:%.*]], label [[IF_THEN:%.*]], label [[IF_ELSE:%.*]]
; CHECK:       if.then:
; CHECK-NEXT:    call void @readOnlyFunc(i8* [[OBJ]], i8* null)
; CHECK-NEXT:    br i1 [[COND1:%.*]], label [[IF_THEN2:%.*]], label [[IF_ELSE2:%.*]]
; CHECK:       if.then2:
; CHECK-NEXT:    br label [[BB1:%.*]]
; CHECK:       if.else2:
; CHECK-NEXT:    br label [[BB1]]
; CHECK:       bb1:
; CHECK-NEXT:    [[TMP1:%.*]] = add i32 1, 2
; CHECK-NEXT:    call void @alterRefCount()
; CHECK-NEXT:    br label [[JOIN:%.*]]
; CHECK:       if.else:
; CHECK-NEXT:    call void @alterRefCount()
; CHECK-NEXT:    call void @use(i8* [[OBJ]])
; CHECK-NEXT:    br label [[JOIN]]
; CHECK:       join:
; CHECK-NEXT:    call void @llvm.objc.release(i8* [[OBJ]])
; CHECK-NEXT:    ret void
;
  %v0 = call i8* @llvm.objc.retain(i8* %obj)
  br i1 %cond0, label %if.then, label %if.else

if.then:
  call void @readOnlyFunc(i8* %obj, i8* null) #0
  br i1 %cond1, label %if.then2, label %if.else2

if.then2:
  br label %bb1

if.else2:
  br label %bb1

bb1:
  add i32 1, 2
  call void @alterRefCount()
  br label %join

if.else:
  call void @alterRefCount()
  call void @use(i8* %obj)
  br label %join

join:
  call void @llvm.objc.release(i8* %obj), !clang.imprecise_release !9
  ret void
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare i8* @llvm.objc.retain(i8*) local_unnamed_addr
declare void @llvm.objc.release(i8*) local_unnamed_addr

attributes #0 = { readonly }

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
