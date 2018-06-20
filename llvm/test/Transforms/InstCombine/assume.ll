; RUN: opt < %s -instcombine -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @foo1(i32* %a) #0 {
entry:
  %0 = load i32, i32* %a, align 4

; Check that the alignment has been upgraded and that the assume has not
; been removed:
; CHECK-LABEL: @foo1
; CHECK-DAG: load i32, i32* %a, align 32
; CHECK-DAG: call void @llvm.assume
; CHECK: ret i32

  %ptrint = ptrtoint i32* %a to i64
  %maskedptr = and i64 %ptrint, 31
  %maskcond = icmp eq i64 %maskedptr, 0
  tail call void @llvm.assume(i1 %maskcond)

  ret i32 %0
}

define i32 @foo2(i32* %a) #0 {
entry:
; Same check as in @foo1, but make sure it works if the assume is first too.
; CHECK-LABEL: @foo2
; CHECK-DAG: load i32, i32* %a, align 32
; CHECK-DAG: call void @llvm.assume
; CHECK: ret i32

  %ptrint = ptrtoint i32* %a to i64
  %maskedptr = and i64 %ptrint, 31
  %maskcond = icmp eq i64 %maskedptr, 0
  tail call void @llvm.assume(i1 %maskcond)

  %0 = load i32, i32* %a, align 4
  ret i32 %0
}

declare void @llvm.assume(i1) #1

define i32 @simple(i32 %a) #1 {
entry:

; CHECK-LABEL: @simple
; CHECK: call void @llvm.assume
; CHECK: ret i32 4

  %cmp = icmp eq i32 %a, 4
  tail call void @llvm.assume(i1 %cmp)
  ret i32 %a
}

define i32 @can1(i1 %a, i1 %b, i1 %c) {
entry:
  %and1 = and i1 %a, %b
  %and  = and i1 %and1, %c
  tail call void @llvm.assume(i1 %and)

; CHECK-LABEL: @can1
; CHECK: call void @llvm.assume(i1 %a)
; CHECK: call void @llvm.assume(i1 %b)
; CHECK: call void @llvm.assume(i1 %c)
; CHECK: ret i32

  ret i32 5
}

define i32 @can2(i1 %a, i1 %b, i1 %c) {
entry:
  %v = or i1 %a, %b
  %w = xor i1 %v, 1
  tail call void @llvm.assume(i1 %w)

; CHECK-LABEL: @can2
; CHECK: %[[V1:[^ ]+]] = xor i1 %a, true
; CHECK: call void @llvm.assume(i1 %[[V1]])
; CHECK: %[[V2:[^ ]+]] = xor i1 %b, true
; CHECK: call void @llvm.assume(i1 %[[V2]])
; CHECK: ret i32

  ret i32 5
}

define i32 @bar1(i32 %a) #0 {
entry:
  %and1 = and i32 %a, 3

; CHECK-LABEL: @bar1
; CHECK: call void @llvm.assume
; CHECK: ret i32 1

  %and = and i32 %a, 7
  %cmp = icmp eq i32 %and, 1
  tail call void @llvm.assume(i1 %cmp)

  ret i32 %and1
}

define i32 @bar2(i32 %a) #0 {
entry:
; CHECK-LABEL: @bar2
; CHECK: call void @llvm.assume
; CHECK: ret i32 1

  %and = and i32 %a, 7
  %cmp = icmp eq i32 %and, 1
  tail call void @llvm.assume(i1 %cmp)

  %and1 = and i32 %a, 3
  ret i32 %and1
}

define i32 @bar3(i32 %a, i1 %x, i1 %y) #0 {
entry:
  %and1 = and i32 %a, 3

; Don't be fooled by other assumes around.
; CHECK-LABEL: @bar3
; CHECK: call void @llvm.assume
; CHECK: ret i32 1

  tail call void @llvm.assume(i1 %x)

  %and = and i32 %a, 7
  %cmp = icmp eq i32 %and, 1
  tail call void @llvm.assume(i1 %cmp)

  tail call void @llvm.assume(i1 %y)

  ret i32 %and1
}

define i32 @bar4(i32 %a, i32 %b) {
entry:
  %and1 = and i32 %b, 3

; CHECK-LABEL: @bar4
; CHECK: call void @llvm.assume
; CHECK: call void @llvm.assume
; CHECK: ret i32 1

  %and = and i32 %a, 7
  %cmp = icmp eq i32 %and, 1
  tail call void @llvm.assume(i1 %cmp)

  %cmp2 = icmp eq i32 %a, %b
  tail call void @llvm.assume(i1 %cmp2)

  ret i32 %and1
}

define i32 @icmp1(i32 %a) #0 {
; CHECK-LABEL: @icmp1(
; CHECK-NEXT:    [[CMP:%.*]] = icmp sgt i32 [[A:%.*]], 5
; CHECK-NEXT:    tail call void @llvm.assume(i1 [[CMP]])
; CHECK-NEXT:    ret i32 1
;
  %cmp = icmp sgt i32 %a, 5
  tail call void @llvm.assume(i1 %cmp)
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp2(i32 %a) #0 {
; CHECK-LABEL: @icmp2(
; CHECK-NEXT:    [[CMP:%.*]] = icmp sgt i32 [[A:%.*]], 5
; CHECK-NEXT:    tail call void @llvm.assume(i1 [[CMP]])
; CHECK-NEXT:    ret i32 0
;
  %cmp = icmp sgt i32 %a, 5
  tail call void @llvm.assume(i1 %cmp)
  %t0 = zext i1 %cmp to i32
  %lnot.ext = xor i32 %t0, 1
  ret i32 %lnot.ext
}

; If the 'not' of a condition is known true, then the condition must be false.

define i1 @assume_not(i1 %cond) {
; CHECK-LABEL: @assume_not(
; CHECK-NEXT:    [[NOTCOND:%.*]] = xor i1 [[COND:%.*]], true
; CHECK-NEXT:    call void @llvm.assume(i1 [[NOTCOND]])
; CHECK-NEXT:    ret i1 false
;
  %notcond = xor i1 %cond, true
  call void @llvm.assume(i1 %notcond)
  ret i1 %cond
}

declare void @escape(i32* %a)

; Canonicalize a nonnull assumption on a load into metadata form.

define i1 @nonnull1(i32** %a) {
; CHECK-LABEL: @nonnull1(
; CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** %a, align 8, !nonnull !6
; CHECK-NEXT:    tail call void @escape(i32* nonnull [[LOAD]])
; CHECK-NEXT:    ret i1 false
;
  %load = load i32*, i32** %a
  %cmp = icmp ne i32* %load, null
  tail call void @llvm.assume(i1 %cmp)
  tail call void @escape(i32* %load)
  %rval = icmp eq i32* %load, null
  ret i1 %rval
}

; Make sure the above canonicalization applies only
; to pointer types.  Doing otherwise would be illegal.

define i1 @nonnull2(i32* %a) {
; CHECK-LABEL: @nonnull2(
; CHECK-NEXT:    [[LOAD:%.*]] = load i32, i32* %a, align 4
; CHECK-NEXT:    [[CMP:%.*]] = icmp ne i32 [[LOAD]], 0
; CHECK-NEXT:    tail call void @llvm.assume(i1 [[CMP]])
; CHECK-NEXT:    [[RVAL:%.*]] = icmp eq i32 [[LOAD]], 0
; CHECK-NEXT:    ret i1 [[RVAL]]
;
  %load = load i32, i32* %a
  %cmp = icmp ne i32 %load, 0
  tail call void @llvm.assume(i1 %cmp)
  %rval = icmp eq i32 %load, 0
  ret i1 %rval
}

; Make sure the above canonicalization does not trigger
; if the assume is control dependent on something else

define i1 @nonnull3(i32** %a, i1 %control) {
; CHECK-LABEL: @nonnull3(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** %a, align 8
; CHECK-NEXT:    br i1 %control, label %taken, label %not_taken
; CHECK:       taken:
; CHECK-NEXT:    [[CMP:%.*]] = icmp ne i32* [[LOAD]], null
; CHECK-NEXT:    tail call void @llvm.assume(i1 [[CMP]])
; CHECK-NEXT:    [[RVAL:%.*]] = icmp eq i32* [[LOAD]], null
; CHECK-NEXT:    ret i1 [[RVAL]]
; CHECK:       not_taken:
; CHECK-NEXT:    ret i1 true
;
entry:
  %load = load i32*, i32** %a
  %cmp = icmp ne i32* %load, null
  br i1 %control, label %taken, label %not_taken
taken:
  tail call void @llvm.assume(i1 %cmp)
  %rval = icmp eq i32* %load, null
  ret i1 %rval
not_taken:
  ret i1 true
}

; Make sure the above canonicalization does not trigger
; if the path from the load to the assume is potentially
; interrupted by an exception being thrown

define i1 @nonnull4(i32** %a) {
; CHECK-LABEL: @nonnull4(
; CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** %a, align 8
; CHECK-NEXT:    tail call void @escape(i32* [[LOAD]])
; CHECK-NEXT:    [[CMP:%.*]] = icmp ne i32* [[LOAD]], null
; CHECK-NEXT:    tail call void @llvm.assume(i1 [[CMP]])
; CHECK-NEXT:    [[RVAL:%.*]] = icmp eq i32* [[LOAD]], null
; CHECK-NEXT:    ret i1 [[RVAL]]
;
  %load = load i32*, i32** %a
  ;; This call may throw!
  tail call void @escape(i32* %load)
  %cmp = icmp ne i32* %load, null
  tail call void @llvm.assume(i1 %cmp)
  %rval = icmp eq i32* %load, null
  ret i1 %rval
}

; PR35846 - https://bugs.llvm.org/show_bug.cgi?id=35846

define i32 @assumption_conflicts_with_known_bits(i32 %a, i32 %b) {
; CHECK-LABEL: @assumption_conflicts_with_known_bits(
; CHECK-NEXT:    tail call void @llvm.assume(i1 false)
; CHECK-NEXT:    ret i32 0
;
  %and1 = and i32 %b, 3
  %B1 = lshr i32 %and1, %and1
  %B3 = shl nuw nsw i32 %and1, %B1
  %cmp = icmp eq i32 %B3, 1
  tail call void @llvm.assume(i1 %cmp)
  %cmp2 = icmp eq i32 %B1, %B3
  tail call void @llvm.assume(i1 %cmp2)
  ret i32 %and1
}

; FIXME:
; PR37726 - https://bugs.llvm.org/show_bug.cgi?id=37726
; There's a loophole in eliminating a redundant assumption when
; we have conflicting assumptions. Verify that debuginfo doesn't
; get in the way of the fold.

define void @debug_interference(i8 %x) {
; CHECK-LABEL: @debug_interference(
; CHECK-NEXT:    [[CMP1:%.*]] = icmp eq i8 [[X:%.*]], 0
; CHECK-NEXT:    tail call void @llvm.dbg.value(metadata i32 5, metadata !7, metadata !DIExpression()), !dbg !9
; CHECK-NEXT:    tail call void @llvm.assume(i1 [[CMP1]])
; CHECK-NEXT:    tail call void @llvm.dbg.value(metadata i32 5, metadata !7, metadata !DIExpression()), !dbg !9
; CHECK-NEXT:    tail call void @llvm.dbg.value(metadata i32 5, metadata !7, metadata !DIExpression()), !dbg !9
; CHECK-NEXT:    tail call void @llvm.assume(i1 false)
; CHECK-NEXT:    ret void
;
  %cmp1 = icmp eq i8 %x, 0
  %cmp2 = icmp ne i8 %x, 0
  tail call void @llvm.assume(i1 %cmp1)
  tail call void @llvm.dbg.value(metadata i32 5, metadata !1, metadata !DIExpression()), !dbg !9
  tail call void @llvm.assume(i1 %cmp1)
  tail call void @llvm.dbg.value(metadata i32 5, metadata !1, metadata !DIExpression()), !dbg !9
  tail call void @llvm.assume(i1 %cmp2)
  tail call void @llvm.dbg.value(metadata i32 5, metadata !1, metadata !DIExpression()), !dbg !9
  tail call void @llvm.assume(i1 %cmp2)
  ret void
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6, !7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "Me", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: null, retainedTypes: null, imports: null)
!1 = !DILocalVariable(name: "", arg: 1, scope: !2, file: null, line: 1, type: null)
!2 = distinct !DISubprogram(name: "debug", linkageName: "debug", scope: null, file: null, line: 0, type: null, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0)
!3 = !DIFile(filename: "consecutive-fences.ll", directory: "")
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{i32 7, !"PIC Level", i32 2}
!9 = !DILocation(line: 0, column: 0, scope: !2)


attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind }

