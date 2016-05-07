; Test CFG simplify removal of branch instructions.
;
; RUN: opt < %s -simplifycfg -S | FileCheck %s
; RUN: opt < %s -passes=simplify-cfg -S | FileCheck %s

define void @test1() {
        br label %1
        ret void
; CHECK-LABEL: @test1(
; CHECK-NEXT: ret void
}

define void @test2() {
        ret void
        ret void
; CHECK-LABEL: @test2(
; CHECK-NEXT: ret void
; CHECK-NEXT: }
}

define void @test3(i1 %T) {
        br i1 %T, label %1, label %1
        ret void
; CHECK-LABEL: @test3(
; CHECK-NEXT: ret void
}

; Folding branch to a common destination.
; CHECK-LABEL: @test4_fold
; CHECK: %cmp1 = icmp eq i32 %a, %b
; CHECK: %cmp2 = icmp ugt i32 %a, 0
; CHECK: %or.cond = and i1 %cmp1, %cmp2
; CHECK: br i1 %or.cond, label %else, label %untaken
; CHECK-NOT: taken:
; CHECK: ret void
define void @test4_fold(i32 %a, i32 %b) {
  %cmp1 = icmp eq i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp ugt i32 %a, 0
  br i1 %cmp2, label %else, label %untaken

else:
  call void @foo()
  ret void

untaken:
  ret void
}

; Prefer a simplification based on a dominating condition rather than folding a
; branch to a common destination.
; CHECK-LABEL: @test4
; CHECK-NOT: br
; CHECK-NOT: br
; CHECK-NOT: call
; CHECK: ret void
define void @test4_no_fold(i32 %a, i32 %b) {
  %cmp1 = icmp eq i32 %a, %b
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp ugt i32 %a, %b
  br i1 %cmp2, label %else, label %untaken

else:
  call void @foo()
  ret void

untaken:
  ret void
}

declare void @foo()

; PR5795
define void @test5(i32 %A) {
  switch i32 %A, label %return [
    i32 2, label %1
    i32 10, label %2
  ]

  ret void

  ret void

return:                                           ; preds = %entry
  ret void
; CHECK-LABEL: @test5(
; CHECK-NEXT: ret void
}


; PR14893
define i8 @test6f() {
; CHECK-LABEL: @test6f
; CHECK: alloca i8, align 1
; CHECK-NEXT: call i8 @test6g
; CHECK-NEXT: icmp eq i8 %tmp, 0
; CHECK-NEXT: load i8, i8* %r, align 1, !dbg !{{[0-9]+$}}

bb0:
  %r = alloca i8, align 1
  %tmp = call i8 @test6g(i8* %r)
  %tmp1 = icmp eq i8 %tmp, 0
  br i1 %tmp1, label %bb2, label %bb1
bb1:
  %tmp3 = load i8, i8* %r, align 1, !range !2, !tbaa !1, !dbg !5
  %tmp4 = icmp eq i8 %tmp3, 1
  br i1 %tmp4, label %bb2, label %bb3
bb2:
  br label %bb3
bb3:
  %tmp6 = phi i8 [ 0, %bb2 ], [ 1, %bb1 ]
  ret i8 %tmp6
}
declare i8 @test6g(i8*)

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!8, !9}

!0 = !{!1, !1, i64 0}
!1 = !{!"foo"}
!2 = !{i8 0, i8 2}
!3 = distinct !DICompileUnit(language: DW_LANG_C99, file: !7, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !4)
!4 = !{}
!5 = !DILocation(line: 23, scope: !6)
!6 = distinct !DISubprogram(name: "foo", scope: !3, file: !7, line: 1, type: !DISubroutineType(types: !4), isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !3, variables: !4)
!7 = !DIFile(filename: "foo.c", directory: "/")
!8 = !{i32 2, !"Dwarf Version", i32 2}
!9 = !{i32 2, !"Debug Info Version", i32 3}
