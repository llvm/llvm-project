; RUN: opt -passes=instcombine -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; If all the operands to a phi node are of the same operation, instcombine
; will try to pull them through the phi node, combining them into a single
; operation. Check that when it does this the combined operation has a merged
; debug location.

; Test folding of a binary operation.  Generated from source:

; extern int foo(void);
; extern int bar(void);
; 
; int binop(int a, int b) {
;   if(a)
;     b -= foo();
;   else
;     b -= bar();
;   return b;
; }

; CHECK: define i32 @binop
; CHECK-LABEL: if.end:
; CHECK: %[[PHI:.*]] = phi i32 [ %call, %if.then ], [ %call1, %if.else ]
; CHECK: sub nsw i32 %b, %[[PHI]], !dbg [[binopMergedLoc:![0-9]+]]
; CHECK: ret i32

define i32 @binop(i32 %a, i32 %b) !dbg !6 {
entry:
  %tobool = icmp ne i32 %a, 0, !dbg !8
  br i1 %tobool, label %if.then, label %if.else, !dbg !8

if.then:                                          ; preds = %entry
  %call = call i32 @foo(), !dbg !9
  %sub = sub nsw i32 %b, %call, !dbg !10
  br label %if.end, !dbg !11

if.else:                                          ; preds = %entry
  %call1 = call i32 @bar(), !dbg !12
  %sub2 = sub nsw i32 %b, %call1, !dbg !13
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %b.addr.0 = phi i32 [ %sub, %if.then ], [ %sub2, %if.else ]
  ret i32 %b.addr.0, !dbg !14
}

; Test folding of a compare.  Generated from source (with editing to
; common the zext):

; extern int foo(void);
; extern int bar(void);
; 
; int cmp(int a, int b) {
;   int r;
;   if(a)
;     r = foo() < b;
;   else
;     r = bar() < b;
;   return r;
; }

; CHECK: define i32 @cmp
; CHECK-LABEL: if.end:
; CHECK: %[[PHI:.*]] = phi i32 [ %call, %if.then ], [ %call1, %if.else ]
; CHECK: icmp slt i32 %[[PHI]], %b, !dbg [[cmpMergedLoc:![0-9]+]]
; CHECK: ret i32

define i32 @cmp(i32 %a, i32 %b) !dbg !15 {
entry:
  %tobool = icmp ne i32 %a, 0, !dbg !16
  br i1 %tobool, label %if.then, label %if.else, !dbg !16

if.then:                                          ; preds = %entry
  %call = call i32 @foo(), !dbg !17
  %cmp = icmp slt i32 %call, %b, !dbg !18
  br label %if.end, !dbg !19

if.else:                                          ; preds = %entry
  %call1 = call i32 @bar(), !dbg !20
  %cmp2 = icmp slt i32 %call1, %b, !dbg !21
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %r.0 = phi i1 [ %cmp, %if.then ], [ %cmp2, %if.else ]
  %conv = zext i1 %r.0 to i32
  ret i32 %conv, !dbg !22
}

; Test folding of getelementptr.  Generated from source:

; extern long long foo2(void);
; extern long long bar2(void);
; 
; int *gep(int a, int *b) {
;   int *r;
;   if(a)
;     r = &b[foo2()];
;   else
;     r = &b[bar2()];
;   return p;
; }

; CHECK: define ptr @gep
; CHECK-LABEL: if.end:
; CHECK: %[[PHI:.*]] = phi i64 [ %call, %if.then ], [ %call1, %if.else ]
; CHECK: getelementptr inbounds i32, ptr %b, i64 %[[PHI]], !dbg [[gepMergedLoc:![0-9]+]]
; CHECK: ret ptr

define ptr @gep(i32 %a, ptr %b) !dbg !23 {
entry:
  %tobool = icmp ne i32 %a, 0, !dbg !24
  br i1 %tobool, label %if.then, label %if.else, !dbg !24

if.then:                                          ; preds = %entry
  %call = call i64 @foo2(), !dbg !25
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %call, !dbg !26
  br label %if.end, !dbg !27

if.else:                                          ; preds = %entry
  %call1 = call i64 @bar2(), !dbg !28
  %arrayidx2 = getelementptr inbounds i32, ptr %b, i64 %call1, !dbg !29
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %r.0 = phi ptr [ %arrayidx, %if.then ], [ %arrayidx2, %if.else ]
  ret ptr %r.0, !dbg !30
}

; Test folding of load.  Generated from source:

; extern int *foo3(void);
; extern int *bar3(void);
; 
; int load(int a) {
;   int r;
;   if(a)
;     r = *foo3();
;   else
;     r = *bar3();
;   return r;
; }

; CHECK: define i32 @load
; CHECK-LABEL: if.end:
; CHECK: %[[PHI:.*]] = phi ptr [ %call, %if.then ], [ %call1, %if.else ]
; CHECK: load i32, ptr %[[PHI]],{{.*}} !dbg [[loadMergedLoc:![0-9]+]]
; CHECK: ret i32

define i32 @load(i32 %a) !dbg !31 {
entry:
  %tobool = icmp ne i32 %a, 0, !dbg !32
  br i1 %tobool, label %if.then, label %if.else, !dbg !32

if.then:                                          ; preds = %entry
  %call = call ptr @foo3(), !dbg !33
  %0 = load i32, ptr %call, align 4, !dbg !34
  br label %if.end, !dbg !35

if.else:                                          ; preds = %entry
  %call1 = call ptr @bar3(), !dbg !36
  %1 = load i32, ptr %call1, align 4, !dbg !37
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %r.0 = phi i32 [ %0, %if.then ], [ %1, %if.else ]
  ret i32 %r.0, !dbg !38
}

; Test folding of a cast.  Generated from source:

; extern int foo(void);
; extern int bar(void);
; 
; long long cast(int a) {
;   long long r;
;   if(a)
;     r = foo();
;   else
;     r = bar();
;   return r;
; }

; CHECK: define i64 @cast
; CHECK-LABEL: if.end:
; CHECK: %[[PHI:.*]] = phi i32 [ %call, %if.then ], [ %call1, %if.else ]
; CHECK: sext i32 %[[PHI]] to i64, !dbg [[castMergedLoc:![0-9]+]]
; CHECK: ret i64

define i64 @cast(i32 %a) !dbg !39 {
entry:
  %tobool = icmp ne i32 %a, 0, !dbg !40
  br i1 %tobool, label %if.then, label %if.else, !dbg !40

if.then:                                          ; preds = %entry
  %call = call i32 @foo(), !dbg !41
  %conv = sext i32 %call to i64, !dbg !41
  br label %if.end, !dbg !42

if.else:                                          ; preds = %entry
  %call1 = call i32 @bar(), !dbg !43
  %conv2 = sext i32 %call1 to i64, !dbg !43
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %r.0 = phi i64 [ %conv, %if.then ], [ %conv2, %if.else ]
  ret i64 %r.0, !dbg !44
}

; Test folding of a binary op with a RHS constant.  Generated from source:

; extern int foo(void);
; extern int bar(void);
; 
; int binop_const(int a) {
;   int r;
;   if(a)
;     r = foo() - 5;
;   else
;     r = bar() - 5;
;   return r;
; }

; CHECK: define i32 @binop_const
; CHECK-LABEL: if.end:
; CHECK: %[[PHI:.*]] = phi i32 [ %call, %if.then ], [ %call1, %if.else ]
; CHECK: add nsw i32 %[[PHI]], -5, !dbg [[binopConstMergedLoc:![0-9]+]]
; CHECK: ret i32

define i32 @binop_const(i32 %a) !dbg !45 {
entry:
  %tobool = icmp ne i32 %a, 0, !dbg !46
  br i1 %tobool, label %if.then, label %if.else, !dbg !46

if.then:                                          ; preds = %entry
  %call = call i32 @foo(), !dbg !47
  %sub = sub nsw i32 %call, 5, !dbg !48
  br label %if.end, !dbg !49

if.else:                                          ; preds = %entry
  %call1 = call i32 @bar(), !dbg !50
  %sub2 = sub nsw i32 %call1, 5, !dbg !51
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %r.0 = phi i32 [ %sub, %if.then ], [ %sub2, %if.else ]
  ret i32 %r.0, !dbg !52
}

; Test folding of a compare with RHS constant.  Generated from source (with
; editing to common the zext):

; extern int foo(void);
; extern int bar(void);
; 
; int cmp_const(int a) {
;   int r;
;   if(a)
;     r = foo() < 10;
;   else
;     r = bar() < 10;
;   return r;
; }

; CHECK: define i32 @cmp_const
; CHECK-LABEL: if.end:
; CHECK: %[[PHI:.*]] = phi i32 [ %call, %if.then ], [ %call1, %if.else ]
; CHECK: icmp slt i32 %[[PHI]], 10, !dbg [[cmpConstMergedLoc:![0-9]+]]
; CHECK: ret i32

define i32 @cmp_const(i32 %a) !dbg !53 {
entry:
  %tobool = icmp ne i32 %a, 0, !dbg !54
  br i1 %tobool, label %if.then, label %if.else, !dbg !54

if.then:                                          ; preds = %entry
  %call = call i32 @foo(), !dbg !55
  %cmp = icmp slt i32 %call, 10, !dbg !56
  br label %if.end, !dbg !57

if.else:                                          ; preds = %entry
  %call1 = call i32 @bar(), !dbg !58
  %cmp2 = icmp slt i32 %call1, 10, !dbg !59
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %r.0 = phi i1 [ %cmp, %if.then ], [ %cmp2, %if.else ]
  %conv = zext i1 %r.0 to i32
  ret i32 %conv, !dbg !60
}

declare i32 @foo()
declare i32 @bar()
declare i64 @foo2()
declare i64 @bar2()
declare ptr @foo3()
declare ptr @bar3()

; CHECK: [[binopMergedLoc]] = !DILocation(line: 0
; CHECK: [[cmpMergedLoc]] = !DILocation(line: 0
; CHECK: [[gepMergedLoc]] = !DILocation(line: 0
; CHECK: [[loadMergedLoc]] = !DILocation(line: 0
; CHECK: [[castMergedLoc]] = !DILocation(line: 0
; CHECK: [[binopConstMergedLoc]] = !DILocation(line: 0
; CHECK: [[cmpConstMergedLoc]] = !DILocation(line: 0

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "binop", scope: !1, file: !1, line: 8, type: !7, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 9, column: 6, scope: !6)
!9 = !DILocation(line: 10, column: 10, scope: !6)
!10 = !DILocation(line: 10, column: 7, scope: !6)
!11 = !DILocation(line: 10, column: 5, scope: !6)
!12 = !DILocation(line: 12, column: 10, scope: !6)
!13 = !DILocation(line: 12, column: 7, scope: !6)
!14 = !DILocation(line: 13, column: 3, scope: !6)
!15 = distinct !DISubprogram(name: "cmp", scope: !1, file: !1, line: 16, type: !7, isLocal: false, isDefinition: true, scopeLine: 16, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!16 = !DILocation(line: 18, column: 6, scope: !15)
!17 = !DILocation(line: 19, column: 9, scope: !15)
!18 = !DILocation(line: 19, column: 15, scope: !15)
!19 = !DILocation(line: 19, column: 5, scope: !15)
!20 = !DILocation(line: 21, column: 9, scope: !15)
!21 = !DILocation(line: 21, column: 15, scope: !15)
!22 = !DILocation(line: 22, column: 3, scope: !15)
!23 = distinct !DISubprogram(name: "gep", scope: !1, file: !1, line: 25, type: !7, isLocal: false, isDefinition: true, scopeLine: 25, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!24 = !DILocation(line: 27, column: 6, scope: !23)
!25 = !DILocation(line: 28, column: 12, scope: !23)
!26 = !DILocation(line: 28, column: 10, scope: !23)
!27 = !DILocation(line: 28, column: 5, scope: !23)
!28 = !DILocation(line: 30, column: 12, scope: !23)
!29 = !DILocation(line: 30, column: 10, scope: !23)
!30 = !DILocation(line: 31, column: 3, scope: !23)
!31 = distinct !DISubprogram(name: "load", scope: !1, file: !1, line: 34, type: !7, isLocal: false, isDefinition: true, scopeLine: 34, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!32 = !DILocation(line: 36, column: 6, scope: !31)
!33 = !DILocation(line: 37, column: 10, scope: !31)
!34 = !DILocation(line: 37, column: 9, scope: !31)
!35 = !DILocation(line: 37, column: 5, scope: !31)
!36 = !DILocation(line: 39, column: 10, scope: !31)
!37 = !DILocation(line: 39, column: 9, scope: !31)
!38 = !DILocation(line: 40, column: 3, scope: !31)
!39 = distinct !DISubprogram(name: "cast", scope: !1, file: !1, line: 43, type: !7, isLocal: false, isDefinition: true, scopeLine: 43, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!40 = !DILocation(line: 45, column: 6, scope: !39)
!41 = !DILocation(line: 46, column: 9, scope: !39)
!42 = !DILocation(line: 46, column: 5, scope: !39)
!43 = !DILocation(line: 48, column: 9, scope: !39)
!44 = !DILocation(line: 49, column: 3, scope: !39)
!45 = distinct !DISubprogram(name: "binop_const", scope: !1, file: !1, line: 52, type: !7, isLocal: false, isDefinition: true, scopeLine: 52, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!46 = !DILocation(line: 54, column: 6, scope: !45)
!47 = !DILocation(line: 55, column: 9, scope: !45)
!48 = !DILocation(line: 55, column: 15, scope: !45)
!49 = !DILocation(line: 55, column: 5, scope: !45)
!50 = !DILocation(line: 57, column: 9, scope: !45)
!51 = !DILocation(line: 57, column: 15, scope: !45)
!52 = !DILocation(line: 58, column: 3, scope: !45)
!53 = distinct !DISubprogram(name: "cmp_const", scope: !1, file: !1, line: 61, type: !7, isLocal: false, isDefinition: true, scopeLine: 61, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!54 = !DILocation(line: 63, column: 6, scope: !53)
!55 = !DILocation(line: 64, column: 9, scope: !53)
!56 = !DILocation(line: 64, column: 15, scope: !53)
!57 = !DILocation(line: 64, column: 5, scope: !53)
!58 = !DILocation(line: 66, column: 9, scope: !53)
!59 = !DILocation(line: 66, column: 15, scope: !53)
!60 = !DILocation(line: 67, column: 3, scope: !53)
