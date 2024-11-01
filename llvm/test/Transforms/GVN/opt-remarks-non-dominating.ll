; RUN: opt < %s -passes=gvn -o /dev/null  -pass-remarks-output=%t -S
; RUN: cat %t | FileCheck %s


; ModuleID = 'bugpoint-reduced-simplified.bc'
source_filename = "gvn-test.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            gvn
; CHECK-NEXT: Name:            LoadClobbered
; CHECK-NEXT: Function:        nonDominating1
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'load of type '
; CHECK-NEXT:   - Type:            i32
; CHECK-NEXT:   - String:          ' not eliminated'
; CHECK-NEXT:   - String:          ' in favor of '
; CHECK-NEXT:   - OtherAccess:     store
; CHECK-NEXT:   - String:          ' because it is clobbered by '
; CHECK-NEXT:   - ClobberedBy:     call
; CHECK-NEXT: ...

; Confirm that the partial redundancy being clobbered by the call to
; clobberingFunc() between store and load is identified.

define dso_local void @nonDominating1(i32* %a, i1 %cond, i32 %b) local_unnamed_addr #0 {
entry:
  br i1 %cond, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 %b, i32* %a, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  tail call void @clobberingFunc() #1
  %0 = load i32, i32* %a, align 4
  %mul2 = shl nsw i32 %0, 1
  store i32 %mul2, i32* %a, align 4
  ret void
}

declare dso_local void @clobberingFunc() local_unnamed_addr #0

; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            gvn
; CHECK-NEXT: Name:            LoadClobbered
; CHECK-NEXT: DebugLoc:        { File: '/tmp/s.c', Line: 3, Column: 3 }
; CHECK-NEXT: Function:        nonDominating2
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'load of type '
; CHECK-NEXT:   - Type:            i32
; CHECK-NEXT:   - String:          ' not eliminated'
; CHECK-NEXT:   - String:          ' in favor of '
; CHECK-NEXT:   - OtherAccess:     load
; CHECK-NEXT:     DebugLoc: { File: '/tmp/s.c', Line: 1, Column: 1 }
; CHECK-NEXT:   - String:          ' because it is clobbered by '
; CHECK-NEXT:   - ClobberedBy:     call
; CHECK-NEXT:     DebugLoc: { File: '/tmp/s.c', Line: 2, Column: 2 }
; CHECK-NEXT: ...
; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            gvn
; CHECK-NEXT: Name:            LoadClobbered
; CHECK-NEXT: DebugLoc:        { File: '/tmp/s.c', Line: 5, Column: 5 }
; CHECK-NEXT: Function:        nonDominating2
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'load of type '
; CHECK-NEXT:   - Type:            i32
; CHECK-NEXT:   - String:          ' not eliminated'
; CHECK-NEXT:   - String:          ' in favor of '
; CHECK-NEXT:   - OtherAccess:     load
; CHECK-NEXT:     DebugLoc: { File: '/tmp/s.c', Line: 3, Column: 3 }
; CHECK-NEXT:   - String:          ' because it is clobbered by '
; CHECK-NEXT:   - ClobberedBy:     call
; CHECK-NEXT:     DebugLoc: { File: '/tmp/s.c', Line: 4, Column: 4 }
; CHECK-NEXT: ...

; More complex version of nonDominating1(), this time with loads. The values
; already loaded into %0 and %1 cannot replace %2 due to clobbering calls.
; %1 is not clobbered by the first call however, and %0 is irrelevant for the
; second one since %1 is more recently available.

define dso_local void @nonDominating2(i32* %a, i1 %cond) local_unnamed_addr #0 {
entry:
  br i1 %cond, label %if.then, label %if.end5

if.then:                                          ; preds = %entry
  %0 = load i32, i32* %a, align 4, !dbg !14
  %mul = mul nsw i32 %0, 10
  tail call void @clobberingFunc() #1, !dbg !15
  %1 = load i32, i32* %a, align 4, !dbg !16
  %mul3 = mul nsw i32 %1, 5
  tail call void @clobberingFunc() #1, !dbg !17
  br label %if.end5

if.end5:                                          ; preds = %if.then, %entry
  %2 = load i32, i32* %a, align 4, !dbg !18
  %mul9 = shl nsw i32 %2, 1
  store i32 %mul9, i32* %a, align 4
  ret void
}

; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            gvn
; CHECK-NEXT: Name:            LoadClobbered
; CHECK-NEXT: Function:        nonDominating3
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'load of type '
; CHECK-NEXT:   - Type:            i32
; CHECK-NEXT:   - String:          ' not eliminated'
; CHECK-NEXT:   - String:          ' because it is clobbered by '
; CHECK-NEXT:   - ClobberedBy:     call
; CHECK-NEXT: ...

; The two stores are both partially available at %0 (were it not for the
; clobbering call), however neither is strictly more recent than the other, so
; no attempt is made to identify what value could have potentially been reused
; otherwise. Just report that the load cannot be eliminated.

define dso_local void @nonDominating3(i32* %a, i32 %b, i32 %c, i1 %cond) local_unnamed_addr #0 {
entry:
  br i1 %cond, label %if.end5.sink.split, label %if.else

if.else:                                          ; preds = %entry
  store i32 %b, i32* %a, align 4
  br label %if.end5

if.end5.sink.split:                               ; preds = %entry
  store i32 %c, i32* %a, align 4
  br label %if.end5

if.end5:                                          ; preds = %if.end5.sink.split, %if.else
  tail call void @clobberingFunc() #1
  %0 = load i32, i32* %a, align 4
  %mul7 = shl nsw i32 %0, 1
  ret void
}

; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            gvn
; CHECK-NEXT: Name:            LoadClobbered
; CHECK-NEXT: Function:        nonDominating4
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'load of type '
; CHECK-NEXT:   - Type:            i32
; CHECK-NEXT:   - String:          ' not eliminated'
; CHECK-NEXT:   - String:          ' in favor of '
; CHECK-NEXT:   - OtherAccess:     store
; CHECK-NEXT:   - String:          ' because it is clobbered by '
; CHECK-NEXT:   - ClobberedBy:     call
; CHECK-NEXT: ...

; Make sure isPotentiallyReachable() is not called for an instruction
; outside the current function, as it will cause a crash.

define dso_local void @nonDominating4(i1 %cond, i32 %b) local_unnamed_addr #0 {
entry:
  br i1 %cond, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 %b, i32* @g, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  tail call void @clobberingFunc() #1
  %0 = load i32, i32* @g, align 4
  %mul2 = shl nsw i32 %0, 1
  store i32 %mul2, i32* @g, align 4
  ret void
}

@g = external global i32

define dso_local void @globalUser(i32 %b) local_unnamed_addr #0 {
entry:
  store i32 %b, i32* @g, align 4
  ret void
}

attributes #0 = { "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.ident = !{!0}
!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!4, !5, !6}

!0 = !{!"clang version 10.0.0 (git@github.com:llvm/llvm-project.git a2f6ae9abffcba260c22bb235879f0576bf3b783)"}

!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 4.0.0 (trunk 282540) (llvm/trunk 282542)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !3)
!2 = !DIFile(filename: "/tmp/s.c", directory: "/tmp")
!3 = !{}
!4 = !{i32 2, !"Dwarf Version", i32 4}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"PIC Level", i32 2}
!8 = distinct !DISubprogram(name: "nonDominating2", scope: !2, file: !2, line: 1, type: !9, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !1, retainedNodes: !3)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11, !12, !12, !13}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DIBasicType(name: "_Bool", size: 8, encoding: DW_ATE_boolean)
!14 = !DILocation(line: 1, column: 1, scope: !8)
!15 = !DILocation(line: 2, column: 2, scope: !8)
!16 = !DILocation(line: 3, column: 3, scope: !8)
!17 = !DILocation(line: 4, column: 4, scope: !8)
!18 = !DILocation(line: 5, column: 5, scope: !8)
