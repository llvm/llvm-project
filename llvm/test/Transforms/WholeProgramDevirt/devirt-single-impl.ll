; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility -pass-remarks=wholeprogramdevirt -stats %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: remark: devirt-single.cc:30:32: single-impl: devirtualized a call to vf
; CHECK: remark: devirt-single.cc:41:32: single-impl: devirtualized a call to vf
; CHECK: remark: devirt-single.cc:51:32: single-impl: devirtualized a call to vf
; CHECK: remark: devirt-single.cc:13:0: devirtualized vf
; CHECK-NOT: devirtualized

@vt1 = constant [1 x ptr] [ptr @vf], !type !8
@vt2 = constant [1 x ptr] [ptr @vf], !type !8

define void @vf(ptr %this) #0 !dbg !7 {
  ret void
}

; CHECK: define void @call
define void @call(ptr %obj) #1 !dbg !5 {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  ; CHECK: call void @vf(
  call void %fptr(ptr %obj), !dbg !6
  ret void
}

declare ptr @llvm.load.relative.i32(ptr, i32)

@vt3 = private unnamed_addr constant [1 x i32] [
  i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf to i64), i64 ptrtoint (ptr @vt3 to i64)) to i32)
], align 4, !type !11

; CHECK: define void @call2
define void @call2(ptr %obj) #1 !dbg !9 {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid2")
  call void @llvm.assume(i1 %p)
  %fptr = call ptr @llvm.load.relative.i32(ptr %vtable, i32 0)
  ; CHECK: call void @vf(
  call void %fptr(ptr %obj), !dbg !10
  ret void
}

@_ZTV1A.local = private unnamed_addr constant { [3 x i32] } { [3 x i32] [
  i32 0,  ; offset to top
  i32 0,  ; rtti
  i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [3 x i32] }, ptr @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32)  ; vfunc offset
] }, align 4, !type !14

; CHECK: define void @call3
define void @call3(ptr %obj) #1 !dbg !12 {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid3")
  call void @llvm.assume(i1 %p)
  %fptr = call ptr @llvm.load.relative.i32(ptr %vtable, i32 8)
  ; CHECK: call void @vf(
  call void %fptr(ptr %obj), !dbg !13
  ret void
}


declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 4.0.0 (trunk 278098)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "devirt-single.cc", directory: ".")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{!"clang version 4.0.0 (trunk 278098)"}
!5 = distinct !DISubprogram(name: "call", linkageName: "_Z4callPv", scope: !1, file: !1, line: 29, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: false, unit: !0)
!6 = !DILocation(line: 30, column: 32, scope: !5)
!7 = distinct !DISubprogram(name: "vf", linkageName: "_ZN3vt12vfEv", scope: !1, file: !1, line: 13, isLocal: false, isDefinition: true, scopeLine: 13, flags: DIFlagPrototyped, isOptimized: false, unit: !0)
!8 = !{i32 0, !"typeid"}

!9 = distinct !DISubprogram(name: "call2", linkageName: "_Z5call2Pv", scope: !1, file: !1, line: 40, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: false, unit: !0)
!10 = !DILocation(line: 41, column: 32, scope: !9)
!11 = !{i32 0, !"typeid2"}

!12 = distinct !DISubprogram(name: "call3", linkageName: "_Z5call3Pv", scope: !1, file: !1, line: 50, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: false, unit: !0)
!13 = !DILocation(line: 51, column: 32, scope: !12)
!14 = !{i32 0, !"typeid3"}

; CHECK: 1 wholeprogramdevirt - Number of whole program devirtualization targets
; CHECK: 3 wholeprogramdevirt - Number of single implementation devirtualizations
