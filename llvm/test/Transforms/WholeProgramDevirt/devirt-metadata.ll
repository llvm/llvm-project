; Test that the needed intrinsics for devirtualization are preserved and not dropped by other
; optimizations.

; RUN: opt -S -O3 %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@vt1 = constant [1 x ptr] [ptr @vf], !type !8
@vt2 = constant [1 x ptr] [ptr @vf2], !type !12

define i1 @vf(ptr %this) #0 !dbg !7 {
  ret i1 true
}

define i1 @vf2(ptr %this) !dbg !11 {
  ret i1 false
}

define void @call(ptr %obj) #1 !dbg !5 {
  %vtable = load ptr, ptr %obj
  ; CHECK: [[P:%[^ ]*]] = tail call i1 @llvm.public.type.test(ptr [[VT:%[^ ]*]], metadata !"typeid")
  ; CHECK-NEXT: call void @llvm.assume(i1 [[P]])
  %p = call i1 @llvm.public.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  call i1 %fptr(ptr %obj), !dbg !6
  ret void
}

define void @call1(ptr %obj) #1 !dbg !9 {
  %vtable = load ptr, ptr %obj
  ; CHECK: [[P:%[^ ]*]] = tail call i1 @llvm.type.test(ptr [[VT:%[^ ]*]], metadata !"typeid1")
  ; CHECK-NEXT: call void @llvm.assume(i1 [[P]])
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid1")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable, align 8
  %1 = call i1 %fptr(ptr %obj), !dbg !10
  ret void
}

declare i1 @llvm.type.test(ptr, metadata)
declare i1 @llvm.public.type.test(ptr, metadata)
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
!7 = distinct !DISubprogram(name: "vf", linkageName: "_ZN3vt12vfEb", scope: !1, file: !1, line: 13, isLocal: false, isDefinition: true, scopeLine: 13, flags: DIFlagPrototyped, isOptimized: false, unit: !0)
!8 = !{i32 0, !"typeid"}

!9 = distinct !DISubprogram(name: "call1", linkageName: "_Z5call1Pv", scope: !1, file: !1, line: 31, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: false, unit: !0)
!10 = !DILocation(line: 35, column: 32, scope: !9)
!11 = distinct !DISubprogram(name: "vf2", linkageName: "_ZN3vt13vf2Eb", scope: !1, file: !1, line: 23, isLocal: false, isDefinition: true, scopeLine: 23, flags: DIFlagPrototyped, isOptimized: false, unit: !0)
!12 = !{i32 0, !"typeid1"}

