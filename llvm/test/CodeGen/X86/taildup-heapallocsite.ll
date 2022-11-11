; RUN: llc < %s -tail-dup-placement-threshold=4 | FileCheck %s

; Based on test case from PR43695:
; __declspec(allocator) ptr alloc(unsigned int size);
; void f2();
; void f1(unsigned int *size_ptr) {
;     ptr hg = alloc(size_ptr ? *size_ptr : 1UL);
;     f2();
; }

; In this case, block placement duplicates the heap allocation site.

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.22.27905"

define dso_local void @taildupit(ptr readonly %size_ptr) !dbg !8 {
entry:
  call void @llvm.dbg.value(metadata ptr %size_ptr, metadata !14, metadata !DIExpression()), !dbg !17
  %tobool = icmp eq ptr %size_ptr, null, !dbg !18
  br i1 %tobool, label %cond.end, label %cond.true, !dbg !18

cond.true:                                        ; preds = %entry
  %0 = load i32, ptr %size_ptr, align 4, !dbg !18, !tbaa !19
  br label %cond.end, !dbg !18

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i32 [ %0, %cond.true ], [ 1, %entry ], !dbg !18
  %call = tail call ptr @alloc(i32 %cond), !dbg !18, !heapallocsite !2
  call void @llvm.dbg.value(metadata ptr %call, metadata !15, metadata !DIExpression()), !dbg !17
  tail call void @f2(), !dbg !23
  ret void, !dbg !24
}

; CHECK-LABEL: taildupit: # @taildupit
; CHECK: testq
; CHECK: je
; CHECK: callq alloc
; CHECK-NEXT: [[L1:.Ltmp[0-9]+]]
; CHECK: jmp f2 # TAILCALL
; CHECK: callq alloc
; CHECK-NEXT: [[L3:.Ltmp[0-9]+]]
; CHECK: jmp f2 # TAILCALL

; CHECK-LABEL: .short 4423                    # Record kind: S_GPROC32_ID
; CHECK:       .short 4446                    # Record kind: S_HEAPALLOCSITE
; CHECK-NEXT:  .secrel32 [[L0:.Ltmp[0-9]+]]
; CHECK-NEXT:  .secidx [[L0]]
; CHECK-NEXT:  .short [[L1]]-[[L0]]
; CHECK-NEXT:  .long 3
; CHECK:       .short 4446                    # Record kind: S_HEAPALLOCSITE
; CHECK-NEXT:  .secrel32 [[L2:.Ltmp[0-9]+]]
; CHECK-NEXT:  .secidx [[L2]]
; CHECK-NEXT:  .short [[L3]]-[[L2]]
; CHECK-NEXT:  .long 3

declare dso_local ptr @alloc(i32)

declare dso_local void @f2()

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 10.0.0 (git@github.com:llvm/llvm-project.git 0650355c09ab8e6605ae37b818270a7a7c8ce2c7)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "t.cpp", directory: "C:\\src\\llvm-project\\build", checksumkind: CSK_MD5, checksum: "b227901e92d848fa564190b0762d757c")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!8 = distinct !DISubprogram(name: "f1", linkageName: "?f1@@YAXPEAI@Z", scope: !1, file: !1, line: 5, type: !9, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!13 = !{!14, !15}
!14 = !DILocalVariable(name: "size_ptr", arg: 1, scope: !8, file: !1, line: 5, type: !11)
!15 = !DILocalVariable(name: "hg", scope: !8, file: !1, line: 6, type: !16)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!17 = !DILocation(line: 0, scope: !8)
!18 = !DILocation(line: 6, scope: !8)
!19 = !{!20, !20, i64 0}
!20 = !{!"int", !21, i64 0}
!21 = !{!"omnipotent char", !22, i64 0}
!22 = !{!"Simple C++ TBAA"}
!23 = !DILocation(line: 7, scope: !8)
!24 = !DILocation(line: 8, scope: !8)
