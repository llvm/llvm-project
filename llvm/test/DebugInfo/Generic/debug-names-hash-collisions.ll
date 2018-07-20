; REQUIRES: object-emission
; RUN: %llc_dwarf -accel-tables=Dwarf -filetype=obj -o %t < %s
; RUN: llvm-dwarfdump -debug-names %t | FileCheck %s
; RUN: llvm-dwarfdump -debug-names -verify %t | FileCheck --check-prefix=VERIFY %s

; Generated from the following C code using
; clang -S -emit-llvm -g col.c
;
; These names were carefully chosen to cause hash collisions. Each type-variable
; pair will hash to the same value. The also happen to demonstrate a flaw in the
; DWARF v5 hash function: A copy constructor and an assignment operator for a
; class will always hash to the same value.
;
; typedef void *_ZN4lldb7SBBlockaSERKS0_;
; _ZN4lldb7SBBlockaSERKS0_ _ZN4lldb7SBBlockC1ERKS0_;
; typedef void *_ZN4lldb7SBErroraSERKS0_;
; _ZN4lldb7SBErroraSERKS0_ _ZN4lldb7SBErrorC1ERKS0_;
; typedef void *_ZN4lldb7SBValueaSERKS0_;
; _ZN4lldb7SBValueaSERKS0_ _ZN4lldb7SBValueC1ERKS0_;
; typedef void *_ZL11numCommutes;
; _ZL11numCommutes _ZL11NumCommutes;
; typedef void *_ZL9NumRemats;
; _ZL9NumRemats _ZL9NumReMats;

; Check that we have the right amount of hashes and names.
; CHECK: Bucket count: 5
; CHECK: Name count: 10

; Check that all the names are present in the output
; CHECK: Bucket 0
; CHECK:     Hash: 0xF8CF70D
; CHECK-NEXT:String: 0x{{[0-9a-f]*}} "_ZN4lldb7SBBlockaSERKS0_"
; CHECK:     Hash: 0xF8CF70D
; CHECK-NEXT:String: 0x{{[0-9a-f]*}} "_ZN4lldb7SBBlockC1ERKS0_"
; CHECK:     Hash: 0x135A482C
; CHECK-NEXT:String: 0x{{[0-9a-f]*}} "_ZN4lldb7SBErroraSERKS0_"
; CHECK:     Hash: 0x135A482C
; CHECK-NEXT:String: 0x{{[0-9a-f]*}} "_ZN4lldb7SBErrorC1ERKS0_"
; CHECK-NOT: String:
; CHECK: Bucket 1
; CHECK-NEXT: EMPTY
; CHECK: Bucket 2
; CHECK:     Hash: 0x2841B989
; CHECK-NEXT:String: 0x{{[0-9a-f]*}} "_ZL11numCommutes"
; CHECK:     Hash: 0x2841B989
; CHECK-NEXT:String: 0x{{[0-9a-f]*}} "_ZL11NumCommutes"
; CHECK:     Hash: 0x3E190F5F
; CHECK-NEXT:String: 0x{{[0-9a-f]*}} "_ZL9NumRemats"
; CHECK:     Hash: 0x3E190F5F
; CHECK-NEXT:String: 0x{{[0-9a-f]*}} "_ZL9NumReMats"
; CHECK-NOT: String:
; CHECK: Bucket 3
; CHECK:     Hash: 0x2642207F
; CHECK-NEXT:String: 0x{{[0-9a-f]*}} "_ZN4lldb7SBValueaSERKS0_"
; CHECK:     Hash: 0x2642207F
; CHECK-NEXT:String: 0x{{[0-9a-f]*}} "_ZN4lldb7SBValueC1ERKS0_"
; CHECK-NOT: String:
; CHECK:  Bucket 4
; CHECK-NEXT: EMPTY

; VERIFY: No errors.

@_ZN4lldb7SBBlockC1ERKS0_ = common dso_local global i8* null, align 8, !dbg !0
@_ZN4lldb7SBErrorC1ERKS0_ = common dso_local global i8* null, align 8, !dbg !6
@_ZN4lldb7SBValueC1ERKS0_ = common dso_local global i8* null, align 8, !dbg !10
@_ZL11NumCommutes = common dso_local global i8* null, align 8, !dbg !13
@_ZL9NumReMats = common dso_local global i8* null, align 8, !dbg !16

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!20, !21, !22}
!llvm.ident = !{!23}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "_ZN4lldb7SBBlockC1ERKS0_", scope: !2, file: !3, line: 1, type: !19, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "/tmp/col.c", directory: "/tmp")
!4 = !{}
!5 = !{!0, !6, !10, !13, !16}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "_ZN4lldb7SBErrorC1ERKS0_", scope: !2, file: !3, line: 2, type: !8, isLocal: false, isDefinition: true)
!8 = !DIDerivedType(tag: DW_TAG_typedef, name: "_ZN4lldb7SBErroraSERKS0_", file: !3, line: 2, baseType: !9)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "_ZN4lldb7SBValueC1ERKS0_", scope: !2, file: !3, line: 3, type: !12, isLocal: false, isDefinition: true)
!12 = !DIDerivedType(tag: DW_TAG_typedef, name: "_ZN4lldb7SBValueaSERKS0_", file: !3, line: 3, baseType: !9)
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression())
!14 = distinct !DIGlobalVariable(name: "_ZL11NumCommutes", scope: !2, file: !3, line: 4, type: !15, isLocal: false, isDefinition: true)
!15 = !DIDerivedType(tag: DW_TAG_typedef, name: "_ZL11numCommutes", file: !3, line: 4, baseType: !9)
!16 = !DIGlobalVariableExpression(var: !17, expr: !DIExpression())
!17 = distinct !DIGlobalVariable(name: "_ZL9NumReMats", scope: !2, file: !3, line: 5, type: !18, isLocal: false, isDefinition: true)
!18 = !DIDerivedType(tag: DW_TAG_typedef, name: "_ZL9NumRemats", file: !3, line: 5, baseType: !9)
!19 = !DIDerivedType(tag: DW_TAG_typedef, name: "_ZN4lldb7SBBlockaSERKS0_", file: !3, line: 1, baseType: !9)
!20 = !{i32 2, !"Dwarf Version", i32 4}
!21 = !{i32 2, !"Debug Info Version", i32 3}
!22 = !{i32 1, !"wchar_size", i32 4}
!23 = !{!"clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)"}
