; Choosing CodeView generates debug metadata for class-scope typedefs that
; Dwarf would normally omit.  Choosing both CodeView and Dwarf triggered
; assertion failures and crashes because the Dwarf handler wasn't prepared for
; those records (in particular, ones with the void type represented by a
; null pointer).
;
; This test was generated with:
;    clang++ -cc1 -emit-llvm -debug-info-kind=limited -dwarf-version=4 -gcodeview -x c++
; on the following source code:
;
;   class A {
;     typedef void _Nodeptr;
;   };
;   class B {
;     A FailedTestsCache;
;     bool m_fn1();
;   };
;   bool B::m_fn1() {}
;
; CodeView generates a DIDerivedType for the _Nodeptr typedef.
;
; RUN: llc %s -o - 2>&1 | FileCheck %s
; CHECK-NOT: Assertion failed

; ModuleID = 'bug.cpp'
source_filename = "bug.cpp"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc"

%class.B = type { %class.A }
%class.A = type { i8 }

; Function Attrs: noinline nounwind optnone
define x86_thiscallcc zeroext i1 @"\01?m_fn1@B@@AAE_NXZ"(ptr %this) #0 align 2 !dbg !9 {
entry:
  %retval = alloca i1, align 1
  %this.addr = alloca ptr, align 4
  store ptr %this, ptr %this.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %this.addr, metadata !22, metadata !DIExpression()), !dbg !24
  %this1 = load ptr, ptr %this.addr, align 4
  call void @llvm.trap(), !dbg !25
  unreachable, !dbg !25

return:                                           ; No predecessors!
  %0 = load i1, ptr %retval, align 1, !dbg !25
  ret i1 %0, !dbg !25
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noreturn nounwind
declare void @llvm.trap() #2

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { noreturn nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 6.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "<stdin>", directory: "D:\5Csrc\5Cbug", checksumkind: CSK_MD5, checksum: "2216f11c5ddda8c48a6f92a6079ad4b6")
!2 = !{}
!3 = !{i32 1, !"NumRegisterParameters", i32 0}
!4 = !{i32 2, !"Dwarf Version", i32 4}
!5 = !{i32 2, !"CodeView", i32 1}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 2}
!8 = !{!"clang version 6.0.0 "}
!9 = distinct !DISubprogram(name: "m_fn1", linkageName: "\01?m_fn1@B@@AAE_NXZ", scope: !11, file: !10, line: 8, type: !18, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !17, retainedNodes: !2)
!10 = !DIFile(filename: "bug.cpp", directory: "D:\5Csrc\5Cbug", checksumkind: CSK_MD5, checksum: "2216f11c5ddda8c48a6f92a6079ad4b6")
!11 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "B", file: !10, line: 4, size: 8, elements: !12, identifier: ".?AVB@@")
!12 = !{!13, !17}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "FailedTestsCache", scope: !11, file: !10, line: 5, baseType: !14, size: 8)
!14 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "A", file: !10, line: 1, size: 8, elements: !15, identifier: ".?AVA@@")
!15 = !{!16}
!16 = !DIDerivedType(tag: DW_TAG_typedef, name: "_Nodeptr", scope: !14, file: !10, line: 2, baseType: null)
!17 = !DISubprogram(name: "m_fn1", linkageName: "\01?m_fn1@B@@AAE_NXZ", scope: !11, file: !10, line: 6, type: !18, isLocal: false, isDefinition: false, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: false)
!18 = !DISubroutineType(cc: DW_CC_BORLAND_thiscall, types: !19)
!19 = !{!20, !21}
!20 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 32, flags: DIFlagArtificial | DIFlagObjectPointer)
!22 = !DILocalVariable(name: "this", arg: 1, scope: !9, type: !23, flags: DIFlagArtificial | DIFlagObjectPointer)
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 32)
!24 = !DILocation(line: 0, scope: !9)
!25 = !DILocation(line: 8, scope: !9)
