; RUN: llc < %s -mtriple=x86_64 -filetype=obj -o %t
; RUN: llvm-dwarfdump -v %t | FileCheck %s

; Source code to regenerate:
; __attribute__((objc_root_class))
; @interface Root
; - (int)direct_method __attribute__((objc_direct));
; @end
;
; @implementation Root
; - (int)direct_method __attribute__((objc_direct)) {
;   return 42;
; }
; @end
;
; clang -O0 -g -gdwarf-5 direct.m -c

; CHECK: DW_TAG_subprogram [3]
; CHECK: DW_AT_APPLE_objc_direct
; CHECK-SAME: DW_FORM_flag_present
; CHECK: DW_TAG_formal_parameter [4]

; ModuleID = 'direct.bc'
source_filename = "direct.m"

%0 = type opaque

define hidden i32 @"\01-[Root direct_method]"(ptr %self, ptr %_cmd) {
entry:
  %retval = alloca i32, align 4
  %0 = load i32, ptr %retval, align 4
  ret i32 %0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!19, !20}
!llvm.ident = !{}

!0 = distinct !DICompileUnit(language: DW_LANG_ObjC, file: !1, producer: "clang version 10.0.0 (https://github.com/llvm/llvm-project d6b2f33e2b6338d24cf756ba220939aecc81210d)", isOptimized: false, runtimeVersion: 2, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "direct.m", directory: "/", checksumkind: CSK_MD5, checksum: "6b49fad130344b0011fc0eef65949390")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "Root", scope: !1, file: !1, line: 2, flags: DIFlagObjcClassComplete, elements: !5, runtimeLang: DW_LANG_ObjC)
!5 = !{!6}
!6 = !DISubprogram(name: "-[Root direct_method]", scope: !4, file: !1, line: 7, type: !7, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagObjCDirect, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !10, !11}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!11 = !DIDerivedType(tag: DW_TAG_typedef, name: "SEL", file: !1, baseType: !12, flags: DIFlagArtificial)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!13 = !DICompositeType(tag: DW_TAG_structure_type, name: "objc_selector", file: !1, flags: DIFlagFwdDecl)
!19 = !{i32 7, !"Dwarf Version", i32 5}
!20 = !{i32 2, !"Debug Info Version", i32 3}
