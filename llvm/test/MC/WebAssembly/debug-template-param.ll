; RUN: llc -filetype=obj %s -o - | llvm-readobj -r  - | FileCheck %s

; Test for PR50408. Compiled from:
; char a();
; template <typename, char b()>
; void f() { b(); }
; void g() { f<char, a>(); }

; CHECK: Section (10) .debug_addr
; CHECK-NEXT:    0x8 R_WASM_FUNCTION_OFFSET_I32 _Z1gv 0
; CHECK-NEXT:   0xC R_WASM_FUNCTION_OFFSET_I32 _Z1fIcXadL_Z1avEEEvv 0
; ensure that the reloc type is correct for _Z1av which is undefined
; CHECK-NEXT:    0x10 R_WASM_FUNCTION_OFFSET_I32 _Z1av 0
; CHECK-NEXT:  }

; ModuleID = 'PR50408.cc'
source_filename = "PR50408.cc"
target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128-ni:1"
target triple = "wasm32-unknown-emscripten"

$_Z1fIcXadL_Z1avEEEvv = comdat any

; Function Attrs: noinline optnone mustprogress
define hidden void @_Z1gv() #0 !dbg !7 {
entry:
  call void @_Z1fIcXadL_Z1avEEEvv(), !dbg !10
  ret void, !dbg !11
}

; Function Attrs: noinline optnone mustprogress
define linkonce_odr hidden void @_Z1fIcXadL_Z1avEEEvv() #0 comdat !dbg !12 {
entry:
  %call = call signext i8 @_Z1av(), !dbg !20
  ret void, !dbg !21
}

declare signext i8 @_Z1av() #1

attributes #0 = { noinline optnone mustprogress "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" }
attributes #1 = { "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 13.0.0 (https://github.com/llvm/llvm-project.git 5027637fa1d409e3ca78dab60dc2e2db6c62c175)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "PR50408.cc", directory: "/s/emr/emscripten-releases/localtests", checksumkind: CSK_MD5, checksum: "285a5682ae46dbbe90ccfb84cdef66c7")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 13.0.0 (https://github.com/llvm/llvm-project.git 5027637fa1d409e3ca78dab60dc2e2db6c62c175)"}
!7 = distinct !DISubprogram(name: "g", linkageName: "_Z1gv", scope: !1, file: !1, line: 5, type: !8, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocation(line: 5, column: 12, scope: !7)
!11 = !DILocation(line: 5, column: 26, scope: !7)
!12 = distinct !DISubprogram(name: "f<char, &a>", linkageName: "_Z1fIcXadL_Z1avEEEvv", scope: !1, file: !1, line: 4, type: !8, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, templateParams: !13, retainedNodes: !2)
!13 = !{!14, !16}
!14 = !DITemplateTypeParameter(type: !15)
!15 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!16 = !DITemplateValueParameter(name: "b", type: !17, value: ptr @_Z1av)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 32)
!18 = !DISubroutineType(types: !19)
!19 = !{!15}
!20 = !DILocation(line: 4, column: 12, scope: !12)
!21 = !DILocation(line: 4, column: 17, scope: !12)
