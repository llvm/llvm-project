; REQUIRES: object-emission

; RUN: %llc_dwarf < %s -filetype=obj | llvm-dwarfdump -debug-dump=line - | FileCheck %s
; RUN: %llc_dwarf < %s -filetype=asm | FileCheck --check-prefix=ASM %s

; If multiple line tables are emitted, one per CU, those line tables can
; unambiguously rely on the comp_dir of their owning CU and use directory '0'
; to refer to it.

; CHECK: .debug_line contents:
; CHECK-NEXT: Line table prologue:
; CHECK-NOT: include_directories
; CHECK: file_names[   1]   0 {{.*}} a.cpp
; CHECK-NOT: file_names

; CHECK: Line table prologue:
; CHECK-NOT: include_directories
; CHECK: file_names[   1]   0 {{.*}} b.cpp
; CHECK-NOT: file_names

; However, if a single line table is emitted and shared between CUs, the
; comp_dir is ambiguous and relying on it would lead to different path
; interpretations depending on which CU lead to the table - so ensure that
; full paths are always emitted in this case, never comp_dir relative.

; ASM: .file   1 "/tmp/dbginfo/a{{[/\\]+}}a.cpp"
; ASM: .file   2 "/tmp/dbginfo/b{{[/\\]+}}b.cpp"

; Generated from the following source compiled to bitcode from within their
; respective directories (with debug info) and linked together with llvm-link

; a/a.cpp
; void func() {
; }

; b/b.cpp
; void func();
; int main() {
;   func();
; }

; Function Attrs: nounwind uwtable
define void @_Z4funcv() #0 !dbg !4 {
entry:
  ret void, !dbg !19
}

; Function Attrs: uwtable
define i32 @main() #1 !dbg !11 {
entry:
  call void @_Z4funcv(), !dbg !20
  ret i32 0, !dbg !21
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0, !8}
!llvm.module.flags = !{!16, !17}
!llvm.ident = !{!18, !18}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "a.cpp", directory: "/tmp/dbginfo/a")
!2 = !{}
!4 = distinct !DISubprogram(name: "func", linkageName: "_Z4funcv", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "a.cpp", directory: "/tmp/dbginfo/a")
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: FullDebug, file: !9, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!9 = !DIFile(filename: "b.cpp", directory: "/tmp/dbginfo/b")
!11 = distinct !DISubprogram(name: "main", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !8, scopeLine: 2, file: !9, scope: !12, type: !13, variables: !2)
!12 = !DIFile(filename: "b.cpp", directory: "/tmp/dbginfo/b")
!13 = !DISubroutineType(types: !14)
!14 = !{!15}
!15 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!16 = !{i32 2, !"Dwarf Version", i32 4}
!17 = !{i32 1, !"Debug Info Version", i32 3}
!18 = !{!"clang version 3.5.0 "}
!19 = !DILocation(line: 2, scope: !4)
!20 = !DILocation(line: 3, scope: !11)
!21 = !DILocation(line: 4, scope: !11)

