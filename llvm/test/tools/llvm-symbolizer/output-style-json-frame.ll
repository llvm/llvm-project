;; This test checks JSON output for FRAME.

; REQUIRES: aarch64-registered-target

;; Show how library errors are reported in the output.
; RUN: llvm-symbolizer "FRAME %t-no-file.o 0" --output-style=JSON | \
; RUN:   FileCheck %s -DMSG=%errc_ENOENT --check-prefix=NO-FILE --strict-whitespace --match-full-lines --implicit-check-not={{.}}
; NO-FILE:[{"Address":"0x0","Error":{"Message":"[[MSG]]"},"ModuleName":"{{.*}}no-file.o"}]

;; Handle invalid argument.
; RUN: llvm-symbolizer "FRAME tmp.o Z" --output-style=JSON | \
; RUN:   FileCheck %s -DMSG=%errc_ENOENT --check-prefix=INVARG --strict-whitespace --match-full-lines --implicit-check-not={{.}}
; INVARG:[{"Error":{"Message":"[[MSG]]"},"ModuleName":"tmp.o","SymName":"Z"}]

; RUN: llc -filetype=obj -o %t.o %s 

;; Resolve out of range address. Expected an empty array.
; RUN: llvm-symbolizer "FRAME %t.o 0x10000000" --output-style=JSON | \
; RUN:   FileCheck %s --check-prefix=NOT-FOUND --strict-whitespace --match-full-lines --implicit-check-not={{.}}
; NOT-FOUND:[{"Address":"0x10000000","Frame":[],"ModuleName":"{{.*}}.o"}]

;; Resolve valid address. Note we check 0, non-zero and missing TagOffset cases.
; RUN: llvm-symbolizer "FRAME %t.o 0" --output-style=JSON | \
; RUN:   FileCheck %s --strict-whitespace --match-full-lines --implicit-check-not={{.}}
; CHECK:[{"Address":"0x0","Frame":[{"DeclFile":"/x.c","DeclLine":2,"FrameOffset":24,"FunctionName":"f","Name":"a","Size":"0x8","TagOffset":"0x0"},{"DeclFile":"/x.c","DeclLine":3,"FrameOffset":16,"FunctionName":"f","Name":"b","Size":"0x8","TagOffset":"0x1"},{"DeclFile":"/x.c","DeclLine":4,"FrameOffset":12,"FunctionName":"f","Name":"c","Size":"0x4","TagOffset":""}],"ModuleName":"{{.*}}.o"}]

target triple="aarch64--"

define void @f() !dbg !6 {
entry:
  %a = alloca ptr
  %b = alloca ptr
  %c = alloca i32 ; To check a variable with a different size.
  ; Note: The following 2 lines declares the tag offsets we are checking in this test.
  ; The tag offset for the 3rd variable is missing for purpose.
  call void @llvm.dbg.declare(metadata ptr %a, metadata !12, metadata !DIExpression(DW_OP_LLVM_tag_offset, 0)), !dbg !15
  call void @llvm.dbg.declare(metadata ptr %b, metadata !13, metadata !DIExpression(DW_OP_LLVM_tag_offset, 1)), !dbg !16
  call void @llvm.dbg.declare(metadata ptr %c, metadata !14, metadata !DIExpression()), !dbg !17
  ret void, !dbg !18
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "x.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang"}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, flags:
DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!10 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !11)
!11 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!19 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocalVariable(name: "a", scope: !6, file: !1, line: 2, type: !9)
!13 = !DILocalVariable(name: "b", scope: !6, file: !1, line: 3, type: !9)
!14 = !DILocalVariable(name: "c", scope: !6, file: !1, line: 4, type: !19)
!15 = !DILocation(line: 2, column: 10, scope: !6)
!16 = !DILocation(line: 3, column: 11, scope: !6)
!17 = !DILocation(line: 4, column: 12, scope: !6)
!18 = !DILocation(line: 5, column: 13, scope: !6)
