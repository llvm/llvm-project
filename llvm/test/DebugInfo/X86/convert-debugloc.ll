; RUN: llc -mtriple=x86_64 -dwarf-version=4 -filetype=obj -O0 < %s | llvm-dwarfdump - \
; RUN:   | FileCheck %s --check-prefix=NOCONV "--implicit-check-not={{DW_TAG|NULL}}"

; Test lldb default: OP_convert is unsupported when using MachO
; RUN: llc -mtriple=x86_64-apple-darwin -dwarf-version=5 -filetype=obj -O0 < %s -debugger-tune=lldb | llvm-dwarfdump - \
; RUN:   | FileCheck %s --check-prefix=CONV "--implicit-check-not={{DW_TAG|NULL}}"
; RUN: llc -mtriple=x86_64-pc-linux-gnu -dwarf-version=5 -filetype=obj -O0 < %s -debugger-tune=lldb | llvm-dwarfdump - \
; RUN:   | FileCheck %s --check-prefix=NOCONV "--implicit-check-not={{DW_TAG|NULL}}"

; Test gdb default: OP_convert is only disabled in split DWARF
; RUN: llc -mtriple=x86_64 -dwarf-version=5 -filetype=obj -O0 < %s -debugger-tune=gdb  | llvm-dwarfdump - \
; RUN:   | FileCheck %s --check-prefix=CONV "--implicit-check-not={{DW_TAG|NULL}}"
; RUN: llc -mtriple=x86_64-pc-linux-gnu  -dwarf-version=5 -filetype=obj -O0 < %s -debugger-tune=gdb   -split-dwarf-file=baz.dwo | llvm-dwarfdump - \
; RUN:   | FileCheck %s --check-prefix=NOCONV --check-prefix=SPLIT "--implicit-check-not={{DW_TAG|NULL}}"

; Test the ability to override the platform default in either direction
; RUN: llc -mtriple=x86_64 -dwarf-version=5 -filetype=obj -O0 < %s -debugger-tune=gdb  -dwarf-op-convert=Disable | llvm-dwarfdump - \
; RUN:   | FileCheck %s --check-prefix=NOCONV "--implicit-check-not={{DW_TAG|NULL}}"
; RUN: llc -mtriple=x86_64-pc-linux-gnu -dwarf-version=5 -filetype=obj -O0 < %s -debugger-tune=lldb -dwarf-op-convert=Enable | llvm-dwarfdump - \
; RUN:   | FileCheck %s --check-prefix=CONV "--implicit-check-not={{DW_TAG|NULL}}"

; Test DW_OP_convert + Split DWARF
; RUN: llc -mtriple=x86_64-pc-linux-gnu -dwarf-version=5 -filetype=obj -O0 < %s -debugger-tune=lldb -dwarf-op-convert=Enable -split-dwarf-file=baz.dwo | llvm-dwarfdump - \
; RUN:   | FileCheck %s --check-prefix=CONV --check-prefix=SPLITCONV --check-prefix=SPLIT "--implicit-check-not={{DW_TAG|NULL}}"

; RUN: llc -mtriple=x86_64 -dwarf-version=5 -filetype=obj -O0 < %s -debugger-tune=gdb  | llvm-dwarfdump -v -debug-info - \
; RUN:   | FileCheck %s --check-prefix=VERBOSE --check-prefix=CONV "--implicit-check-not={{DW_TAG|NULL}}"


; SPLITCONV: Compile Unit:{{.*}} DWO_id = 0x62f17241069b1fa3
; SPLIT: DW_TAG_skeleton_unit

; CONV: DW_TAG_compile_unit
; CONV:[[SIG8:.*]]:   DW_TAG_base_type
; CONV-NEXT:DW_AT_name {{.*}}"DW_ATE_signed_8")
; CONV-NEXT:DW_AT_encoding {{.*}}DW_ATE_signed)
; CONV-NEXT:DW_AT_byte_size {{.*}}0x01)
; CONV-NOT: DW_AT
; CONV:[[SIG32:.*]]:   DW_TAG_base_type
; CONV-NEXT:DW_AT_name {{.*}}"DW_ATE_signed_32")
; CONV-NEXT:DW_AT_encoding {{.*}}DW_ATE_signed)
; CONV-NEXT:DW_AT_byte_size {{.*}}0x04)
; CONV-NOT: DW_AT
; CONV:   DW_TAG_subprogram
; CONV:     DW_TAG_formal_parameter
; CONV:     DW_TAG_variable
; CONV:     DW_AT_location {{.*}}DW_OP_constu 0x20, DW_OP_lit0, DW_OP_plus, DW_OP_convert (
; VERBOSE-SAME: [[SIG8]] ->
; CONV-SAME: [[SIG8]]) "DW_ATE_signed_8", DW_OP_convert (
; VERBOSE-SAME: [[SIG32]] ->
; CONV-SAME: [[SIG32]]) "DW_ATE_signed_32", DW_OP_stack_value)
; CONV:       DW_AT_name {{.*}}"y")
; CONV:     NULL
; CONV:   DW_TAG_base_type
; CONV:     DW_AT_name {{.*}}"signed char")
; CONV:   DW_TAG_base_type
; CONV:     DW_AT_name {{.*}}"int")
; CONV:   NULL

; NOCONV: DW_TAG_compile_unit
; NOCONV:   DW_TAG_subprogram
; NOCONV:     DW_TAG_formal_parameter
; NOCONV:     DW_TAG_variable
; NOCONV:       DW_AT_location (
; NOCONV:         {{.*}}, DW_OP_dup, DW_OP_constu 0x7, DW_OP_shr, DW_OP_lit0, DW_OP_not, DW_OP_mul, DW_OP_constu 0x8, DW_OP_shl, DW_OP_or, DW_OP_stack_value)
; NOCONV:       DW_AT_name ("y")
; NOCONV:     NULL
; NOCONV:   DW_TAG_base_type
; NOCONV:     DW_AT_name ("signed char")
; NOCONV:   DW_TAG_base_type
; NOCONV:     DW_AT_name ("int")
; NOCONV:   NULL


; Function Attrs: noinline nounwind uwtable
define dso_local signext i8 @foo(i8 signext %x) !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i8 %x, metadata !11, metadata !DIExpression()), !dbg !12
;; This test depends on "convert" surviving all the way to the final object.
;; So, insert something before DW_OP_LLVM_convert that the expression folder
;; will not attempt to eliminate.  At the moment, only "convert" ops are folded.
;; If you have to change the expression, the expected DWO_id also changes.
  call void @llvm.dbg.value(metadata i8 32, metadata !13, metadata !DIExpression(DW_OP_lit0, DW_OP_plus, DW_OP_LLVM_convert, 8, DW_ATE_signed, DW_OP_LLVM_convert, 32, DW_ATE_signed, DW_OP_stack_value)), !dbg !15
  ret i8 %x, !dbg !16
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata)

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (trunk 353791) (llvm/trunk 353801)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "dbg.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "2a034da6937f5b9cf6dd2d89127f57fd")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 9.0.0 (trunk 353791) (llvm/trunk 353801)"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !8, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "signed char", size: 8, encoding: DW_ATE_signed_char)
!11 = !DILocalVariable(name: "x", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!12 = !DILocation(line: 1, column: 29, scope: !7)
!13 = !DILocalVariable(name: "y", scope: !7, file: !1, line: 3, type: !14)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DILocation(line: 3, column: 14, scope: !7)
!16 = !DILocation(line: 4, column: 3, scope: !7)
