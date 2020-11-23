; RUN: llc -debugger-tune=gdb -filetype=obj %s -o - | llvm-dwarfdump - \
; RUN:   | FileCheck %s --check-prefixes=GDB,BOTH
; RUN: llc -debugger-tune=sce -filetype=obj %s -o - | llvm-dwarfdump - \
; RUN:   | FileCheck %s --check-prefixes=SCE,BOTH

; BOTH: DW_TAG_variable
; BOTH-NEXT:  DW_AT_location
; GDB-NEXT:      {{.*}}: DW_OP_implicit_value 0x2 0xe0 0x51
; SCE-NEXT:      {{.*}}: DW_OP_constu 0x51e0, DW_OP_stack_value
; GDB-NEXT:      {{.*}}: DW_OP_implicit_value 0x2 0x40 0x51
; SCE-NEXT:      {{.*}}: DW_OP_constu 0x5140, DW_OP_stack_value
; BOTH-NEXT:  DW_AT_name    ("a")

source_filename = "-"
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7-unknown-unknown"

define dso_local arm_aapcscc void @f() local_unnamed_addr !dbg !8 {
entry:
  call void @llvm.dbg.value(metadata half 0xH51E0, metadata !13, metadata !DIExpression()), !dbg !15
  call arm_aapcscc void bitcast (void (...)* @g to void ()*)(), !dbg !15
  call void @llvm.dbg.value(metadata half 0xH5140, metadata !13, metadata !DIExpression()), !dbg !15
  call arm_aapcscc void bitcast (void (...)* @g to void ()*)(), !dbg !15
  ret void, !dbg !15
}

declare dso_local arm_aapcscc void @g(...) local_unnamed_addr

declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #2 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "-", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 1, !"min_enum_size", i32 4}
!7 = !{!"clang version 12.0.0"}
!8 = distinct !DISubprogram(name: "f", scope: !1, file: !1, type: !10, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !{!13}
!13 = !DILocalVariable(name: "a", scope: !8, file: !1, type: !14)
!14 = !DIBasicType(name: "_Float16", size: 16, encoding: DW_ATE_float)
!15 = !DILocation(line: 0, scope: !8)
