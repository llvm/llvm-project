; RUN: llc -split-dwarf-file=test.dwo -dwarf-version=4 %s -mtriple=i386-unknown-linux-gnu -filetype=obj -o - | \
; RUN: llvm-dwarfdump -v - | FileCheck %s -check-prefix=DWARF4

; RUN: llc -split-dwarf-file=test.dwo -dwarf-version=5 %s -mtriple=i386-unknown-linux-gnu -filetype=obj -o - | \
; RUN: llvm-dwarfdump -v - | FileCheck %s -check-prefix=DWARF5

; Source:
; void foo() {
; }
;
; void bar() {
; }

; DWARF4: .debug_info contents:
; DWARF4: Compile Unit:{{.*}}version = 0x0004
; DWARF4-NOT: Compile Unit
; DWARF4: DW_TAG_compile_unit
; DWARF4-NOT: DW_TAG_{{.*}}
; DWARF4: DW_AT_GNU_dwo_name{{.*}}test.dwo
; DWARF4: DW_AT_GNU_addr_base{{.*}}0x00000000
; DWARF4: .debug_addr contents:
; DWARF4-NEXT: 0x00000000: Addr Section: length = 0x00000000, version = 0x0004, addr_size = 0x04, seg_size = 0x00
; DWARF4-NEXT: Addrs: [
; DWARF4-NEXT: 0x00000000
; DWARF4-NEXT: 0x00000010
; DWARF4-NEXT: ]

; DWARF5: .debug_info contents:
; DWARF5: Compile Unit:{{.*}}version = 0x0005
; DWARF5-NOT: Compile Unit
; DWARF5: DW_TAG_compile_unit
; DWARF5-NOT: DW_TAG_{{.*}}
; DWARF5: DW_AT_GNU_dwo_name{{.*}}test.dwo
; DWARF5: DW_AT_addr_base{{.*}}0x00000008
; DWARF5: DW_AT_low_pc [DW_FORM_addrx] (indexed (00000000) address = 0x0000000000000000 ".text")
; DWARF5: .debug_addr contents:
; DWARF5-NEXT: 0x00000000: Addr Section: length = 0x0000000c, version = 0x0005, addr_size = 0x04, seg_size = 0x00
; DWARF5-NEXT: Addrs: [
; DWARF5-NEXT: 0x00000000
; DWARF5-NEXT: 0x00000010
; DWARF5-NEXT: ]

; ModuleID = './test.c'
source_filename = "./test.c"
target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone
define void @foo() #0 !dbg !8 {
entry:
  ret void, !dbg !12
}

; Function Attrs: noinline nounwind optnone
define void @bar() #0 !dbg !13 {
entry:
  ret void, !dbg !14
}

attributes #0 = { noinline nounwind optnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.1", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 1, !"NumRegisterParameters", i32 0}
!4 = !{i32 2, !"Dwarf Version", i32 5}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 4}
!7 = !{!"clang version 6.0.1"}
!8 = distinct !DISubprogram(name: "foo", scope: !9, file: !9, line: 1, type: !10, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !0)
!9 = !DIFile(filename: "./test.c", directory: "/tmp")
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !DILocation(line: 2, column: 3, scope: !8)
!13 = distinct !DISubprogram(name: "bar", scope: !9, file: !9, line: 5, type: !10, isLocal: false, isDefinition: true, scopeLine: 5, isOptimized: false, unit: !0)
!14 = !DILocation(line: 6, column: 3, scope: !13)
