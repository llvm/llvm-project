; Non-PIC tests
; RUN: sed -e 's/\[\[TLS_MODE\]\]/(localexec)/' %s | llc -filetype=obj -mattr=+bulk-memory,atomics - -o %t.localexec.o
; RUN: sed -e 's/\[\[TLS_MODE\]\]//' %s | llc -filetype=obj -mattr=+bulk-memory,atomics - -o %t.generaldynamic.o
; RUN: llvm-dwarfdump %t.localexec.o | FileCheck %s --check-prefixes=CHECK,NOPIC
; RUN: llvm-dwarfdump %t.generaldynamic.o | FileCheck %s --check-prefixes=CHECK,NOPIC
; RUN: llvm-readobj -r %t.localexec.o | FileCheck %s --check-prefixes=RELOCS-NOSPLIT
; RUN: llvm-readobj -r %t.generaldynamic.o | FileCheck %s --check-prefixes=RELOCS-NOSPLIT

; PIC tests
; RUN: sed -e 's/\[\[TLS_MODE\]\]/(localexec)/' %s | llc -filetype=obj -mattr=+bulk-memory,atomics -relocation-model=pic - -o %t.localexec.pic.o
; RUN: sed -e 's/\[\[TLS_MODE\]\]//' %s | llc -filetype=obj -mattr=+bulk-memory,atomics -relocation-model=pic - -o %t.generaldynamic.pic.o
; RUN: llvm-dwarfdump %t.localexec.pic.o | FileCheck %s --check-prefixes=CHECK,PIC
; RUN: llvm-dwarfdump %t.generaldynamic.pic.o | FileCheck %s --check-prefixes=CHECK,PIC
; RUN: llvm-readobj -r %t.localexec.pic.o | FileCheck %s --check-prefixes=RELOCS-NOSPLIT,RELOCS-PIC-NOSPLIT
; RUN: llvm-readobj -r %t.generaldynamic.pic.o | FileCheck %s --check-prefixes=RELOCS-NOSPLIT,RELOCS-PIC-NOSPLIT

; Non-PIC + split DWARF tests
; RUN: sed -e 's/\[\[TLS_MODE\]\]/(localexec)/' %s | llc -filetype=obj -mattr=+bulk-memory,atomics -split-dwarf-file=%t.localexec.split.dwo -split-dwarf-output=%t.localexec.split.dwo - -o %t.localexec.split.o
; RUN: sed -e 's/\[\[TLS_MODE\]\]//' %s | llc -filetype=obj -mattr=+bulk-memory,atomics -split-dwarf-file=%t.generaldynamic.split.dwo -split-dwarf-output=%t.generaldynamic.split.dwo - -o %t.generaldynamic.split.o
; RUN: llvm-dwarfdump %t.localexec.split.dwo | FileCheck %s --check-prefixes=CHECK,NOPIC
; RUN: llvm-dwarfdump %t.generaldynamic.split.dwo | FileCheck %s --check-prefixes=CHECK,NOPIC
; RUN: llvm-readobj -r %t.localexec.split.dwo | FileCheck %s --check-prefixes=RELOCS-SPLIT
; RUN: llvm-readobj -r %t.generaldynamic.split.dwo | FileCheck %s --check-prefixes=RELOCS-SPLIT

; PIC + split DWARF tests
; RUN: sed -e 's/\[\[TLS_MODE\]\]/(localexec)/' %s | llc -filetype=obj -mattr=+bulk-memory,atomics -relocation-model=pic -split-dwarf-file=%t.localexec.pic.split.dwo -split-dwarf-output=%t.localexec.pic.split.dwo - -o %t.localexec.pic.split.o
; RUN: sed -e 's/\[\[TLS_MODE\]\]//' %s | llc -filetype=obj -mattr=+bulk-memory,atomics -relocation-model=pic -split-dwarf-file=%t.generaldynamic.pic.split.dwo -split-dwarf-output=%t.generaldynamic.pic.split.dwo - -o %t.generaldynamic.pic.split.o
; RUN: llvm-dwarfdump %t.localexec.pic.split.dwo | FileCheck %s --check-prefixes=CHECK,PIC
; RUN: llvm-dwarfdump %t.generaldynamic.pic.split.dwo | FileCheck %s --check-prefixes=CHECK,PIC
; RUN: llvm-readobj -r %t.localexec.pic.split.dwo | FileCheck %s --check-prefixes=RELOCS-SPLIT
; RUN: llvm-readobj -r %t.generaldynamic.pic.split.dwo | FileCheck %s --check-prefixes=RELOCS-SPLIT

; This test is generated from the following C code, after which some unnecessary
; debug info is removed.

; int external_var0 = 111;
; int external_var1 = 222;
; static int internal_var0 = 333;
; static int internal_var1 = 444;
; _Thread_local int external_tls_var0 = 555;
; _Thread_local int external_tls_var1 = 666;
; _Thread_local int internal_tls_var0 = 777;
; _Thread_local int internal_tls_var1 = 888;
;
; void foo(int, int, int, int, int, int, int, int);
;
; void test_tls_pic_globals() {
;   foo(external_var0, external_var1, internal_var0, internal_var1,
;       external_tls_var0, external_tls_var1, internal_tls_var0,
;       internal_tls_var1);
; }

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-f128:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-emscripten"

@external_var0 = global i32 111, align 4, !dbg !0
@external_var1 = global i32 222, align 4, !dbg !5
@internal_var0 = internal global i32 333, align 4, !dbg !8
@internal_var1 = internal global i32 444, align 4, !dbg !10
@external_tls_var0 = thread_local[[TLS_MODE]] global i32 555, align 4, !dbg !12
@external_tls_var1 = thread_local[[TLS_MODE]] global i32 666, align 4, !dbg !14
@internal_tls_var0 = internal thread_local[[TLS_MODE]] global i32 777, align 4, !dbg !16
@internal_tls_var1 = internal thread_local[[TLS_MODE]] global i32 888, align 4, !dbg !18

define void @foo(i32, i32, i32, i32, i32, i32, i32, i32) {
  ret void
}

define void @test_tls_pic_globals() !dbg !24 {
entry:
  %0 = load i32, ptr @external_var0, align 4
  %1 = load i32, ptr @external_var1, align 4
  %2 = load i32, ptr @internal_var0, align 4
  %3 = load i32, ptr @internal_var1, align 4
  %4 = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @external_tls_var0)
  %5 = load i32, ptr %4, align 4
  %6 = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @external_tls_var1)
  %7 = load i32, ptr %6, align 4
  %8 = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @internal_tls_var0)
  %9 = load i32, ptr %8, align 4
  %10 = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @internal_tls_var1)
  %11 = load i32, ptr %10, align 4
  call void @foo(i32 %0, i32 %1, i32 %2, i32 %3, i32 %5, i32 %7, i32 %9, i32 %11)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!20, !21, !22, !23}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "external_var0", scope: !2, file: !3, line: 4, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4)
!3 = !DIFile(filename: "test.c", directory: "")
!4 = !{!0, !5, !8, !10, !12, !14, !16, !18}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "external_var1", scope: !2, file: !3, line: 4, type: !7, isLocal: false, isDefinition: true)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression())
!9 = distinct !DIGlobalVariable(name: "internal_var0", scope: !2, file: !3, line: 6, type: !7, isLocal: true, isDefinition: true)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "internal_var1", scope: !2, file: !3, line: 7, type: !7, isLocal: true, isDefinition: true)
!12 = !DIGlobalVariableExpression(var: !13, expr: !DIExpression())
!13 = distinct !DIGlobalVariable(name: "external_tls_var0", scope: !2, file: !3, line: 8, type: !7, isLocal: false, isDefinition: true)
!14 = !DIGlobalVariableExpression(var: !15, expr: !DIExpression())
!15 = distinct !DIGlobalVariable(name: "external_tls_var1", scope: !2, file: !3, line: 9, type: !7, isLocal: false, isDefinition: true)
!16 = !DIGlobalVariableExpression(var: !17, expr: !DIExpression())
!17 = distinct !DIGlobalVariable(name: "internal_tls_var0", scope: !2, file: !3, line: 9, type: !7, isLocal: true, isDefinition: true)
!18 = !DIGlobalVariableExpression(var: !19, expr: !DIExpression())
!19 = distinct !DIGlobalVariable(name: "internal_tls_var1", scope: !2, file: !3, line: 9, type: !7, isLocal: true, isDefinition: true)
!20 = !{i32 7, !"Dwarf Version", i32 5}
!21 = !{i32 2, !"Debug Info Version", i32 3}
!22 = !{i32 1, !"wchar_size", i32 4}
!23 = !{i32 8, !"PIC Level", i32 2}
!24 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 16, type: !25, spFlags: DISPFlagDefinition, unit: !2)
!25 = !DISubroutineType(types: !26)
!26 = !{null}

; Tests if TLS variables and global variables in PIC mode have a correct debug
; info location. TLS variables should have their location relative to __tls_base
; global, and global variables in PIC objects should have their location
; relative to __memory_base global.

; CHECK:      DW_TAG_variable
; CHECK-NEXT:   DW_AT_name  ("external_var0")
; CHECK:        DW_AT_external  (true)
; NOPIC:        DW_AT_location  (DW_OP_addrx 0x0)
; PIC:          DW_AT_location  (DW_OP_WASM_location 0x3 0x{{[0-9]+}}, DW_OP_addrx 0x0, DW_OP_plus)

; CHECK:      DW_TAG_variable
; CHECK-NEXT:   DW_AT_name  ("external_var1")
; CHECK:        DW_AT_external  (true)
; NOPIC:        DW_AT_location  (DW_OP_addrx 0x1)
; PIC:          DW_AT_location  (DW_OP_WASM_location 0x3 0x{{[0-9]+}}, DW_OP_addrx 0x1, DW_OP_plus)

; CHECK:      DW_TAG_variable
; CHECK-NEXT:   DW_AT_name  ("internal_var0")
; NOPIC:        DW_AT_location  (DW_OP_addrx 0x2)
; PIC:          DW_AT_location  (DW_OP_WASM_location 0x3 0x{{[0-9]+}}, DW_OP_addrx 0x2, DW_OP_plus)

; CHECK:      DW_TAG_variable
; CHECK-NEXT:   DW_AT_name  ("internal_var1")
; NOPIC:        DW_AT_location  (DW_OP_addrx 0x3)
; PIC:          DW_AT_location  (DW_OP_WASM_location 0x3 0x{{[0-9]+}}, DW_OP_addrx 0x3, DW_OP_plus)

; CHECK:      DW_TAG_variable
; CHECK-NEXT:   DW_AT_name  ("external_tls_var0")
; CHECK:        DW_AT_external  (true)
; CHECK:        DW_AT_location  (DW_OP_WASM_location 0x3 0x{{[0-9]+}}, DW_OP_addrx 0x4, DW_OP_plus)

; CHECK:      DW_TAG_variable
; CHECK-NEXT:   DW_AT_name  ("external_tls_var1")
; CHECK:        DW_AT_external  (true)
; CHECK:        DW_AT_location  (DW_OP_WASM_location 0x3 0x{{[0-9]+}}, DW_OP_addrx 0x5, DW_OP_plus)

; CHECK:      DW_TAG_variable
; CHECK-NEXT:   DW_AT_name  ("internal_tls_var0")
; CHECK:        DW_AT_location  (DW_OP_WASM_location 0x3 0x{{[0-9]+}}, DW_OP_addrx 0x6, DW_OP_plus)

; CHECK:      DW_TAG_variable
; CHECK-NEXT:   DW_AT_name  ("internal_tls_var1")
; CHECK:        DW_AT_location  (DW_OP_WASM_location 0x3 0x{{[0-9]+}}, DW_OP_addrx 0x7, DW_OP_plus)

; In non-split DWARF, .debug_info section contains relocations referring to
; __stack_pointer, __tls_base, and __memory_base (if used)

; RELOCS-NOSPLIT:         Relocations [
; RELOCS-NOSPLIT:           Section (8) .debug_info {
; RELOCS-NOSPLIT-DAG:         0x{{.*}} R_WASM_GLOBAL_INDEX_I32 __tls_base
; RELOCS-NOSPLIT-DAG:         0x{{.*}} R_WASM_GLOBAL_INDEX_I32 __stack_pointer
; RELOCS-PIC-NOSPLIT-DAG:     0x{{.*}} R_WASM_GLOBAL_INDEX_I32 __memory_base
; RELOCS-NOSPLIT:           }
; RELOCS-NOSPLIT:           Section (9) .debug_str_offsets {

; In split DWARF, there should be no relocations in .dwo files.

; RELOCS-SPLIT:           Relocations [
; RELOCS-SPLIT-NEXT:      ]
