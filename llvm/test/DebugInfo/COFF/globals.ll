; RUN: llc < %s | FileCheck %s --check-prefix=ASM
; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s --check-prefix=OBJ

; C++ source to regenerate:
; $ cat t.cpp
; int first;
; template <typename T> struct A { static const int comdat = 3; };
; const int *middle = &A<void>::comdat;
; int last;
; $ clang t.cpp -S -emit-llvm -g -gcodeview -o t.ll

; ASM:        .section        .debug$S,"dr"
; ASM:        .p2align        2
; ASM:        .long   4                       # Debug section magic

; ASM:        .short  {{.*-.*}}               # Record length
; ASM:        .short  4364                    # Record kind: S_LDATA32
; ASM:        .long   116                     # Type
; ASM:        .secrel32       "?first@@3HA"   # DataOffset
; ASM:        .secidx "?first@@3HA"           # Segment
; ASM:        .asciz  "first"                 # Name

; ASM:        .short  {{.*-.*}}               # Record length
; ASM:        .short  4371                    # Record kind: S_GTHREAD32
; ASM:        .long   4097                    # Type
; ASM:        .secrel32       "?middle@@3PEBHEB" # DataOffset
; ASM:        .secidx "?middle@@3PEBHEB"      # Segment
; ASM:        .asciz  "middle"                # Name

; ASM:        .short  {{.*-.*}}               # Record length
; ASM:        .short  4365                    # Record kind: S_GDATA32
; ASM:        .long   116                     # Type
; ASM:        .secrel32       "?last@@3HA"    # DataOffset
; ASM:        .secidx "?last@@3HA"            # Segment
; ASM:        .asciz  "last"                  # Name

; ASM:        .section        .debug$S,"dr",associative,"?comdat@?$A@X@@2HB"
; ASM:        .p2align        2
; ASM:        .long   4                       # Debug section magic

; ASM:        .short  {{.*-.*}}               # Record length
; ASM:        .short  4365                    # Record kind: S_GDATA32
; ASM:        .long   4096                    # Type
; ASM:        .secrel32       "?comdat@?$A@X@@2HB" # DataOffset
; ASM:        .secidx "?comdat@?$A@X@@2HB"    # Segment
; ASM:        .asciz  "comdat"                # Name

; OBJ: CodeViewTypes [
; OBJ:   Section: .debug$T
; OBJ:   Magic: 0x4
; OBJ:   Modifier (0x1000) {
; OBJ:     TypeLeafKind: LF_MODIFIER (0x1001)
; OBJ:     ModifiedType: int (0x74)
; OBJ:     Modifiers [ (0x1)
; OBJ:       Const (0x1)
; OBJ:     ]
; OBJ:   }
; OBJ:   Pointer (0x1001) {
; OBJ:     TypeLeafKind: LF_POINTER (0x1002)
; OBJ:     PointeeType: const int (0x1000)
; OBJ:     PointerAttributes: 0x1000C
; OBJ:     PtrType: Near64 (0xC)
; OBJ:     PtrMode: Pointer (0x0)
; OBJ:     IsFlat: 0
; OBJ:     IsConst: 0
; OBJ:     IsVolatile: 0
; OBJ:     IsUnaligned: 0
; OBJ:   }
; OBJ: ]

; OBJ: CodeViewDebugInfo [
; OBJ:   Section: .debug$S
; OBJ:   Magic: 0x4
; OBJ:   Subsection [
; OBJ:     SubSectionType: Symbols (0xF1)
; OBJ:     DataSym {
; OBJ:       Kind: S_LDATA32 (0x110C)
; OBJ:       DataOffset: ?first@@3HA+0x0
; OBJ:       Type: int (0x74)
; OBJ:       DisplayName: first
; OBJ:       LinkageName: ?first@@3HA
; OBJ:     }
; OBJ:     ThreadLocalDataSym {
; OBJ:       DataOffset: ?middle@@3PEBHEB+0x0
; OBJ:       Type: const int* (0x1001)
; OBJ:       DisplayName: middle
; OBJ:       LinkageName: ?middle@@3PEBHEB
; OBJ:     }
; OBJ:     DataSym {
; OBJ:       Kind: S_GDATA32 (0x110D)
; OBJ:       DataOffset: ?last@@3HA+0x0
; OBJ:       Type: int (0x74)
; OBJ:       DisplayName: last
; OBJ:       LinkageName: ?last@@3HA
; OBJ:     }
; OBJ:   ]
; OBJ: ]
; OBJ: CodeViewDebugInfo [
; OBJ:   Section: .debug$S (7)
; OBJ:   Magic: 0x4
; OBJ:   Subsection [
; OBJ:     SubSectionType: Symbols (0xF1)
; OBJ:     SubSectionSize: 0x15
; OBJ:     DataSym {
; OBJ:       DataOffset: ?comdat@?$A@X@@2HB+0x0
; OBJ:       Type: const int (0x1000)
; OBJ:       DisplayName: comdat
; OBJ:       LinkageName: ?comdat@?$A@X@@2HB
; OBJ:     }
; OBJ:   ]
; OBJ: ]

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.23918"

$"\01?comdat@?$A@X@@2HB" = comdat any

@"\01?first@@3HA" = internal global i32 0, align 4
@"\01?comdat@?$A@X@@2HB" = linkonce_odr constant i32 3, comdat, align 4
@"\01?middle@@3PEBHEB" = thread_local global i32* @"\01?comdat@?$A@X@@2HB", align 8
@"\01?last@@3HA" = global i32 0, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!16, !17, !18}
!llvm.ident = !{!19}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 (trunk 271937)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!2 = !{}
!3 = !{!4, !6, !13, !15}
!4 = distinct !DIGlobalVariable(name: "first", linkageName: "\01?first@@3HA", scope: !0, file: !1, line: 1, type: !5, isLocal: true, isDefinition: true, variable: i32* @"\01?first@@3HA")
!5 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = distinct !DIGlobalVariable(name: "comdat", linkageName: "\01?comdat@?$A@X@@2HB", scope: !0, file: !1, line: 2, type: !7, isLocal: false, isDefinition: true, variable: i32* @"\01?comdat@?$A@X@@2HB", declaration: !8)
!7 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !5)
!8 = !DIDerivedType(tag: DW_TAG_member, name: "comdat", scope: !9, file: !1, line: 2, baseType: !7, flags: DIFlagStaticMember, extraData: i32 3)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A<void>", file: !1, line: 2, size: 8, align: 8, elements: !10, templateParams: !11)
!10 = !{!8}
!11 = !{!12}
!12 = !DITemplateTypeParameter(name: "T", type: null)
!13 = distinct !DIGlobalVariable(name: "middle", linkageName: "\01?middle@@3PEBHEB", scope: !0, file: !1, line: 3, type: !14, isLocal: false, isDefinition: true, variable: i32** @"\01?middle@@3PEBHEB")
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, align: 64)
!15 = distinct !DIGlobalVariable(name: "last", linkageName: "\01?last@@3HA", scope: !0, file: !1, line: 4, type: !5, isLocal: false, isDefinition: true, variable: i32* @"\01?last@@3HA")
!16 = !{i32 2, !"CodeView", i32 1}
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = !{i32 1, !"PIC Level", i32 2}
!19 = !{!"clang version 3.9.0 (trunk 271937)"}
