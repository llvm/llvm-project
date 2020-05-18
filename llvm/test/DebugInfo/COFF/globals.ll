; RUN: llc < %s | FileCheck %s --check-prefix=ASM
; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s --check-prefix=OBJ
; RUN: llc < %s | llvm-mc -filetype=obj --triple=x86_64-windows | llvm-readobj - --codeview | FileCheck %s --check-prefix=OBJ
; RUN: llc < %s -filetype=obj | obj2yaml | FileCheck %s --check-prefix=YAML

; C++ source to regenerate:
; $ cat a.cpp
; int first;
; template <typename T> struct A { static const int comdat = 3; };
; thread_local const int *middle = &A<void>::comdat;
; namespace foo {
; thread_local int globalTLS = 4;
; static thread_local int staticTLS = 5;
; int justGlobal = 6;
; static int globalStatic = 7;
; }
; int last;
; int bar() {
;   return foo::globalStatic + foo::globalTLS + foo::staticTLS;
; }
; $ clang-cl a.cpp /c /Z7 /GS- /clang:-S /clang:-emit-llvm

; ASM:        .section        .debug$S,"dr"
; ASM-NEXT:   .p2align        2
; ASM-NEXT:   .long   4                       # Debug section magic

; ASM:        .short  4365                    # Record kind: S_GDATA32
; ASM-NEXT:   .long   116                     # Type
; ASM-NEXT:   .secrel32       "?first@@3HA"   # DataOffset
; ASM-NEXT:   .secidx "?first@@3HA"           # Segment
; ASM-NEXT:   .asciz  "first"                 # Name
; ASM-NEXT:   .p2align        2

; ASM:        .short  4371                    # Record kind: S_GTHREAD32
; ASM-NEXT:   .long   4100                    # Type
; ASM-NEXT:   .secrel32       "?middle@@3PEBHEB" # DataOffset
; ASM-NEXT:   .secidx "?middle@@3PEBHEB"      # Segment
; ASM-NEXT:   .asciz  "middle"                # Name
; ASM-NEXT:   .p2align        2

; ASM:        .short  4371                    # Record kind: S_GTHREAD32
; ASM-NEXT:   .long   116                     # Type
; ASM-NEXT:   .secrel32       "?globalTLS@foo@@3HA" # DataOffset
; ASM-NEXT:   .secidx "?globalTLS@foo@@3HA"   # Segment
; ASM-NEXT:   .asciz  "foo::globalTLS"        # Name
; ASM-NEXT:   .p2align        2

; ASM:        .short  4365                    # Record kind: S_GDATA32
; ASM-NEXT:   .long   116                     # Type
; ASM-NEXT:   .secrel32       "?justGlobal@foo@@3HA" # DataOffset
; ASM-NEXT:   .secidx "?justGlobal@foo@@3HA"  # Segment
; ASM-NEXT:   .asciz  "foo::justGlobal"       # Name
; ASM-NEXT:   .p2align        2

; ASM:        .short  4365                    # Record kind: S_GDATA32
; ASM-NEXT:   .long   116                     # Type
; ASM-NEXT:   .secrel32       "?last@@3HA"    # DataOffset
; ASM-NEXT:   .secidx "?last@@3HA"            # Segment
; ASM-NEXT:   .asciz  "last"                  # Name
; ASM-NEXT:   .p2align        2

; ASM:        .short  4364                    # Record kind: S_LDATA32
; ASM-NEXT:   .long   116                     # Type
; ASM-NEXT:   .secrel32       "?globalStatic@foo@@3HA" # DataOffset
; ASM-NEXT:   .secidx "?globalStatic@foo@@3HA" # Segment
; ASM-NEXT:   .asciz  "foo::globalStatic"     # Name
; ASM-NEXT:   .p2align        2

; ASM:        .short  4370                    # Record kind: S_LTHREAD32
; ASM-NEXT:   .long   116                     # Type
; ASM-NEXT:   .secrel32       "?staticTLS@foo@@3HA" # DataOffset
; ASM-NEXT:   .secidx "?staticTLS@foo@@3HA"   # Segment
; ASM-NEXT:   .asciz  "foo::staticTLS"        # Name
; ASM-NEXT:   .p2align        2

; ASM:        .section        .debug$S,"dr",associative,"?comdat@?$A@X@@2HB"
; ASM-NEXT:   .p2align        2
; ASM-NEXT:   .long   4                       # Debug section magic

; ASM:        .short  4365                    # Record kind: S_GDATA32
; ASM-NEXT:   .long   4099                    # Type
; ASM-NEXT:   .secrel32       "?comdat@?$A@X@@2HB" # DataOffset
; ASM-NEXT:   .secidx "?comdat@?$A@X@@2HB"    # Segment
; ASM-NEXT:   .asciz  "comdat"                # Name

; OBJ: CodeViewDebugInfo [
; OBJ:   Section: .debug$S
; OBJ:   Magic: 0x4
; OBJ:   Subsection [

; OBJ-LABEL:    GlobalData {
; OBJ-NEXT:       Kind: S_GDATA32 (0x110D)
; OBJ-NEXT:       DataOffset: ?first@@3HA+0x0
; OBJ-NEXT:       Type: int (0x74)
; OBJ-NEXT:       DisplayName: first
; OBJ-NEXT:       LinkageName: ?first@@3HA
; OBJ-NEXT:     }
; OBJ-NEXT:     GlobalTLS {
; OBJ-NEXT:       Kind: S_GTHREAD32 (0x1113)
; OBJ-NEXT:       DataOffset: ?middle@@3PEBHEB+0x0
; OBJ-NEXT:       Type: const int* (0x1004)
; OBJ-NEXT:       DisplayName: middle
; OBJ-NEXT:       LinkageName: ?middle@@3PEBHEB
; OBJ-NEXT:     }
; OBJ-NEXT:     GlobalTLS {
; OBJ-NEXT:       Kind: S_GTHREAD32 (0x1113)
; OBJ-NEXT:       DataOffset: ?globalTLS@foo@@3HA+0x0
; OBJ-NEXT:       Type: int (0x74)
; OBJ-NEXT:       DisplayName: foo::globalTLS
; OBJ-NEXT:       LinkageName: ?globalTLS@foo@@3HA
; OBJ-NEXT:     }
; OBJ-NEXT:     GlobalData {
; OBJ-NEXT:       Kind: S_GDATA32 (0x110D)
; OBJ-NEXT:       DataOffset: ?justGlobal@foo@@3HA+0x0
; OBJ-NEXT:       Type: int (0x74)
; OBJ-NEXT:       DisplayName: foo::justGlobal
; OBJ-NEXT:       LinkageName: ?justGlobal@foo@@3HA
; OBJ-NEXT:     }
; OBJ-NEXT:     GlobalData {
; OBJ-NEXT:       Kind: S_GDATA32 (0x110D)
; OBJ-NEXT:       DataOffset: ?last@@3HA+0x0
; OBJ-NEXT:       Type: int (0x74)
; OBJ-NEXT:       DisplayName: last
; OBJ-NEXT:       LinkageName: ?last@@3HA
; OBJ-NEXT:     }
; OBJ-NEXT:     DataSym {
; OBJ-NEXT:       Kind: S_LDATA32 (0x110C)
; OBJ-NEXT:       DataOffset: ?globalStatic@foo@@3HA+0x0
; OBJ-NEXT:       Type: int (0x74)
; OBJ-NEXT:       DisplayName: foo::globalStatic
; OBJ-NEXT:       LinkageName: ?globalStatic@foo@@3HA
; OBJ-NEXT:     }
; OBJ-NEXT:     ThreadLocalDataSym {
; OBJ-NEXT:       Kind: S_LTHREAD32 (0x1112)
; OBJ-NEXT:       DataOffset: ?staticTLS@foo@@3HA+0x0
; OBJ-NEXT:       Type: int (0x74)
; OBJ-NEXT:       DisplayName: foo::staticTLS
; OBJ-NEXT:       LinkageName: ?staticTLS@foo@@3HA
; OBJ-NEXT:     }

; OBJ:    GlobalData {
; OBJ-NEXT:      Kind: S_GDATA32 (0x110D)
; OBJ-LABEL:      DataOffset: ?comdat@?$A@X@@2HB+0x0
; OBJ-NEXT:      Type: const int (0x1003)
; OBJ-NEXT:      DisplayName: comdat
; OBJ-NEXT:      LinkageName: ?comdat@?$A@X@@2HB

; YAML-LABEL:  - Name:            '.debug$S'
; YAML:    Subsections:
; YAML:      - !Symbols
; YAML:        Records:
; YAML:          - Kind:            S_COMPILE3
; YAML:            Compile3Sym:

; YAML:      - !Symbols
; YAML-NEXT:        Records:
; YAML-LABEL:        - Kind:            S_GDATA32
; YAML-NEXT:            DataSym:
; YAML-NOT: Segment
; YAML-NEXT:              Type:            116
; YAML-NOT: Segment
; YAML-NEXT:              DisplayName:     first
; YAML-NOT: Segment
; YAML-NEXT:          - Kind:            S_GTHREAD32
; YAML-NEXT:            ThreadLocalDataSym:
; YAML-NEXT:              Type:            4100
; YAML-NEXT:              DisplayName:     middle
; YAML-NEXT:          - Kind:            S_GTHREAD32
; YAML-NEXT:            ThreadLocalDataSym:
; YAML-NEXT:              Type:            116
; YAML-NEXT:              DisplayName:     'foo::globalTLS'
; YAML-NEXT:          - Kind:            S_GDATA32
; YAML-NEXT:            DataSym:
; YAML-NOT: Segment
; YAML-NEXT:              Type:            116
; YAML-NOT: Segment
; YAML-NEXT:              DisplayName:     'foo::justGlobal'
; YAML-NOT: Segment
; YAML-NEXT:          - Kind:            S_GDATA32
; YAML-NEXT:            DataSym:
; YAML-NEXT:              Type:            116
; YAML-NEXT:              DisplayName:     last
; YAML-NEXT:          - Kind:            S_LDATA32
; YAML-NEXT:            DataSym:
; YAML-NEXT:              Type:            116
; YAML-NEXT:              DisplayName:     'foo::globalStatic'
; YAML-NEXT:          - Kind:            S_LTHREAD32
; YAML-NEXT:            ThreadLocalDataSym:
; YAML-NEXT:              Type:            116
; YAML-NEXT:              DisplayName:     'foo::staticTLS'

; ModuleID = 'a.cpp'
source_filename = "a.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.25.28614"

$"?comdat@?$A@X@@2HB" = comdat any

@"?first@@3HA" = dso_local global i32 0, align 4, !dbg !0
@"?comdat@?$A@X@@2HB" = linkonce_odr dso_local constant i32 3, comdat, align 4, !dbg !6
@"?middle@@3PEBHEB" = dso_local thread_local global i32* @"?comdat@?$A@X@@2HB", align 8, !dbg !15
@"?globalTLS@foo@@3HA" = dso_local thread_local global i32 4, align 4, !dbg !18
@"?justGlobal@foo@@3HA" = dso_local global i32 6, align 4, !dbg !21
@"?last@@3HA" = dso_local global i32 0, align 4, !dbg !23
@"?globalStatic@foo@@3HA" = internal global i32 7, align 4, !dbg !25
@"?staticTLS@foo@@3HA" = internal thread_local global i32 5, align 4, !dbg !27

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @"?bar@@YAHXZ"() #0 !dbg !36 {
entry:
  %0 = load i32, i32* @"?globalStatic@foo@@3HA", align 4, !dbg !39
  %1 = load i32, i32* @"?globalTLS@foo@@3HA", align 4, !dbg !39
  %add = add nsw i32 %0, %1, !dbg !39
  %2 = load i32, i32* @"?staticTLS@foo@@3HA", align 4, !dbg !39
  %add1 = add nsw i32 %add, %2, !dbg !39
  ret i32 %add1, !dbg !39
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!2}
!llvm.linker.options = !{!29, !30}
!llvm.module.flags = !{!31, !32, !33, !34}
!llvm.ident = !{!35}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "first", linkageName: "?first@@3HA", scope: !2, file: !3, line: 1, type: !9, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project.git f5b1301ce8575f6d82e87031a1a5485c33637a93)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "a.cpp", directory: "F:\\llvm-project\\__test", checksumkind: CSK_MD5, checksum: "65c2a7701cffb7a2e8d4caf1cc24caa7")
!4 = !{}
!5 = !{!0, !6, !15, !18, !21, !23, !25, !27}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "comdat", linkageName: "?comdat@?$A@X@@2HB", scope: !2, file: !3, line: 2, type: !8, isLocal: false, isDefinition: true, declaration: !10)
!8 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !9)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "comdat", scope: !11, file: !3, line: 2, baseType: !8, flags: DIFlagStaticMember, extraData: i32 3)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A<void>", file: !3, line: 2, size: 8, flags: DIFlagTypePassByValue, elements: !12, templateParams: !13, identifier: ".?AU?$A@X@@")
!12 = !{!10}
!13 = !{!14}
!14 = !DITemplateTypeParameter(name: "T", type: null)
!15 = !DIGlobalVariableExpression(var: !16, expr: !DIExpression())
!16 = distinct !DIGlobalVariable(name: "middle", linkageName: "?middle@@3PEBHEB", scope: !2, file: !3, line: 3, type: !17, isLocal: false, isDefinition: true)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!18 = !DIGlobalVariableExpression(var: !19, expr: !DIExpression())
!19 = distinct !DIGlobalVariable(name: "globalTLS", linkageName: "?globalTLS@foo@@3HA", scope: !20, file: !3, line: 5, type: !9, isLocal: false, isDefinition: true)
!20 = !DINamespace(name: "foo", scope: null)
!21 = !DIGlobalVariableExpression(var: !22, expr: !DIExpression())
!22 = distinct !DIGlobalVariable(name: "justGlobal", linkageName: "?justGlobal@foo@@3HA", scope: !20, file: !3, line: 7, type: !9, isLocal: false, isDefinition: true)
!23 = !DIGlobalVariableExpression(var: !24, expr: !DIExpression())
!24 = distinct !DIGlobalVariable(name: "last", linkageName: "?last@@3HA", scope: !2, file: !3, line: 10, type: !9, isLocal: false, isDefinition: true)
!25 = !DIGlobalVariableExpression(var: !26, expr: !DIExpression())
!26 = distinct !DIGlobalVariable(name: "globalStatic", linkageName: "?globalStatic@foo@@3HA", scope: !20, file: !3, line: 8, type: !9, isLocal: true, isDefinition: true)
!27 = !DIGlobalVariableExpression(var: !28, expr: !DIExpression())
!28 = distinct !DIGlobalVariable(name: "staticTLS", linkageName: "?staticTLS@foo@@3HA", scope: !20, file: !3, line: 6, type: !9, isLocal: true, isDefinition: true)
!29 = !{!"/DEFAULTLIB:libcmt.lib"}
!30 = !{!"/DEFAULTLIB:oldnames.lib"}
!31 = !{i32 2, !"CodeView", i32 1}
!32 = !{i32 2, !"Debug Info Version", i32 3}
!33 = !{i32 1, !"wchar_size", i32 2}
!34 = !{i32 7, !"PIC Level", i32 2}
!35 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git f5b1301ce8575f6d82e87031a1a5485c33637a93)"}
!36 = distinct !DISubprogram(name: "bar", linkageName: "?bar@@YAHXZ", scope: !3, file: !3, line: 11, type: !37, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!37 = !DISubroutineType(types: !38)
!38 = !{!9}
!39 = !DILocation(line: 12, scope: !36)
