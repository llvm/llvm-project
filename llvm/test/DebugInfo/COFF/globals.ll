; RUN: llc < %s | FileCheck %s --check-prefix=ASM
; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s --check-prefix=OBJ
; RUN: llc < %s | llvm-mc -filetype=obj --triple=x86_64-windows | llvm-readobj - --codeview | FileCheck %s --check-prefix=OBJ
; RUN: llc < %s -filetype=obj | obj2yaml | FileCheck %s --check-prefixes=YAML,YAML-STDOUT
; RUN: llc < %s -filetype=obj -o %t
; RUN: obj2yaml < %t | FileCheck %s --check-prefixes=YAML,YAML-FILE

; Repeat with experimental debuginfo iterators:
; RUN: llc --try-experimental-debuginfo-iterators < %s | FileCheck %s --check-prefix=ASM
; RUN: llc --try-experimental-debuginfo-iterators < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s --check-prefix=OBJ
; RUN: llc --try-experimental-debuginfo-iterators < %s | llvm-mc -filetype=obj --triple=x86_64-windows | llvm-readobj - --codeview | FileCheck %s --check-prefix=OBJ
; RUN: llc --try-experimental-debuginfo-iterators < %s -filetype=obj | obj2yaml | FileCheck %s --check-prefixes=YAML,YAML-STDOUT
; RUN: llc --try-experimental-debuginfo-iterators < %s -filetype=obj -o %t
; RUN: obj2yaml < %t | FileCheck %s --check-prefixes=YAML,YAML-FILE

; C++ source to regenerate:
; $ cat a.cpp
; int first;
; 
; template <typename T> struct A { static const int comdat = 3; };
; 
; thread_local const int *middle = &A<void>::comdat;
; 
; namespace foo {
; thread_local int globalTLS = 4;
; static thread_local int staticTLS = 5;
; int justGlobal = 6;
; static int globalStatic = 7;
; constexpr int constExpr = 8;
; const int constVal = 9;
; 
; struct Data {
;   inline static thread_local int DataStaticTLS = 11;
;   int DataGlobal = 12;
;   inline static int DataStatic = 13;
;   constexpr static int DataConstExpr = 14;
;   const int DataConstVal = 15;
; };
; } // namespace foo
; 
; int last;
; 
; int bar() {
;   struct Local {
;     int LocalGlobal = 12;
;     const int LocalConstVal = 15;
;   };
;   foo::Data D;
;   Local L;
;   return foo::globalStatic + foo::globalTLS + foo::staticTLS + foo::justGlobal +
;          foo::globalStatic + foo::constExpr + foo::constVal + D.DataStaticTLS +
;          D.DataGlobal + D.DataStatic + D.DataConstExpr + D.DataConstVal +
;          L.LocalGlobal + L.LocalConstVal;
; }
; 
; $ clang-cl a.cpp /c /GS- /Z7 /GR- /std:c++17 /clang:-S /clang:-emit-llvm

; ASM:        .section        .debug$S,"dr"
; ASM-NEXT:   .p2align        2, 0x0
; ASM-NEXT:   .long   4                       # Debug section magic

; ASM:        .short  4365                    # Record kind: S_GDATA32
; ASM-NEXT:   .long   116                     # Type
; ASM-NEXT:   .secrel32       "?first@@3HA"   # DataOffset
; ASM-NEXT:   .secidx "?first@@3HA"           # Segment
; ASM-NEXT:   .asciz  "first"                 # Name
; ASM-NEXT:   .p2align        2, 0x0

; ASM:        .short  4371                    # Record kind: S_GTHREAD32
; ASM-NEXT:   .long   4117                    # Type
; ASM-NEXT:   .secrel32       "?middle@@3PEBHEB" # DataOffset
; ASM-NEXT:   .secidx "?middle@@3PEBHEB"      # Segment
; ASM-NEXT:   .asciz  "middle"                # Name
; ASM-NEXT:   .p2align        2, 0x0

; ASM:        .short  4371                    # Record kind: S_GTHREAD32
; ASM-NEXT:   .long   116                     # Type
; ASM-NEXT:   .secrel32       "?globalTLS@foo@@3HA" # DataOffset
; ASM-NEXT:   .secidx "?globalTLS@foo@@3HA"   # Segment
; ASM-NEXT:   .asciz  "foo::globalTLS"        # Name
; ASM-NEXT:   .p2align        2, 0x0

; ASM:        .short  4365                    # Record kind: S_GDATA32
; ASM-NEXT:   .long   116                     # Type
; ASM-NEXT:   .secrel32       "?justGlobal@foo@@3HA" # DataOffset
; ASM-NEXT:   .secidx "?justGlobal@foo@@3HA"  # Segment
; ASM-NEXT:   .asciz  "foo::justGlobal"       # Name
; ASM-NEXT:   .p2align        2, 0x0

; ASM:        .short  4365                    # Record kind: S_GDATA32
; ASM-NEXT:   .long   116                     # Type
; ASM-NEXT:   .secrel32       "?last@@3HA"    # DataOffset
; ASM-NEXT:   .secidx "?last@@3HA"            # Segment
; ASM-NEXT:   .asciz  "last"                  # Name
; ASM-NEXT:   .p2align        2, 0x0

; ASM:        .short	4359                    # Record kind: S_CONSTANT
; ASM-NEXT:	  .long	4100                    # Type
; ASM-NEXT:   .byte	0x08, 0x00              # Value
; ASM-NEXT:	  .asciz	"foo::constExpr"        # Name
; ASM-NEXT:   .p2align	2, 0x0

; ASM:        .short	4359                    # Record kind: S_CONSTANT
; ASM-NEXT:   .long	4100                    # Type
; ASM-NEXT:   .byte	0x09, 0x00              # Value
; ASM-NEXT:   .asciz	"foo::constVal"         # Name
; ASM-NEXT:   .p2align	2, 0x0

; ASM:        .short	4359                    # Record kind: S_CONSTANT
; ASM-NEXT:   .long	4100                    # Type
; ASM-NEXT:   .byte	0x0e, 0x00              # Value
; ASM-NEXT:   .asciz	"foo::Data::DataConstExpr" # Name
; ASM-NEXT:   .p2align	2, 0x0

; ASM:        .short  4364                    # Record kind: S_LDATA32
; ASM-NEXT:   .long   116                     # Type
; ASM-NEXT:   .secrel32       "?globalStatic@foo@@3HA" # DataOffset
; ASM-NEXT:   .secidx "?globalStatic@foo@@3HA" # Segment
; ASM-NEXT:   .asciz  "foo::globalStatic"     # Name
; ASM-NEXT:   .p2align        2, 0x0

; ASM:        .short  4370                    # Record kind: S_LTHREAD32
; ASM-NEXT:   .long   116                     # Type
; ASM-NEXT:   .secrel32       "?staticTLS@foo@@3HA" # DataOffset
; ASM-NEXT:   .secidx "?staticTLS@foo@@3HA"   # Segment
; ASM-NEXT:   .asciz  "foo::staticTLS"        # Name
; ASM-NEXT:   .p2align        2, 0x0

; ASM:        .section        .debug$S,"dr",associative,"?comdat@?$A@X@@2HB"
; ASM-NEXT:   .p2align        2, 0x0
; ASM-NEXT:   .long   4                       # Debug section magic

; ASM:        .short  4365                    # Record kind: S_GDATA32
; ASM-NEXT:   .long   4100                    # Type
; ASM-NEXT:   .secrel32       "?comdat@?$A@X@@2HB" # DataOffset
; ASM-NEXT:   .secidx "?comdat@?$A@X@@2HB"    # Segment
; ASM-NEXT:   .asciz  "A<void>::comdat"       # Name

; ASM:	      .section	.debug$S,"dr",associative,"?DataStaticTLS@Data@foo@@2HA"
; ASM-NEXT:	  .p2align	2, 0x0          # Symbol subsection for ?DataStaticTLS@Data@foo@@2HA

; ASM:	      .short	4371                    # Record kind: S_GTHREAD32
; ASM-NEXT:	  .long	116                     # Type
; ASM-NEXT:	  .secrel32	"?DataStaticTLS@Data@foo@@2HA" # DataOffset
; ASM-NEXT:	  .secidx	"?DataStaticTLS@Data@foo@@2HA" # Segment
; ASM-NEXT:	  .asciz	"foo::Data::DataStaticTLS"         # Name
; ASM-NEXT:   .p2align	2, 0x0

; ASM:        .section        .debug$S,"dr",associative,"?DataStatic@Data@foo@@2HA"
; ASM-NEXT:   .p2align        2, 0x0          # Symbol subsection for ?DataStatic@Data@foo@@2HA

; ASM:        .short  4365                    # Record kind: S_GDATA32
; ASM-NEXT:   .long   116                     # Type
; ASM-NEXT:   .secrel32       "?DataStatic@Data@foo@@2HA" # DataOffset
; ASM-NEXT:   .secidx "?DataStatic@Data@foo@@2HA" # Segment
; ASM-NEXT:   .asciz  "foo::Data::DataStatic" # Name
; ASM-NEXT:   .p2align        2, 0x0

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
; OBJ-NEXT:       Type: const int* (0x1015)
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
; OBJ-NEXT:    ConstantSym {
; OBJ-NEXT:      Kind: S_CONSTANT (0x1107)
; OBJ-NEXT:      Type: const int (0x1004)
; OBJ-NEXT:      Value: 8
; OBJ-NEXT:      Name: foo::constExpr
; OBJ-NEXT:    }
; OBJ-NEXT:    ConstantSym {
; OBJ-NEXT:      Kind: S_CONSTANT (0x1107)
; OBJ-NEXT:      Type: const int (0x1004)
; OBJ-NEXT:      Value: 9
; OBJ-NEXT:      Name: foo::constVal
; OBJ-NEXT:    }
; OBJ-NEXT:    ConstantSym {
; OBJ-NEXT:      Kind: S_CONSTANT (0x1107)
; OBJ-NEXT:      Type: const int (0x1004)
; OBJ-NEXT:      Value: 14
; OBJ-NEXT:      Name: foo::Data::DataConstExpr
; OBJ-NEXT:    }
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
; OBJ-NEXT:      Type: const int (0x1004)
; OBJ-NEXT:      DisplayName: A<void>::comdat
; OBJ-NEXT:      LinkageName: ?comdat@?$A@X@@2HB

; OBJ:    GlobalTLS {
; OBJ-NEXT:      Kind: S_GTHREAD32 (0x1113)
; OBJ-NEXT:      DataOffset: ?DataStaticTLS@Data@foo@@2HA+0x0
; OBJ-NEXT:      Type: int (0x74)
; OBJ-NEXT:      DisplayName: foo::Data::DataStaticTLS
; OBJ-NEXT:      LinkageName: ?DataStaticTLS@Data@foo@@2HA
; OBJ-NEXT:    }

; OBJ:    GlobalData {
; OBJ-NEXT:      Kind: S_GDATA32 (0x110D)
; OBJ-NEXT:      DataOffset: ?DataStatic@Data@foo@@2HA+0x0
; OBJ-NEXT:      Type: int (0x74)
; OBJ-NEXT:      DisplayName: foo::Data::DataStatic
; OBJ-NEXT:      LinkageName: ?DataStatic@Data@foo@@2HA
; OBJ-NEXT:    }

; YAML-LABEL:  - Name:            '.debug$S'
; YAML:    Subsections:
; YAML:      - !Symbols
; YAML:        Records:
; YAML:          - Kind:            S_OBJNAME
; YAML:            ObjNameSym:
; YAML:               Signature:       0
; YAML-STDOUT:        ObjectName:      ''
; YAML-FILE:          ObjectName:      '{{.*}}'
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
; YAML-NEXT:              Type:            4117
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
; YAML-NEXT:          - Kind:            S_CONSTANT
; YAML-NEXT:            ConstantSym:
; YAML-NEXT:              Type:            4100
; YAML-NEXT:              Value:           8
; YAML-NEXT:              Name:            'foo::constExpr'
; YAML-NEXT:          - Kind:            S_CONSTANT
; YAML-NEXT:            ConstantSym:
; YAML-NEXT:              Type:            4100
; YAML-NEXT:              Value:           9
; YAML-NEXT:              Name:            'foo::constVal'
; YAML-NEXT:          - Kind:            S_CONSTANT
; YAML-NEXT:            ConstantSym:
; YAML-NEXT:              Type:            4100
; YAML-NEXT:              Value:           14
; YAML-NEXT:              Name:            'foo::Data::DataConstExpr'
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

%"struct.foo::Data" = type { i32, i32 }
%struct.Local = type { i32, i32 }

$"??0Data@foo@@QEAA@XZ" = comdat any

$"?comdat@?$A@X@@2HB" = comdat any

$"?DataStaticTLS@Data@foo@@2HA" = comdat any

$"?DataStatic@Data@foo@@2HA" = comdat any

@"?first@@3HA" = dso_local global i32 0, align 4, !dbg !0
@"?comdat@?$A@X@@2HB" = linkonce_odr dso_local constant i32 3, comdat, align 4, !dbg !17
@"?middle@@3PEBHEB" = dso_local thread_local global ptr @"?comdat@?$A@X@@2HB", align 8, !dbg !24
@"?globalTLS@foo@@3HA" = dso_local thread_local global i32 4, align 4, !dbg !27
@"?justGlobal@foo@@3HA" = dso_local global i32 6, align 4, !dbg !29
@"?last@@3HA" = dso_local global i32 0, align 4, !dbg !31
@"?globalStatic@foo@@3HA" = internal global i32 7, align 4, !dbg !43
@"?staticTLS@foo@@3HA" = internal thread_local global i32 5, align 4, !dbg !45
@"?DataStaticTLS@Data@foo@@2HA" = linkonce_odr dso_local thread_local global i32 11, comdat, align 4, !dbg !37
@"?DataStatic@Data@foo@@2HA" = linkonce_odr dso_local global i32 13, comdat, align 4, !dbg !39

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @"?bar@@YAHXZ"() #0 !dbg !54 {
entry:
  %D = alloca %"struct.foo::Data", align 4
  %L = alloca %struct.Local, align 4
  call void @llvm.dbg.declare(metadata ptr %D, metadata !57, metadata !DIExpression()), !dbg !58
  %call = call ptr @"??0Data@foo@@QEAA@XZ"(ptr %D) #2, !dbg !58
  call void @llvm.dbg.declare(metadata ptr %L, metadata !59, metadata !DIExpression()), !dbg !64
  %call1 = call ptr @"??0Local@?1??bar@@YAHXZ@QEAA@XZ"(ptr %L) #2, !dbg !64
  %0 = load i32, ptr @"?globalStatic@foo@@3HA", align 4, !dbg !65
  %1 = load i32, ptr @"?globalTLS@foo@@3HA", align 4, !dbg !65
  %add = add nsw i32 %0, %1, !dbg !65
  %2 = load i32, ptr @"?staticTLS@foo@@3HA", align 4, !dbg !65
  %add2 = add nsw i32 %add, %2, !dbg !65
  %3 = load i32, ptr @"?justGlobal@foo@@3HA", align 4, !dbg !65
  %add3 = add nsw i32 %add2, %3, !dbg !65
  %4 = load i32, ptr @"?globalStatic@foo@@3HA", align 4, !dbg !65
  %add4 = add nsw i32 %add3, %4, !dbg !65
  %add5 = add nsw i32 %add4, 8, !dbg !65
  %add6 = add nsw i32 %add5, 9, !dbg !65
  %5 = load i32, ptr @"?DataStaticTLS@Data@foo@@2HA", align 4, !dbg !65
  %add7 = add nsw i32 %add6, %5, !dbg !65
  %6 = load i32, ptr %D, align 4, !dbg !65
  %add8 = add nsw i32 %add7, %6, !dbg !65
  %7 = load i32, ptr @"?DataStatic@Data@foo@@2HA", align 4, !dbg !65
  %add9 = add nsw i32 %add8, %7, !dbg !65
  %add10 = add nsw i32 %add9, 14, !dbg !65
  %DataConstVal = getelementptr inbounds %"struct.foo::Data", ptr %D, i32 0, i32 1, !dbg !65
  %8 = load i32, ptr %DataConstVal, align 4, !dbg !65
  %add11 = add nsw i32 %add10, %8, !dbg !65
  %9 = load i32, ptr %L, align 4, !dbg !65
  %add12 = add nsw i32 %add11, %9, !dbg !65
  %LocalConstVal = getelementptr inbounds %struct.Local, ptr %L, i32 0, i32 1, !dbg !65
  %10 = load i32, ptr %LocalConstVal, align 4, !dbg !65
  %add13 = add nsw i32 %add12, %10, !dbg !65
  ret i32 %add13, !dbg !65
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local ptr @"??0Data@foo@@QEAA@XZ"(ptr returned %this) unnamed_addr #0 comdat align 2 !dbg !66 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %this.addr, metadata !71, metadata !DIExpression()), !dbg !73
  %this1 = load ptr, ptr %this.addr, align 8
  store i32 12, ptr %this1, align 4, !dbg !74
  %DataConstVal = getelementptr inbounds %"struct.foo::Data", ptr %this1, i32 0, i32 1, !dbg !74
  store i32 15, ptr %DataConstVal, align 4, !dbg !74
  ret ptr %this1, !dbg !74
}

; Function Attrs: noinline nounwind optnone uwtable
define internal ptr @"??0Local@?1??bar@@YAHXZ@QEAA@XZ"(ptr returned %this) unnamed_addr #0 align 2 !dbg !75 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %this.addr, metadata !80, metadata !DIExpression()), !dbg !82
  %this1 = load ptr, ptr %this.addr, align 8
  store i32 12, ptr %this1, align 4, !dbg !83
  %LocalConstVal = getelementptr inbounds %struct.Local, ptr %this1, i32 0, i32 1, !dbg !83
  store i32 15, ptr %LocalConstVal, align 4, !dbg !83
  ret ptr %this1, !dbg !83
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!2}
!llvm.linker.options = !{!47, !48}
!llvm.module.flags = !{!49, !50, !51, !52}
!llvm.ident = !{!53}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "first", linkageName: "?first@@3HA", scope: !2, file: !3, line: 1, type: !10, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project.git 202f144bffd0be254a829924195e1b8ebabcbb79)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !16, nameTableKind: None)
!3 = !DIFile(filename: "a.cpp", directory: "F:\\llvm-project\\__test", checksumkind: CSK_MD5, checksum: "ae8137877dbd6fb10cfa1fc9ea4a39ca")
!4 = !{}
!5 = !{!6}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Data", scope: !7, file: !3, line: 15, size: 64, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !8, identifier: ".?AUData@foo@@")
!7 = !DINamespace(name: "foo", scope: null)
!8 = !{!9, !11, !12, !13, !15}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "DataStaticTLS", scope: !6, file: !3, line: 16, baseType: !10, flags: DIFlagStaticMember, extraData: i32 11)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_member, name: "DataGlobal", scope: !6, file: !3, line: 17, baseType: !10, size: 32)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "DataStatic", scope: !6, file: !3, line: 18, baseType: !10, flags: DIFlagStaticMember, extraData: i32 13)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "DataConstExpr", scope: !6, file: !3, line: 19, baseType: !14, flags: DIFlagStaticMember, extraData: i32 14)
!14 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !10)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "DataConstVal", scope: !6, file: !3, line: 20, baseType: !14, size: 32, offset: 32)
!16 = !{!0, !17, !24, !27, !29, !31, !33, !35, !37, !39, !41, !43, !45}
!17 = !DIGlobalVariableExpression(var: !18, expr: !DIExpression())
!18 = distinct !DIGlobalVariable(name: "comdat", linkageName: "?comdat@?$A@X@@2HB", scope: !2, file: !3, line: 3, type: !14, isLocal: false, isDefinition: true, declaration: !19)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "comdat", scope: !20, file: !3, line: 3, baseType: !14, flags: DIFlagStaticMember, extraData: i32 3)
!20 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A<void>", file: !3, line: 3, size: 8, flags: DIFlagTypePassByValue, elements: !21, templateParams: !22, identifier: ".?AU?$A@X@@")
!21 = !{!19}
!22 = !{!23}
!23 = !DITemplateTypeParameter(name: "T", type: null)
!24 = !DIGlobalVariableExpression(var: !25, expr: !DIExpression())
!25 = distinct !DIGlobalVariable(name: "middle", linkageName: "?middle@@3PEBHEB", scope: !2, file: !3, line: 5, type: !26, isLocal: false, isDefinition: true)
!26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!27 = !DIGlobalVariableExpression(var: !28, expr: !DIExpression())
!28 = distinct !DIGlobalVariable(name: "globalTLS", linkageName: "?globalTLS@foo@@3HA", scope: !7, file: !3, line: 8, type: !10, isLocal: false, isDefinition: true)
!29 = !DIGlobalVariableExpression(var: !30, expr: !DIExpression())
!30 = distinct !DIGlobalVariable(name: "justGlobal", linkageName: "?justGlobal@foo@@3HA", scope: !7, file: !3, line: 10, type: !10, isLocal: false, isDefinition: true)
!31 = !DIGlobalVariableExpression(var: !32, expr: !DIExpression())
!32 = distinct !DIGlobalVariable(name: "last", linkageName: "?last@@3HA", scope: !2, file: !3, line: 24, type: !10, isLocal: false, isDefinition: true)
!33 = !DIGlobalVariableExpression(var: !34, expr: !DIExpression(DW_OP_constu, 8, DW_OP_stack_value))
!34 = distinct !DIGlobalVariable(name: "constExpr", scope: !7, file: !3, line: 12, type: !14, isLocal: true, isDefinition: true)
!35 = !DIGlobalVariableExpression(var: !36, expr: !DIExpression(DW_OP_constu, 9, DW_OP_stack_value))
!36 = distinct !DIGlobalVariable(name: "constVal", scope: !7, file: !3, line: 13, type: !14, isLocal: true, isDefinition: true)
!37 = !DIGlobalVariableExpression(var: !38, expr: !DIExpression())
!38 = distinct !DIGlobalVariable(name: "DataStaticTLS", linkageName: "?DataStaticTLS@Data@foo@@2HA", scope: !2, file: !3, line: 16, type: !10, isLocal: false, isDefinition: true, declaration: !9)
!39 = !DIGlobalVariableExpression(var: !40, expr: !DIExpression())
!40 = distinct !DIGlobalVariable(name: "DataStatic", linkageName: "?DataStatic@Data@foo@@2HA", scope: !2, file: !3, line: 18, type: !10, isLocal: false, isDefinition: true, declaration: !12)
!41 = !DIGlobalVariableExpression(var: !42, expr: !DIExpression(DW_OP_constu, 14, DW_OP_stack_value))
!42 = distinct !DIGlobalVariable(name: "DataConstExpr", scope: !2, file: !3, line: 19, type: !14, isLocal: true, isDefinition: true, declaration: !13)
!43 = !DIGlobalVariableExpression(var: !44, expr: !DIExpression())
!44 = distinct !DIGlobalVariable(name: "globalStatic", linkageName: "?globalStatic@foo@@3HA", scope: !7, file: !3, line: 11, type: !10, isLocal: true, isDefinition: true)
!45 = !DIGlobalVariableExpression(var: !46, expr: !DIExpression())
!46 = distinct !DIGlobalVariable(name: "staticTLS", linkageName: "?staticTLS@foo@@3HA", scope: !7, file: !3, line: 9, type: !10, isLocal: true, isDefinition: true)
!47 = !{!"/DEFAULTLIB:libcmt.lib"}
!48 = !{!"/DEFAULTLIB:oldnames.lib"}
!49 = !{i32 2, !"CodeView", i32 1}
!50 = !{i32 2, !"Debug Info Version", i32 3}
!51 = !{i32 1, !"wchar_size", i32 2}
!52 = !{i32 7, !"PIC Level", i32 2}
!53 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 202f144bffd0be254a829924195e1b8ebabcbb79)"}
!54 = distinct !DISubprogram(name: "bar", linkageName: "?bar@@YAHXZ", scope: !3, file: !3, line: 26, type: !55, scopeLine: 26, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!55 = !DISubroutineType(types: !56)
!56 = !{!10}
!57 = !DILocalVariable(name: "D", scope: !54, file: !3, line: 31, type: !6)
!58 = !DILocation(line: 31, scope: !54)
!59 = !DILocalVariable(name: "L", scope: !54, file: !3, line: 32, type: !60)
!60 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Local", scope: !54, file: !3, line: 27, size: 64, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !61, identifier: ".?AULocal@?1??bar@@YAHXZ@")
!61 = !{!62, !63}
!62 = !DIDerivedType(tag: DW_TAG_member, name: "LocalGlobal", scope: !60, file: !3, line: 28, baseType: !10, size: 32)
!63 = !DIDerivedType(tag: DW_TAG_member, name: "LocalConstVal", scope: !60, file: !3, line: 29, baseType: !14, size: 32, offset: 32)
!64 = !DILocation(line: 32, scope: !54)
!65 = !DILocation(line: 33, scope: !54)
!66 = distinct !DISubprogram(name: "Data", linkageName: "??0Data@foo@@QEAA@XZ", scope: !6, file: !3, line: 15, type: !67, scopeLine: 15, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, declaration: !70, retainedNodes: !4)
!67 = !DISubroutineType(types: !68)
!68 = !{null, !69}
!69 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!70 = !DISubprogram(name: "Data", scope: !6, type: !67, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: 0)
!71 = !DILocalVariable(name: "this", arg: 1, scope: !66, type: !72, flags: DIFlagArtificial | DIFlagObjectPointer)
!72 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!73 = !DILocation(line: 0, scope: !66)
!74 = !DILocation(line: 15, scope: !66)
!75 = distinct !DISubprogram(name: "Local", linkageName: "??0Local@?1??bar@@YAHXZ@QEAA@XZ", scope: !60, file: !3, line: 27, type: !76, scopeLine: 27, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, declaration: !79, retainedNodes: !4)
!76 = !DISubroutineType(types: !77)
!77 = !{null, !78}
!78 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !60, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!79 = !DISubprogram(name: "Local", scope: !60, type: !76, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: 0)
!80 = !DILocalVariable(name: "this", arg: 1, scope: !75, type: !81, flags: DIFlagArtificial | DIFlagObjectPointer)
!81 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !60, size: 64)
!82 = !DILocation(line: 0, scope: !75)
!83 = !DILocation(line: 27, scope: !75)
