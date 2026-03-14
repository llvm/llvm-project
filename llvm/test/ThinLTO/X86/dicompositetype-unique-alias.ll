; RUN: opt -module-summary -o %t1.bc %s
; RUN: opt -module-summary -o %t2.bc %S/Inputs/dicompositetype-unique-alias.ll
; RUN: llvm-lto --thinlto-action=run %t1.bc %t2.bc -thinlto-save-temps=%t3.
; RUN: llvm-dis %t3.0.3.imported.bc -o - | FileCheck %s
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t --save-temps \
; RUN: -r %t1.bc,_ZN1CD2Ev,pl \
; RUN: -r %t1.bc,_ZN4CFVSD1Ev,l \
; RUN: -r %t1.bc,_ZN4CFVSD2Ev,l \
; RUN: -r %t1.bc,_Z3Getv,l \
; RUN: -r %t2.bc,_ZN4CFVSD1Ev,pl \
; RUN: -r %t2.bc,_ZN4CFVSD2Ev,pl \
; RUN: -r %t2.bc,_Z3Getv,l
; RUN: llvm-dis %t.1.3.import.bc -o - | FileCheck %s

; Only llvm-lto2 adds the dso_local keyword, hence the {{.*}}
; CHECK: define available_externally{{.*}} void @_ZN4CFVSD1Ev

; Confirm that we only have a single DICompositeType after importing
; both an alias and its aliasee, since ODR Type Uniquing is enabled.
; CHECK: DICompositeType
; CHECK-NOT: DICompositeType

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-scei-ps4"

%class.C = type <{ ptr, %class.A, %struct.CFVS, [6 x i8] }>
%class.A = type { %struct.Vec }
%struct.Vec = type { i8 }
%struct.CFVS = type { %struct.Vec }
%struct.S = type { i8 }

define void @_ZN1CD2Ev(ptr %this) unnamed_addr align 2 {
entry:
  %this.addr = alloca ptr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  %m = getelementptr inbounds %class.C, ptr %this1, i32 0, i32 2
  call void @_ZN4CFVSD1Ev(ptr %m), !dbg !50
  call void @_ZN4CFVSD2Ev(ptr %m), !dbg !50
  ret void
}

declare void @_ZN4CFVSD1Ev(ptr) unnamed_addr
declare void @_ZN4CFVSD2Ev(ptr) unnamed_addr

declare dereferenceable(1) ptr @_Z3Getv()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 6.0.0 (trunk 321360) (llvm/trunk 321359)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "bz188598-a.cpp", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!8 = distinct !DISubprogram(name: "~C", linkageName: "_ZN1CD2Ev", scope: !1, file: !1, line: 9, type: !47, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!47 = !DISubroutineType(types: !48)
!48 = !{!55}
!50 = !DILocation(line: 9, scope: !51)
!51 = distinct !DILexicalBlock(scope: !8, file: !1, line: 9)
!55 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
