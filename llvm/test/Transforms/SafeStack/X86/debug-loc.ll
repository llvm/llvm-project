; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s

; Test debug location for the local variables moved onto the unsafe stack.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { [100 x i8] }

; Function Attrs: safestack uwtable
define void @f(%struct.S* byval align 8 %zzz) #0 !dbg !12 {
; CHECK: define void @f

entry:
; CHECK: %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr

  %xxx = alloca %struct.S, align 1
  call void @llvm.dbg.declare(metadata %struct.S* %zzz, metadata !18, metadata !19), !dbg !20
  call void @llvm.dbg.declare(metadata %struct.S* %xxx, metadata !21, metadata !19), !dbg !22

; dbg.declare for %zzz and %xxx are gone; replaced with dbg.declare based off the unsafe stack pointer
; CHECK-NOT: call void @llvm.dbg.declare
; CHECK: call void @llvm.dbg.declare(metadata i8* %[[USP]], metadata ![[VAR_ARG:.*]], metadata !DIExpression(DW_OP_constu, 104, DW_OP_minus))
; CHECK-NOT: call void @llvm.dbg.declare
; CHECK: call void @llvm.dbg.declare(metadata i8* %[[USP]], metadata ![[VAR_LOCAL:.*]], metadata !DIExpression(DW_OP_constu, 208, DW_OP_minus))
; CHECK-NOT: call void @llvm.dbg.declare

  call void @Capture(%struct.S* %zzz), !dbg !23
  call void @Capture(%struct.S* %xxx), !dbg !24

; dbg.declare appears before the first use
; CHECK:   call void @Capture
; CHECK:   call void @Capture

  ret void, !dbg !25
}

; CHECK-DAG: ![[VAR_ARG]] = !DILocalVariable(name: "zzz"
; 100 aligned up to 8

; CHECK-DAG: ![[VAR_LOCAL]] = !DILocalVariable(name: "xxx"

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @Capture(%struct.S*) #2

attributes #0 = { safestack uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!15, !16}
!llvm.ident = !{!17}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 254019) (llvm/trunk 254036)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "../llvm/2.cc", directory: "/code/build-llvm")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !1, line: 4, size: 800, align: 8, elements: !5, identifier: "_ZTS1S")
!5 = !{!6}
!6 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !4, file: !1, line: 5, baseType: !7, size: 800, align: 8)
!7 = !DICompositeType(tag: DW_TAG_array_type, baseType: !8, size: 800, align: 8, elements: !9)
!8 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!9 = !{!10}
!10 = !DISubrange(count: 100)
!12 = distinct !DISubprogram(name: "f", linkageName: "_Z1f1S", scope: !1, file: !1, line: 10, type: !13, isLocal: false, isDefinition: true, scopeLine: 10, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!13 = !DISubroutineType(types: !14)
!14 = !{null, !4}
!15 = !{i32 2, !"Dwarf Version", i32 4}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{!"clang version 3.8.0 (trunk 254019) (llvm/trunk 254036)"}
!18 = !DILocalVariable(name: "zzz", arg: 1, scope: !12, file: !1, line: 10, type: !4)
!19 = !DIExpression()
!20 = !DILocation(line: 10, column: 10, scope: !12)
!21 = !DILocalVariable(name: "xxx", scope: !12, file: !1, line: 11, type: !4)
!22 = !DILocation(line: 11, column: 5, scope: !12)
!23 = !DILocation(line: 12, column: 3, scope: !12)
!24 = !DILocation(line: 13, column: 3, scope: !12)
!25 = !DILocation(line: 14, column: 1, scope: !12)
