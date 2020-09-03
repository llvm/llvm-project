; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s

; Test llvm.dbg.value for the local variables moved onto the unsafe stack.
; SafeStack rewrites them relative to the unsafe stack pointer (base address of
; the unsafe stack frame).

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline safestack uwtable
define void @f() #0 !dbg !6 {
entry:
; CHECK:   %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
  %x1 = alloca i32, align 4
  %x2 = alloca i32, align 4
  %0 = bitcast i32* %x1 to i8*, !dbg !13
  %1 = bitcast i32* %x2 to i8*, !dbg !14

; Unhandled dbg.value: expression does not start with OP_DW_deref
; CHECK: call void @llvm.dbg.value(metadata i32* undef, metadata !{{.*}}, metadata !{{.*}})
  tail call void @llvm.dbg.value(metadata i32* %x1, metadata !10, metadata !23), !dbg !16

; Unhandled dbg.value: expression does not start with OP_DW_deref
; CHECK: call void @llvm.dbg.value(metadata i32* undef, metadata !{{.*}}, metadata !{{.*}})
  tail call void @llvm.dbg.value(metadata i32* %x1, metadata !10, metadata !24), !dbg !16

; Supported dbg.value: rewritted based on the [[USP]] value.
; CHECK: call void @llvm.dbg.value(metadata i8* %[[USP]], metadata ![[X1:.*]], metadata !DIExpression(DW_OP_constu, 4, DW_OP_minus, DW_OP_deref, DW_OP_LLVM_fragment, 0, 4))
  tail call void @llvm.dbg.value(metadata i32* %x1, metadata !10, metadata !25), !dbg !16

; Supported dbg.value: rewritted based on the [[USP]] value.
; CHECK: call void @llvm.dbg.value(metadata i8* %[[USP]], metadata ![[X1:.*]], metadata !DIExpression(DW_OP_constu, 4, DW_OP_minus, DW_OP_deref))
  tail call void @llvm.dbg.value(metadata i32* %x1, metadata !10, metadata !15), !dbg !16
  call void @capture(i32* nonnull %x1), !dbg !17

; An extra non-dbg.value metadata use of %x2. Replaced with undef.
; CHECK: call void @llvm.random.metadata.use(metadata i32* undef
  call void @llvm.random.metadata.use(metadata i32* %x2)

; CHECK: call void @llvm.dbg.value(metadata i8* %[[USP]], metadata ![[X2:.*]], metadata !DIExpression(DW_OP_constu, 8, DW_OP_minus, DW_OP_deref))
  call void @llvm.dbg.value(metadata i32* %x2, metadata !12, metadata !15), !dbg !18
  call void @capture(i32* nonnull %x2), !dbg !19
  ret void, !dbg !20
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

declare void @capture(i32*) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

declare void @llvm.random.metadata.use(metadata)

attributes #0 = { noinline safestack uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 (trunk 271022) (llvm/trunk 271027)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "../llvm/2.cc", directory: "/code/build-llvm")

!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 3.9.0 (trunk 271022) (llvm/trunk 271027)"}
!6 = distinct !DISubprogram(name: "f", linkageName: "_Z1fv", scope: !1, file: !1, line: 4, type: !7, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !9)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !{!10, !12}

; CHECK-DAG: ![[X1]] = !DILocalVariable(name: "x1",
!10 = !DILocalVariable(name: "x1", scope: !6, file: !1, line: 5, type: !11)
!11 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)

; CHECK-DAG: ![[X2]] = !DILocalVariable(name: "x2",
!12 = !DILocalVariable(name: "x2", scope: !6, file: !1, line: 6, type: !11)
!13 = !DILocation(line: 5, column: 3, scope: !6)
!14 = !DILocation(line: 6, column: 3, scope: !6)

!15 = !DIExpression(DW_OP_deref)
!16 = !DILocation(line: 5, column: 7, scope: !6)
!17 = !DILocation(line: 8, column: 3, scope: !6)
!18 = !DILocation(line: 6, column: 7, scope: !6)
!19 = !DILocation(line: 9, column: 3, scope: !6)
!20 = !DILocation(line: 10, column: 1, scope: !6)
!21 = !DILocation(line: 10, column: 1, scope: !22)
!22 = !DILexicalBlockFile(scope: !6, file: !1, discriminator: 1)
!23 = !DIExpression()
!24 = !DIExpression(DW_OP_constu, 42, DW_OP_minus)
!25 = !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 0, 4)
