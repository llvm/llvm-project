; RUN: llc < %s | FileCheck %s
; RUN: llc -filetype=obj < %s | llvm-dwarfdump -debug-loc - | FileCheck %s --check-prefix=DWARF

; Compile the following with -O1:

; struct IntPair { int x, y; };
; int g(void);
; int bitpiece_spill() {
;   struct IntPair o = {g(), 0};
;   // Force o.x to spill
;   asm volatile("" : : : "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp",
;                "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15");
;   return o.x;
; }

; CHECK-LABEL: bitpiece_spill:                         # @bitpiece_spill
; CHECK:               callq   g
; CHECK:               movl    %eax, [[offs:[0-9]+]](%rsp)          # 4-byte Spill
; CHECK:               #DEBUG_VALUE: bitpiece_spill:o <- [DW_OP_LLVM_fragment 32 32] 0
; CHECK:               #DEBUG_VALUE: bitpiece_spill:o <- [DW_OP_plus_uconst [[offs]], DW_OP_LLVM_fragment 0 32] [%rsp+0]
; CHECK:               #APP
; CHECK:               #NO_APP
; CHECK:               movl    [[offs]](%rsp), %eax          # 4-byte Reload
; CHECK:               retq

; DWARF: .debug_loc contents:
; DWARF-NEXT: 0x00000000:
; DWARF-NEXT: {{.*}}: DW_OP_breg7 RSP+{{[0-9]+}}, DW_OP_piece 0x4, DW_OP_constu 0x0, DW_OP_stack_value, DW_OP_piece 0x4

; ModuleID = 't.c'
source_filename = "t.c"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24210"

%struct.IntPair = type { i32, i32 }

; Function Attrs: nounwind uwtable
define i32 @bitpiece_spill() local_unnamed_addr #0 !dbg !7 {
entry:
  tail call void @llvm.dbg.declare(metadata %struct.IntPair* undef, metadata !12, metadata !17), !dbg !18
  %call = tail call i32 @g() #3, !dbg !19
  tail call void @llvm.dbg.value(metadata i32 %call, metadata !12, metadata !20), !dbg !18
  tail call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !21), !dbg !18
  tail call void asm sideeffect "", "~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{dirflag},~{fpsr},~{flags}"() #3, !dbg !22, !srcloc !23
  ret i32 %call, !dbg !24
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i32 @g() local_unnamed_addr #2

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.c", directory: "C:\5Csrc\5Cllvm\5Cbuild")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 4.0.0 "}
!7 = distinct !DISubprogram(name: "bitpiece_spill", scope: !1, file: !1, line: 3, type: !8, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: true, unit: !0, variables: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "o", scope: !7, file: !1, line: 4, type: !13)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "IntPair", file: !1, line: 1, size: 64, align: 32, elements: !14)
!14 = !{!15, !16}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !13, file: !1, line: 1, baseType: !10, size: 32, align: 32)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !13, file: !1, line: 1, baseType: !10, size: 32, align: 32, offset: 32)
!17 = !DIExpression()
!18 = !DILocation(line: 4, column: 18, scope: !7)
!19 = !DILocation(line: 4, column: 23, scope: !7)
!20 = !DIExpression(DW_OP_LLVM_fragment, 0, 32)
!21 = !DIExpression(DW_OP_LLVM_fragment, 32, 32)
!22 = !DILocation(line: 6, column: 3, scope: !7)
!23 = !{i32 138}
!24 = !DILocation(line: 8, column: 3, scope: !7)
