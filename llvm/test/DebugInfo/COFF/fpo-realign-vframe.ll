; RUN: llc < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -filetype=obj < %s | llvm-readobj --codeview - | FileCheck %s --check-prefix=OBJ

; PR38857

; When stack realignment is required by dynamic allocas are not used, the
; compiler will address locals with the ESP register. However, if call argument
; set up uses PUSH instructions, ESP may vary over the course of the function.
; This means it's not useful as a base register for describing the locations of
; variables. Instead, our CodeView output prefers to use the VFRAME virtual
; register, which is defined in the FPO data as $T0. Make sure we define it.

; Original C++ test case, which uses __thiscall to encourage PUSH conversion:
; struct Foo {
;   int x = 42;
;   int __declspec(noinline) foo();
;   void __declspec(noinline) bar(int *a, int *b, ptr c);
; };
; int Foo::foo() {
;   int a = 1;
;   int b = 2;
;   double __declspec(align(8)) force_alignment = 0.42;
;   bar(&a, &b, &force_alignment);
;   x += (int)force_alignment;
;   return x;
; }
; void Foo::bar(int *a, int *b, ptr c) {
;   __debugbreak();
;   *c += *a + *b;
; }
; int main() {
;   Foo o;
;   o.foo();
; }
; This stops the debugger in bar, and locals in Foo::foo would be corrupt.

; More reduced C code to generate this IR:
; int getval(void);
; void usevals(int *, int *, ptr);
; int realign_with_csrs(int x) {
;   int a = getval();
;   double __declspec(align(8)) force_alignment = 0.42;
;   usevals(&a, &x, &force_alignment);
;   return x;
; }

; Match the prologue for the .cv_fpo* directives.
; ASM-LABEL: _realign_with_csrs:
; ASM:         .cv_fpo_proc    _realign_with_csrs 4
; ASM: # %bb.0:                                # %entry
; ASM:         pushl   %ebp
; ASM:         .cv_fpo_pushreg %ebp
; ASM:         movl    %esp, %ebp
; ASM:         .cv_fpo_setframe        %ebp
; ASM:         andl    $-8, %esp
; ASM:         .cv_fpo_stackalign      8
; ASM:         subl    $16, %esp
; ASM:         .cv_fpo_stackalloc      16
; ASM:         .cv_fpo_endprologue

; 'x' should be EBP-relative, 'a' and 'force_alignment' ESP relative.
; ASM:         calll   _getval
; ASM-DAG:     leal    8(%esp), %[[LEA_DBL:[^ ]*]]
; ASM-DAG:     leal    8(%ebp), %[[LEA_X:[^ ]*]]
; ASM-DAG:     leal    4(%esp), %[[LEA_A:[^ ]*]]
; ASM:         pushl   %[[LEA_DBL]]
; ASM:         pushl   %[[LEA_X]]
; ASM:         pushl   %[[LEA_A]]
; ASM:         calll   _usevals
; ASM:         addl    $12, %esp

; OBJ: Subsection [
; OBJ:   SubSectionType: Symbols (0xF1)
; OBJ: ]
; OBJ: Subsection [
; OBJ:   SubSectionType: FrameData (0xF5)
;   	Really, the only important FrameFunc is the last one.
; OBJ:   FrameData {
; OBJ:   }
; OBJ:   FrameData {
; OBJ:   }
; OBJ:   FrameData {
; OBJ:   }
; OBJ:   FrameData {
; OBJ:     FrameFunc [
; OBJ-NEXT:   $T1 $ebp 4 + =
; OBJ-NEXT:   $T0 $T1 4 - 8 @ =
; OBJ-NEXT:   $eip $T1 ^ =
; OBJ-NEXT:   $esp $T1 4 + =
; OBJ-NEXT:   $ebp $T1 4 - ^ =
; OBJ-NEXT: ]
; OBJ:   }
; OBJ: ]
; OBJ: Subsection [
; OBJ:   SubSectionType: Symbols (0xF1)
; OBJ:   GlobalProcIdSym {
; OBJ:     Kind: S_GPROC32_ID (0x1147)
; OBJ:     Flags [ (0x81)
; OBJ:       HasFP (0x1)
; OBJ:       HasOptimizedDebugInfo (0x80)
; OBJ:     ]
; OBJ:     DisplayName: realign_with_csrs
; OBJ:     LinkageName: _realign_with_csrs
; OBJ:   }
; 	The frame register for locals should be VFRAME, and EBP for parameters.
; OBJ:   FrameProcSym {
; OBJ:     Kind: S_FRAMEPROC (0x1012)
; OBJ:     TotalFrameBytes: 0x14
; OBJ:     LocalFramePtrReg: VFRAME (0x7536)
; OBJ:     ParamFramePtrReg: EBP (0x16)
; OBJ:   }
; 	As seen in ASM, offset of x is 8.
; OBJ:   LocalSym {
; OBJ:     Kind: S_LOCAL (0x113E)
; OBJ:     Type: int (0x74)
; OBJ:     Flags [ (0x1)
; OBJ:       IsParameter (0x1)
; OBJ:     ]
; OBJ:     VarName: x
; OBJ:   }
; OBJ:   DefRangeFramePointerRelSym {
; OBJ:     Kind: S_DEFRANGE_FRAMEPOINTER_REL (0x1142)
; OBJ:     Offset: 8
; OBJ:   }
; 	ESP is VFRAME - 16, ESP offset of 'a' is 4, so -12.
; OBJ:   LocalSym {
; OBJ:     Kind: S_LOCAL (0x113E)
; OBJ:     Type: int (0x74)
; OBJ:     Flags [ (0x0)
; OBJ:     ]
; OBJ:     VarName: a
; OBJ:   }
; OBJ:   DefRangeFramePointerRelSym {
; OBJ:     Kind: S_DEFRANGE_FRAMEPOINTER_REL (0x1142)
; OBJ:     Offset: -12
; OBJ:   }
; 	ESP is VFRAME - 16, ESP offset of 'force_alignment' is 8, so -8.
; OBJ:   LocalSym {
; OBJ:     Kind: S_LOCAL (0x113E)
; OBJ:     Type: double (0x41)
; OBJ:     Flags [ (0x0)
; OBJ:     ]
; OBJ:     VarName: force_alignment
; OBJ:   }
; OBJ:   DefRangeFramePointerRelSym {
; OBJ:     Kind: S_DEFRANGE_FRAMEPOINTER_REL (0x1142)
; OBJ:     Offset: -8
; OBJ:   }
; OBJ:   ProcEnd {
; OBJ:     Kind: S_PROC_ID_END (0x114F)
; OBJ:   }
; OBJ: ]

; ModuleID = 't.c'
source_filename = "t.c"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc19.14.26433"

; Function Attrs: nounwind
define dso_local i32 @realign_with_csrs(i32 %x) local_unnamed_addr #0 !dbg !8 {
entry:
  %x.addr = alloca i32, align 4
  %a = alloca i32, align 4
  %force_alignment = alloca double, align 8
  store i32 %x, ptr %x.addr, align 4, !tbaa !17
  call void @llvm.dbg.declare(metadata ptr %x.addr, metadata !13, metadata !DIExpression()), !dbg !21
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %a) #4, !dbg !22
  call void @llvm.dbg.declare(metadata ptr %a, metadata !14, metadata !DIExpression()), !dbg !22
  %call = tail call i32 @getval() #4, !dbg !22
  store i32 %call, ptr %a, align 4, !dbg !22, !tbaa !17
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %force_alignment) #4, !dbg !23
  call void @llvm.dbg.declare(metadata ptr %force_alignment, metadata !15, metadata !DIExpression()), !dbg !23
  store double 4.200000e-01, ptr %force_alignment, align 8, !dbg !23, !tbaa !24
  call void @usevals(ptr nonnull %a, ptr nonnull %x.addr, ptr nonnull %force_alignment) #4, !dbg !26
  %0 = load i32, ptr %x.addr, align 4, !dbg !27, !tbaa !17
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %force_alignment) #4, !dbg !28
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %a) #4, !dbg !28
  ret i32 %0, !dbg !27
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #2

declare dso_local i32 @getval() local_unnamed_addr #3

declare dso_local void @usevals(ptr, ptr, ptr) local_unnamed_addr #3

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #2

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { argmemonly nounwind }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 8.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "C:\5Csrc\5Cllvm-project\5Cbuild", checksumkind: CSK_MD5, checksum: "a646950309d5d01d8087fc10fea33941")
!2 = !{}
!3 = !{i32 1, !"NumRegisterParameters", i32 0}
!4 = !{i32 2, !"CodeView", i32 1}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 2}
!7 = !{!"clang version 8.0.0 "}
!8 = distinct !DISubprogram(name: "realign_with_csrs", scope: !1, file: !1, line: 3, type: !9, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13, !14, !15}
!13 = !DILocalVariable(name: "x", arg: 1, scope: !8, file: !1, line: 3, type: !11)
!14 = !DILocalVariable(name: "a", scope: !8, file: !1, line: 4, type: !11)
!15 = !DILocalVariable(name: "force_alignment", scope: !8, file: !1, line: 5, type: !16, align: 64)
!16 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!17 = !{!18, !18, i64 0}
!18 = !{!"int", !19, i64 0}
!19 = !{!"omnipotent char", !20, i64 0}
!20 = !{!"Simple C/C++ TBAA"}
!21 = !DILocation(line: 3, scope: !8)
!22 = !DILocation(line: 4, scope: !8)
!23 = !DILocation(line: 5, scope: !8)
!24 = !{!25, !25, i64 0}
!25 = !{!"double", !19, i64 0}
!26 = !DILocation(line: 6, scope: !8)
!27 = !DILocation(line: 7, scope: !8)
!28 = !DILocation(line: 8, scope: !8)
