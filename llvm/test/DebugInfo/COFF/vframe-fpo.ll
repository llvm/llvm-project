; RUN: llc < %s | FileCheck %s --check-prefix=ASM
; RUN: llc < %s -filetype=obj | llvm-readobj -codeview | FileCheck %s --check-prefix=CODEVIEW

; This test checks that for 32-bit x86 we use VFRAME and
; S_DEFRANGE_FRAMEPOINTER_REL with the right offsets. The test has two function
; calls with different stack depths, which makes it impossible to correctly
; describe locals in memory as being ESP-relative.

; The following source can be used with a debugger to check that locals are
; displayed correctly:
; $ cat fpo.cpp
; #if 0
; void __attribute__((optnone)) __declspec(noinline) f(int &a, int &b) {
;   __debugbreak();
;   a += b;
; }
; void __attribute__((optnone)) __declspec(noinline) g(int &a, int &b, int &c) {
;   __debugbreak();
;   a += b;
;   a += c;
; }
; #endif
; void f(int &a, int &b);
; void g(int &a, int &b, int &c);
; int main() {
;   int a = 1;
;   int b = 2;
;   int c = 3;
;   f(a, b);
;   g(a, b, c);
;   return a + b + c;
; }
; $ clang -S -g -gcodeview t.cpp -emit-llvm -o vframe-fpo.ll -Os

; ASM-LABEL: _main:
; ASM:      # %bb.0:                                # %entry
; ASM-NEXT:         pushl   %ebx
; ASM-NEXT:         .cv_fpo_pushreg %ebx
; ASM-NEXT:         pushl   %edi
; ASM-NEXT:         .cv_fpo_pushreg %edi
; ASM-NEXT:         pushl   %esi
; ASM-NEXT:         .cv_fpo_pushreg %esi
; ASM-NEXT:         subl    $12, %esp
; ASM-NEXT:         .cv_fpo_stackalloc      12
; ASM-NEXT:         .cv_fpo_endprologue

; Store locals.
; ASM:         movl    $1, {{.*}}
; ASM:         movl    $2, {{.*}}
; ASM:         movl    $3, {{.*}}

; ASM that store-to-push conversion fires.
; ASM:         pushl
; ASM-NEXT:    pushl
; ASM-NEXT:    calll   "?f@@YAXAAH0@Z"
; ASM-NEXT:    addl    $8, %esp
; ASM:         pushl
; ASM-NEXT:    pushl
; ASM-NEXT:    pushl
; ASM-NEXT:    calll   "?g@@YAXAAH00@Z"

; CODEVIEW:      CodeViewDebugInfo [
; CODEVIEW-NEXT:   Section: .debug$S (4)
; CODEVIEW-NEXT:   Magic: 0x4
; CODEVIEW-NEXT:   Subsection [
; CODEVIEW-NEXT:     SubSectionType: Symbols (0xF1)
; CODEVIEW-NEXT:     SubSectionSize:
; CODEVIEW-NEXT:     Compile3Sym {
; CODEVIEW-NEXT:       Kind: S_COMPILE3 (0x113C)
; CODEVIEW:          }
; CODEVIEW:        ]
; CODEVIEW:        Subsection [
; CODEVIEW-NEXT:     SubSectionType: FrameData (0xF5)
; CODEVIEW-NEXT:     SubSectionSize:
; CODEVIEW-NEXT:     LinkageName: _main
; CODEVIEW:          FrameData {
; CODEVIEW:          }
; CODEVIEW:          FrameData {
; CODEVIEW:          }
; CODEVIEW:          FrameData {
; CODEVIEW:          }
; CODEVIEW:          FrameData {
; CODEVIEW:          }
; CODEVIEW:          FrameData {
; CODEVIEW-NEXT:       RvaStart:
; CODEVIEW-NEXT:       CodeSize:
; CODEVIEW-NEXT:       LocalSize: 0xC
; CODEVIEW-NEXT:       ParamsSize: 0x0
; CODEVIEW-NEXT:       MaxStackSize: 0x0
; CODEVIEW-NEXT:       PrologSize:
; CODEVIEW-NEXT:       SavedRegsSize: 0xC
; CODEVIEW-NEXT:       Flags [ (0x0)
; CODEVIEW-NEXT:       ]

; $T0 is the CFA, the address of the return address, and our defranges are
; relative to it.
; CODEVIEW-NEXT:       FrameFunc [
; CODEVIEW-NEXT:         $T0 .raSearch =
; CODEVIEW-NEXT:         $eip $T0 ^ =
; CODEVIEW-NEXT:         $esp $T0 4 + =
; CODEVIEW-NEXT:         $ebx $T0 4 - ^ =
; CODEVIEW-NEXT:         $edi $T0 8 - ^ =
; CODEVIEW-NEXT:         $esi $T0 12 - ^ =
; CODEVIEW-NEXT:       ]
; CODEVIEW-NEXT:     }

; We push 16 bytes in the prologue, so our local variables are at offsets -16,
; -20, and -24.

; CODEVIEW:      Subsection [
; CODEVIEW-NEXT:   SubSectionType: Symbols (0xF1)
; CODEVIEW-NEXT:   SubSectionSize:
; CODEVIEW-NEXT:   GlobalProcIdSym {
; CODEVIEW-NEXT:     Kind: S_GPROC32_ID (0x1147)
; CODEVIEW:          DisplayName: main
; CODEVIEW:          LinkageName: _main
; CODEVIEW-NEXT:   }
; CODEVIEW-NEXT:   FrameProcSym {
; CODEVIEW-NEXT:     Kind: S_FRAMEPROC (0x1012)
; CODEVIEW-NEXT:     TotalFrameBytes: 0xC
; CODEVIEW-NEXT:     PaddingFrameBytes: 0x0
; CODEVIEW-NEXT:     OffsetToPadding: 0x0
; CODEVIEW-NEXT:     BytesOfCalleeSavedRegisters: 0xC
; CODEVIEW-NEXT:     OffsetOfExceptionHandler: 0x0
; CODEVIEW-NEXT:     SectionIdOfExceptionHandler: 0x0
; CODEVIEW-NEXT:     Flags [ (0x14000)
; CODEVIEW-NEXT:     ]
; CODEVIEW-NEXT:     LocalFramePtrReg: VFRAME (0x7536)
; CODEVIEW-NEXT:     ParamFramePtrReg: VFRAME (0x7536)
; CODEVIEW-NEXT:   }
; CODEVIEW-NEXT:   LocalSym {
; CODEVIEW-NEXT:     Kind: S_LOCAL (0x113E)
; CODEVIEW:          VarName: a
; CODEVIEW-NEXT:   }
; CODEVIEW-NEXT:   DefRangeFramePointerRelSym {
; CODEVIEW-NEXT:     Kind: S_DEFRANGE_FRAMEPOINTER_REL (0x1142)
; CODEVIEW-NEXT:     Offset: -16
; CODEVIEW-NEXT:     LocalVariableAddrRange {
; CODEVIEW-NEXT:       OffsetStart:
; CODEVIEW-NEXT:       ISectStart:
; CODEVIEW-NEXT:       Range:
; CODEVIEW-NEXT:     }
; CODEVIEW-NEXT:   }
; CODEVIEW-NEXT:   LocalSym {
; CODEVIEW-NEXT:     Kind: S_LOCAL (0x113E)
; CODEVIEW:          VarName: b
; CODEVIEW-NEXT:   }
; CODEVIEW-NEXT:   DefRangeFramePointerRelSym {
; CODEVIEW-NEXT:     Kind: S_DEFRANGE_FRAMEPOINTER_REL (0x1142)
; CODEVIEW-NEXT:     Offset: -20
; CODEVIEW-NEXT:     LocalVariableAddrRange {
; CODEVIEW-NEXT:       OffsetStart:
; CODEVIEW-NEXT:       ISectStart:
; CODEVIEW-NEXT:       Range:
; CODEVIEW-NEXT:     }
; CODEVIEW-NEXT:   }
; CODEVIEW-NEXT:   LocalSym {
; CODEVIEW-NEXT:     Kind: S_LOCAL (0x113E)
; CODEVIEW:          VarName: c
; CODEVIEW-NEXT:   }
; CODEVIEW-NEXT:   DefRangeFramePointerRelSym {
; CODEVIEW-NEXT:     Kind: S_DEFRANGE_FRAMEPOINTER_REL (0x1142)
; CODEVIEW-NEXT:     Offset: -24
; CODEVIEW-NEXT:     LocalVariableAddrRange {
; CODEVIEW-NEXT:       OffsetStart:
; CODEVIEW-NEXT:       ISectStart:
; CODEVIEW-NEXT:       Range:
; CODEVIEW-NEXT:     }
; CODEVIEW-NEXT:   }
; CODEVIEW-NEXT:   ProcEnd {
; CODEVIEW-NEXT:     Kind: S_PROC_ID_END (0x114F)
; CODEVIEW-NEXT:   }
; CODEVIEW-NEXT: ]


; ModuleID = 'fpo.cpp'
source_filename = "fpo.cpp"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc19.14.26433"

; Function Attrs: norecurse optsize
define dso_local i32 @main() local_unnamed_addr #0 !dbg !8 {
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  %0 = bitcast i32* %a to i8*, !dbg !16
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %0) #4, !dbg !16
  call void @llvm.dbg.declare(metadata i32* %a, metadata !13, metadata !DIExpression()), !dbg !16
  store i32 1, i32* %a, align 4, !dbg !16, !tbaa !17
  %1 = bitcast i32* %b to i8*, !dbg !21
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %1) #4, !dbg !21
  call void @llvm.dbg.declare(metadata i32* %b, metadata !14, metadata !DIExpression()), !dbg !21
  store i32 2, i32* %b, align 4, !dbg !21, !tbaa !17
  %2 = bitcast i32* %c to i8*, !dbg !22
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %2) #4, !dbg !22
  call void @llvm.dbg.declare(metadata i32* %c, metadata !15, metadata !DIExpression()), !dbg !22
  store i32 3, i32* %c, align 4, !dbg !22, !tbaa !17
  call void @"?f@@YAXAAH0@Z"(i32* nonnull dereferenceable(4) %a, i32* nonnull dereferenceable(4) %b) #5, !dbg !23
  call void @"?g@@YAXAAH00@Z"(i32* nonnull dereferenceable(4) %a, i32* nonnull dereferenceable(4) %b, i32* nonnull dereferenceable(4) %c) #5, !dbg !24
  %3 = load i32, i32* %a, align 4, !dbg !25, !tbaa !17
  %4 = load i32, i32* %b, align 4, !dbg !25, !tbaa !17
  %add = add nsw i32 %4, %3, !dbg !25
  %5 = load i32, i32* %c, align 4, !dbg !25, !tbaa !17
  %add1 = add nsw i32 %add, %5, !dbg !25
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %2) #4, !dbg !26
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %1) #4, !dbg !26
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %0) #4, !dbg !26
  ret i32 %add1, !dbg !25
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

; Function Attrs: optsize
declare dso_local void @"?f@@YAXAAH0@Z"(i32* dereferenceable(4), i32* dereferenceable(4)) local_unnamed_addr #3

; Function Attrs: optsize
declare dso_local void @"?g@@YAXAAH00@Z"(i32* dereferenceable(4), i32* dereferenceable(4), i32* dereferenceable(4)) local_unnamed_addr #3

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

attributes #0 = { norecurse optsize "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { optsize "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }
attributes #5 = { optsize }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "fpo.cpp", directory: "C:\5Csrc\5Cllvm-project\5Cbuild", checksumkind: CSK_MD5, checksum: "d0bb7e43f4e54936a94da008319a7de3")
!2 = !{}
!3 = !{i32 1, !"NumRegisterParameters", i32 0}
!4 = !{i32 2, !"CodeView", i32 1}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 2}
!7 = !{!"clang version 8.0.0 "}
!8 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 14, type: !9, isLocal: false, isDefinition: true, scopeLine: 14, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13, !14, !15}
!13 = !DILocalVariable(name: "a", scope: !8, file: !1, line: 15, type: !11)
!14 = !DILocalVariable(name: "b", scope: !8, file: !1, line: 16, type: !11)
!15 = !DILocalVariable(name: "c", scope: !8, file: !1, line: 17, type: !11)
!16 = !DILocation(line: 15, scope: !8)
!17 = !{!18, !18, i64 0}
!18 = !{!"int", !19, i64 0}
!19 = !{!"omnipotent char", !20, i64 0}
!20 = !{!"Simple C++ TBAA"}
!21 = !DILocation(line: 16, scope: !8)
!22 = !DILocation(line: 17, scope: !8)
!23 = !DILocation(line: 18, scope: !8)
!24 = !DILocation(line: 19, scope: !8)
!25 = !DILocation(line: 20, scope: !8)
!26 = !DILocation(line: 21, scope: !8)
