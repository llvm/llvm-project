; RUN: llc -enable-shrink-wrap=true < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -enable-shrink-wrap=true -filetype=obj < %s | llvm-readobj --codeview | FileCheck %s --check-prefix=OBJ

; C source:
; int doSomething(int*);
; int __fastcall shrink_wrap_basic(int a, int b, int c, int d) {
;   if (a < b)
;     return a;
;   for (int i = c; i < d; ++i)
;     doSomething(&c);
;   return doSomething(&c);
; }

; ASM: @shrink_wrap_basic@16:                  # @"\01@shrink_wrap_basic@16"
; ASM:         .cv_fpo_proc    @shrink_wrap_basic@16 8
; ASM:         .cv_loc 0 1 3 9                 # t.c:3:9
; ASM:         movl    %ecx, %eax
; ASM:         cmpl    %edx, %ecx
; ASM:         jl      [[EPILOGUE:LBB0_[0-9]+]]

; ASM:         pushl   %ebx
; ASM:         .cv_fpo_pushreg %ebx
; ASM:         pushl   %edi
; ASM:         .cv_fpo_pushreg %edi
; ASM:         pushl   %esi
; ASM:         .cv_fpo_pushreg %esi
; ASM:         .cv_fpo_endprologue

; ASM:         calll   _doSomething

; ASM:         popl    %esi
; ASM:         popl    %edi
; ASM:         popl    %ebx
; ASM: [[EPILOGUE]]:                                 # %return
; ASM:         retl    $8
; ASM: Ltmp10:
; ASM:         .cv_fpo_endproc

; Note how RvaStart advances 7 bytes to skip the shrink-wrapped portion.
; OBJ: SubSectionType: FrameData (0xF5)
; OBJ:    FrameData {
; OBJ:      RvaStart: 0x0
; OBJ:      CodeSize: 0x34
; OBJ:      PrologSize: 0x9
; OBJ:      FrameFunc [
; OBJ-NEXT:   $T0 .raSearch =
; OBJ-NEXT:   $eip $T0 ^ =
; OBJ-NEXT:   $esp $T0 4 + =
; OBJ-NEXT: ]
; OBJ:    }
; OBJ:    FrameData {
; OBJ:      RvaStart: 0x7
; OBJ:      CodeSize: 0x2D
; OBJ:      PrologSize: 0x2
; OBJ:      FrameFunc [
; OBJ-NEXT:   $T0 .raSearch =
; OBJ-NEXT:   $eip $T0 ^ =
; OBJ-NEXT:   $esp $T0 4 + =
; OBJ-NEXT:   $ebx $T0 4 - ^ =
; OBJ-NEXT: ]
; OBJ:    }
; OBJ:    FrameData {
; OBJ:      RvaStart: 0x8
; OBJ:      CodeSize: 0x2C
; OBJ:      PrologSize: 0x1
; OBJ:      FrameFunc [
; OBJ-NEXT:   $T0 .raSearch =
; OBJ-NEXT:   $eip $T0 ^ =
; OBJ-NEXT:   $esp $T0 4 + =
; OBJ-NEXT:   $ebx $T0 4 - ^ =
; OBJ-NEXT:   $edi $T0 8 - ^ =
; OBJ-NEXT: ]
; OBJ:    }
; OBJ:    FrameData {
; OBJ:      RvaStart: 0x9
; OBJ:      CodeSize: 0x2B
; OBJ:      PrologSize: 0x0
; OBJ:      FrameFunc [
; OBJ-NEXT:   $T0 .raSearch =
; OBJ-NEXT:   $eip $T0 ^ =
; OBJ-NEXT:   $esp $T0 4 + =
; OBJ-NEXT:   $ebx $T0 4 - ^ =
; OBJ-NEXT:   $edi $T0 8 - ^ =
; OBJ-NEXT:   $esi $T0 12 - ^ =
; OBJ-NEXT: ]
; OBJ:    }
; OBJ-NOT: FrameData

; ModuleID = 't.c'
source_filename = "t.c"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc19.11.25508"

; Function Attrs: nounwind
define x86_fastcallcc i32 @"\01@shrink_wrap_basic@16"(i32 inreg %a, i32 inreg %b, i32 %c, i32 %d) local_unnamed_addr #0 !dbg !8 {
entry:
  %c.addr = alloca i32, align 4
  tail call void @llvm.dbg.value(metadata i32 %d, metadata !13, metadata !DIExpression()), !dbg !19
  tail call void @llvm.dbg.value(metadata i32 %c, metadata !14, metadata !DIExpression()), !dbg !20
  store i32 %c, i32* %c.addr, align 4, !tbaa !21
  tail call void @llvm.dbg.value(metadata i32 %b, metadata !15, metadata !DIExpression()), !dbg !25
  tail call void @llvm.dbg.value(metadata i32 %a, metadata !16, metadata !DIExpression()), !dbg !26
  %cmp = icmp slt i32 %a, %b, !dbg !27
  br i1 %cmp, label %return, label %for.cond.preheader, !dbg !29

for.cond.preheader:                               ; preds = %entry
  br label %for.cond, !dbg !30

for.cond:                                         ; preds = %for.cond.preheader, %for.cond
  %i.0 = phi i32 [ %inc, %for.cond ], [ %c, %for.cond.preheader ]
  call void @llvm.dbg.value(metadata i32 %i.0, metadata !17, metadata !DIExpression()), !dbg !32
  %cmp1 = icmp slt i32 %i.0, %d, !dbg !30
  call void @llvm.dbg.value(metadata i32* %c.addr, metadata !14, metadata !DIExpression()), !dbg !20
  %call = call i32 @doSomething(i32* nonnull %c.addr) #3, !dbg !33
  %inc = add nsw i32 %i.0, 1, !dbg !34
  call void @llvm.dbg.value(metadata i32 %inc, metadata !17, metadata !DIExpression()), !dbg !32
  br i1 %cmp1, label %for.cond, label %return, !dbg !35, !llvm.loop !36

return:                                           ; preds = %for.cond, %entry
  %retval.0 = phi i32 [ %a, %entry ], [ %call, %for.cond ]
  ret i32 %retval.0, !dbg !38
}

declare i32 @doSomething(i32*) local_unnamed_addr #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.c", directory: "C:\5Csrc\5Cllvm-project\5Cbuild", checksumkind: CSK_MD5, checksum: "32f118fd5dd7e65ff7733c49b2f804ef")
!2 = !{}
!3 = !{i32 1, !"NumRegisterParameters", i32 0}
!4 = !{i32 2, !"CodeView", i32 1}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 2}
!7 = !{!"clang version 6.0.0 "}
!8 = distinct !DISubprogram(name: "shrink_wrap_basic", linkageName: "\01@shrink_wrap_basic@16", scope: !1, file: !1, line: 2, type: !9, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(cc: DW_CC_BORLAND_msfastcall, types: !10)
!10 = !{!11, !11, !11, !11, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13, !14, !15, !16, !17}
!13 = !DILocalVariable(name: "d", arg: 4, scope: !8, file: !1, line: 2, type: !11)
!14 = !DILocalVariable(name: "c", arg: 3, scope: !8, file: !1, line: 2, type: !11)
!15 = !DILocalVariable(name: "b", arg: 2, scope: !8, file: !1, line: 2, type: !11)
!16 = !DILocalVariable(name: "a", arg: 1, scope: !8, file: !1, line: 2, type: !11)
!17 = !DILocalVariable(name: "i", scope: !18, file: !1, line: 5, type: !11)
!18 = distinct !DILexicalBlock(scope: !8, file: !1, line: 5, column: 3)
!19 = !DILocation(line: 2, column: 59, scope: !8)
!20 = !DILocation(line: 2, column: 52, scope: !8)
!21 = !{!22, !22, i64 0}
!22 = !{!"int", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C/C++ TBAA"}
!25 = !DILocation(line: 2, column: 45, scope: !8)
!26 = !DILocation(line: 2, column: 38, scope: !8)
!27 = !DILocation(line: 3, column: 9, scope: !28)
!28 = distinct !DILexicalBlock(scope: !8, file: !1, line: 3, column: 7)
!29 = !DILocation(line: 3, column: 7, scope: !8)
!30 = !DILocation(line: 5, column: 21, scope: !31)
!31 = distinct !DILexicalBlock(scope: !18, file: !1, line: 5, column: 3)
!32 = !DILocation(line: 5, column: 12, scope: !18)
!33 = !DILocation(line: 0, scope: !8)
!34 = !DILocation(line: 5, column: 26, scope: !31)
!35 = !DILocation(line: 5, column: 3, scope: !18)
!36 = distinct !{!36, !35, !37}
!37 = !DILocation(line: 6, column: 19, scope: !18)
!38 = !DILocation(line: 8, column: 1, scope: !8)
