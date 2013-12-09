; RUN: llc -O0 -mtriple=x86_64-apple-darwin -filetype=asm %s -o - | FileCheck %s
; Ensure that we generate an indirect location for the variable length array a.
; CHECK: ##DEBUG_VALUE: vla:a <- RDX
; CHECK: DW_OP_breg1
; rdar://problem/13658587
;
; generated from:
;
; int vla(int n) {
;   int a[n];
;   a[0] = 42;
;   return a[n-1];
; }
;
; int main(int argc, char** argv) {
;    return vla(argc);
; }

; ModuleID = 'vla.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; Function Attrs: nounwind ssp uwtable
define i32 @vla(i32 %n) nounwind ssp uwtable {
entry:
  %n.addr = alloca i32, align 4
  %saved_stack = alloca i8*
  %cleanup.dest.slot = alloca i32
  store i32 %n, i32* %n.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %n.addr}, metadata !15), !dbg !16
  %0 = load i32* %n.addr, align 4, !dbg !17
  %1 = zext i32 %0 to i64, !dbg !17
  %2 = call i8* @llvm.stacksave(), !dbg !17
  store i8* %2, i8** %saved_stack, !dbg !17
  %vla = alloca i32, i64 %1, align 16, !dbg !17
  call void @llvm.dbg.declare(metadata !{i32* %vla}, metadata !18), !dbg !17
  %arrayidx = getelementptr inbounds i32* %vla, i64 0, !dbg !22
  store i32 42, i32* %arrayidx, align 4, !dbg !22
  %3 = load i32* %n.addr, align 4, !dbg !23
  %sub = sub nsw i32 %3, 1, !dbg !23
  %idxprom = sext i32 %sub to i64, !dbg !23
  %arrayidx1 = getelementptr inbounds i32* %vla, i64 %idxprom, !dbg !23
  %4 = load i32* %arrayidx1, align 4, !dbg !23
  store i32 1, i32* %cleanup.dest.slot
  %5 = load i8** %saved_stack, !dbg !24
  call void @llvm.stackrestore(i8* %5), !dbg !24
  ret i32 %4, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

; Function Attrs: nounwind
declare i8* @llvm.stacksave() nounwind

; Function Attrs: nounwind
declare void @llvm.stackrestore(i8*) nounwind

; Function Attrs: nounwind ssp uwtable
define i32 @main(i32 %argc, i8** %argv) nounwind ssp uwtable {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  store i32 0, i32* %retval
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %argc.addr}, metadata !25), !dbg !26
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata !{i8*** %argv.addr}, metadata !27), !dbg !26
  %0 = load i32* %argc.addr, align 4, !dbg !28
  %call = call i32 @vla(i32 %0), !dbg !28
  ret i32 %call, !dbg !28
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!29}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.3 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/vla.c] [DW_LANG_C99]
!1 = metadata !{metadata !"vla.c", metadata !""}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4, metadata !9}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"vla", metadata !"vla", metadata !"", i32 1, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32)* @vla, null, null, metadata !2, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [vla]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/vla.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !8}
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"main", metadata !"main", metadata !"", i32 7, metadata !10, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32, i8**)* @main, null, null, metadata !2, i32 7} ; [ DW_TAG_subprogram ] [line 7] [def] [main]
!10 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !11, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!11 = metadata !{metadata !8, metadata !8, metadata !12}
!12 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !13} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!13 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !14} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from char]
!14 = metadata !{i32 786468, null, null, metadata !"char", i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ] [char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]
!15 = metadata !{i32 786689, metadata !4, metadata !"n", metadata !5, i32 16777217, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [n] [line 1]
!16 = metadata !{i32 1, i32 0, metadata !4, null}
!17 = metadata !{i32 2, i32 0, metadata !4, null}
!18 = metadata !{i32 786688, metadata !4, metadata !"a", metadata !5, i32 2, metadata !19, i32 8192, i32 0} ; [ DW_TAG_auto_variable ] [a] [line 2]
!19 = metadata !{i32 786433, null, null, metadata !"", i32 0, i64 0, i64 32, i32 0, i32 0, metadata !8, metadata !20, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 0, align 32, offset 0] [from int]
!20 = metadata !{metadata !21}
!21 = metadata !{i32 786465, i64 0, i64 -1}       ; [ DW_TAG_subrange_type ] [unbounded]
!22 = metadata !{i32 3, i32 0, metadata !4, null}
!23 = metadata !{i32 4, i32 0, metadata !4, null}
!24 = metadata !{i32 5, i32 0, metadata !4, null}
!25 = metadata !{i32 786689, metadata !9, metadata !"argc", metadata !5, i32 16777223, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [argc] [line 7]
!26 = metadata !{i32 7, i32 0, metadata !9, null}
!27 = metadata !{i32 786689, metadata !9, metadata !"argv", metadata !5, i32 33554439, metadata !12, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [argv] [line 7]
!28 = metadata !{i32 8, i32 0, metadata !9, null}
!29 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
