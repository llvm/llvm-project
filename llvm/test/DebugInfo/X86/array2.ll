; ModuleID = 'array.c'
;
; From (clang -g -c -O0):
;
; void f(int* p) {
;   p[0] = 42;
; }
;
; int main(int argc, char** argv) {
;   int array[4] = { 0, 1, 2, 3 };
;   f(array);
;   return array[0];
; }
;
; RUN: opt %s -O2 -S -o - | FileCheck %s
; Test that we do not lower dbg.declares for arrays.
;
; CHECK: define i32 @main
; CHECK: call void @llvm.dbg.value
; CHECK: call void @llvm.dbg.value
; CHECK: call void @llvm.dbg.declare
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

@main.array = private unnamed_addr constant [4 x i32] [i32 0, i32 1, i32 2, i32 3], align 16

; Function Attrs: nounwind ssp uwtable
define void @f(i32* %p) #0 {
entry:
  %p.addr = alloca i32*, align 8
  store i32* %p, i32** %p.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %p.addr, metadata !19, metadata !{!"0x102"}), !dbg !20
  %0 = load i32** %p.addr, align 8, !dbg !21
  %arrayidx = getelementptr inbounds i32* %0, i64 0, !dbg !21
  store i32 42, i32* %arrayidx, align 4, !dbg !21
  ret void, !dbg !22
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind ssp uwtable
define i32 @main(i32 %argc, i8** %argv) #0 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %array = alloca [4 x i32], align 16
  store i32 0, i32* %retval
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !23, metadata !{!"0x102"}), !dbg !24
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !25, metadata !{!"0x102"}), !dbg !24
  call void @llvm.dbg.declare(metadata [4 x i32]* %array, metadata !26, metadata !{!"0x102"}), !dbg !30
  %0 = bitcast [4 x i32]* %array to i8*, !dbg !30
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* bitcast ([4 x i32]* @main.array to i8*), i64 16, i32 16, i1 false), !dbg !30
  %arraydecay = getelementptr inbounds [4 x i32]* %array, i32 0, i32 0, !dbg !31
  call void @f(i32* %arraydecay), !dbg !31
  %arrayidx = getelementptr inbounds [4 x i32]* %array, i32 0, i64 0, !dbg !32
  %1 = load i32* %arrayidx, align 4, !dbg !32
  ret i32 %1, !dbg !32
}

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #2

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!16, !17}
!llvm.ident = !{!18}

!0 = !{!"0x11\0012\00clang version 3.5.0 \000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [array.c] [DW_LANG_C99]
!1 = !{!"array.c", !""}
!2 = !{}
!3 = !{!4, !10}
!4 = !{!"0x2e\00f\00f\00\001\000\001\000\006\00256\000\001", !1, !5, !6, null, void (i32*)* @f, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [f]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [array.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null, !8}
!8 = !{!"0xf\00\000\0064\0064\000\000", null, null, !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = !{!"0x2e\00main\00main\00\005\000\001\000\006\00256\000\005", !1, !5, !11, null, i32 (i32, i8**)* @main, null, null, !2} ; [ DW_TAG_subprogram ] [line 5] [def] [main]
!11 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = !{!9, !9, !13}
!13 = !{!"0xf\00\000\0064\0064\000\000", null, null, !14} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!14 = !{!"0xf\00\000\0064\0064\000\000", null, null, !15} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from char]
!15 = !{!"0x24\00char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ] [char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]
!16 = !{i32 2, !"Dwarf Version", i32 2}
!17 = !{i32 1, !"Debug Info Version", i32 2}
!18 = !{!"clang version 3.5.0 "}
!19 = !{!"0x101\00p\0016777217\000", !4, !5, !8} ; [ DW_TAG_arg_variable ] [p] [line 1]
!20 = !MDLocation(line: 1, scope: !4)
!21 = !MDLocation(line: 2, scope: !4)
!22 = !MDLocation(line: 3, scope: !4)
!23 = !{!"0x101\00argc\0016777221\000", !10, !5, !9} ; [ DW_TAG_arg_variable ] [argc] [line 5]
!24 = !MDLocation(line: 5, scope: !10)
!25 = !{!"0x101\00argv\0033554437\000", !10, !5, !13} ; [ DW_TAG_arg_variable ] [argv] [line 5]
!26 = !{!"0x100\00array\006\000", !10, !5, !27} ; [ DW_TAG_auto_variable ] [array] [line 6]
!27 = !{!"0x1\00\000\00128\0032\000\000", null, null, !9, !28, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 128, align 32, offset 0] [from int]
!28 = !{!29}
!29 = !{!"0x21\000\004"}        ; [ DW_TAG_subrange_type ] [0, 3]
!30 = !MDLocation(line: 6, scope: !10)
!31 = !MDLocation(line: 7, scope: !10)
!32 = !MDLocation(line: 8, scope: !10)
