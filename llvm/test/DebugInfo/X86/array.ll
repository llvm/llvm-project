; ModuleID = 'array.c'
;
; From (clang -g -c -O1):
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
; RUN: llc -filetype=asm %s -o - | FileCheck %s
; Test that we only emit register-indirect locations for the array array.
; rdar://problem/14874886
;
; CHECK:     ##DEBUG_VALUE: main:array <- [R{{.*}}+0]
; CHECK-NOT: ##DEBUG_VALUE: main:array <- R{{.*}}
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

@main.array = private unnamed_addr constant [4 x i32] [i32 0, i32 1, i32 2, i32 3], align 16

; Function Attrs: nounwind ssp uwtable
define void @f(i32* nocapture %p) #0 {
  tail call void @llvm.dbg.value(metadata i32* %p, i64 0, metadata !11, metadata !{!"0x102"}), !dbg !28
  store i32 42, i32* %p, align 4, !dbg !29, !tbaa !30
  ret void, !dbg !34
}

; Function Attrs: nounwind ssp uwtable
define i32 @main(i32 %argc, i8** nocapture readnone %argv) #0 {
  %array = alloca [4 x i32], align 16
  tail call void @llvm.dbg.value(metadata i32 %argc, i64 0, metadata !19, metadata !{!"0x102"}), !dbg !35
  tail call void @llvm.dbg.value(metadata i8** %argv, i64 0, metadata !20, metadata !{!"0x102"}), !dbg !35
  tail call void @llvm.dbg.value(metadata [4 x i32]* %array, i64 0, metadata !21, metadata !{!"0x102"}), !dbg !36
  %1 = bitcast [4 x i32]* %array to i8*, !dbg !36
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* bitcast ([4 x i32]* @main.array to i8*), i64 16, i32 16, i1 false), !dbg !36
  tail call void @llvm.dbg.value(metadata [4 x i32]* %array, i64 0, metadata !21, metadata !{!"0x102"}), !dbg !36
  %2 = getelementptr inbounds [4 x i32]* %array, i64 0, i64 0, !dbg !37
  call void @f(i32* %2), !dbg !37
  tail call void @llvm.dbg.value(metadata [4 x i32]* %array, i64 0, metadata !21, metadata !{!"0x102"}), !dbg !36
  %3 = load i32* %2, align 16, !dbg !38, !tbaa !30
  ret i32 %3, !dbg !38
}

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!25, !26}
!llvm.ident = !{!27}

!0 = !{!"0x11\0012\00clang version 3.5.0 \001\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/array.c] [DW_LANG_C99]
!1 = !{!"array.c", !""}
!2 = !{}
!3 = !{!4, !12}
!4 = !{!"0x2e\00f\00f\00\001\000\001\000\006\00256\001\001", !1, !5, !6, null, void (i32*)* @f, null, null, !10} ; [ DW_TAG_subprogram ] [line 1] [def] [f]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/array.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null, !8}
!8 = !{!"0xf\00\000\0064\0064\000\000", null, null, !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = !{!11}
!11 = !{!"0x101\00p\0016777217\000", !4, !5, !8} ; [ DW_TAG_arg_variable ] [p] [line 1]
!12 = !{!"0x2e\00main\00main\00\005\000\001\000\006\00256\001\005", !1, !5, !13, null, i32 (i32, i8**)* @main, null, null, !18} ; [ DW_TAG_subprogram ] [line 5] [def] [main]
!13 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !14, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = !{!9, !9, !15}
!15 = !{!"0xf\00\000\0064\0064\000\000", null, null, !16} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!16 = !{!"0xf\00\000\0064\0064\000\000", null, null, !17} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from char]
!17 = !{!"0x24\00char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ] [char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]
!18 = !{!19, !20, !21}
!19 = !{!"0x101\00argc\0016777221\000", !12, !5, !9} ; [ DW_TAG_arg_variable ] [argc] [line 5]
!20 = !{!"0x101\00argv\0033554437\000", !12, !5, !15} ; [ DW_TAG_arg_variable ] [argv] [line 5]
!21 = !{!"0x100\00array\006\000", !12, !5, !22} ; [ DW_TAG_auto_variable ] [array] [line 6]
!22 = !{!"0x1\00\000\00128\0032\000\000", null, null, !9, !23, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 128, align 32, offset 0] [from int]
!23 = !{!24}
!24 = !{!"0x21\000\004"}        ; [ DW_TAG_subrange_type ] [0, 3]
!25 = !{i32 2, !"Dwarf Version", i32 2}
!26 = !{i32 1, !"Debug Info Version", i32 2}
!27 = !{!"clang version 3.5.0 "}
!28 = !MDLocation(line: 1, scope: !4)
!29 = !MDLocation(line: 2, scope: !4)
!30 = !{!31, !31, i64 0}
!31 = !{!"int", !32, i64 0}
!32 = !{!"omnipotent char", !33, i64 0}
!33 = !{!"Simple C/C++ TBAA"}
!34 = !MDLocation(line: 3, scope: !4)
!35 = !MDLocation(line: 5, scope: !12)
!36 = !MDLocation(line: 6, scope: !12)
!37 = !MDLocation(line: 7, scope: !12)
!38 = !MDLocation(line: 8, scope: !12)
