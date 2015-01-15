; RUN: llc -filetype=asm < %s | FileCheck %s
; CHECK: @DEBUG_VALUE: h:x <- [R{{.*}}+{{.*}}]
; generated from:
; clang -cc1 -triple  thumbv7 -S -O1 arm.cpp  -g
;
; int f();
; void g(float);
; void h(int, int, int, int, float x) {
;    g(x = f());
; }
;
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:32-n32-S64"
target triple = "thumbv7-apple-ios"

; Function Attrs: nounwind
define arm_aapcscc void @_Z1hiiiif(i32, i32, i32, i32, float %x) #0 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %0, i64 0, metadata !12, metadata !{!"0x102"}), !dbg !18
  tail call void @llvm.dbg.value(metadata i32 %1, i64 0, metadata !13, metadata !{!"0x102"}), !dbg !18
  tail call void @llvm.dbg.value(metadata i32 %2, i64 0, metadata !14, metadata !{!"0x102"}), !dbg !18
  tail call void @llvm.dbg.value(metadata i32 %3, i64 0, metadata !15, metadata !{!"0x102"}), !dbg !18
  tail call void @llvm.dbg.value(metadata float %x, i64 0, metadata !16, metadata !{!"0x102"}), !dbg !18
  %call = tail call arm_aapcscc i32 @_Z1fv() #3, !dbg !19
  %conv = sitofp i32 %call to float, !dbg !19
  tail call void @llvm.dbg.value(metadata float %conv, i64 0, metadata !16, metadata !{!"0x102"}), !dbg !19
  tail call arm_aapcscc void @_Z1gf(float %conv) #3, !dbg !19
  ret void, !dbg !20
}

declare arm_aapcscc void @_Z1gf(float)

declare arm_aapcscc i32 @_Z1fv()

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { nounwind  }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17, !21}

!0 = !{!"0x11\004\00clang version 3.4 (trunk 190804) (llvm/trunk 190797)\001\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [//<unknown>] [DW_LANG_C_plus_plus]
!1 = !{!"/<unknown>", !""}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00h\00h\00_Z1hiiiif\003\000\001\000\006\00256\001\003", !5, !6, !7, null, void (i32, i32, i32, i32, float)* @_Z1hiiiif, null, null, !11} ; [ DW_TAG_subprogram ] [line 3] [def] [h]
!5 = !{!"/arm.cpp", !""}
!6 = !{!"0x29", !5}          ; [ DW_TAG_file_type ] [//arm.cpp]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{null, !9, !9, !9, !9, !10}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = !{!"0x24\00float\000\0032\0032\000\000\004", null, null} ; [ DW_TAG_base_type ] [float] [line 0, size 32, align 32, offset 0, enc DW_ATE_float]
!11 = !{!12, !13, !14, !15, !16}
!12 = !{!"0x101\00\0016777219\000", !4, !6, !9} ; [ DW_TAG_arg_variable ] [line 3]
!13 = !{!"0x101\00\0033554435\000", !4, !6, !9} ; [ DW_TAG_arg_variable ] [line 3]
!14 = !{!"0x101\00\0050331651\000", !4, !6, !9} ; [ DW_TAG_arg_variable ] [line 3]
!15 = !{!"0x101\00\0067108867\000", !4, !6, !9} ; [ DW_TAG_arg_variable ] [line 3]
!16 = !{!"0x101\00x\0083886083\000", !4, !6, !10} ; [ DW_TAG_arg_variable ] [x] [line 3]
!17 = !{i32 2, !"Dwarf Version", i32 4}
!18 = !MDLocation(line: 3, scope: !4)
!19 = !MDLocation(line: 4, scope: !4)
!20 = !MDLocation(line: 5, scope: !4)
!21 = !{i32 1, !"Debug Info Version", i32 2}
