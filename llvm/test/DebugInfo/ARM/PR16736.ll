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
  tail call void @llvm.dbg.value(metadata !{i32 %0}, i64 0, metadata !12), !dbg !18
  tail call void @llvm.dbg.value(metadata !{i32 %1}, i64 0, metadata !13), !dbg !18
  tail call void @llvm.dbg.value(metadata !{i32 %2}, i64 0, metadata !14), !dbg !18
  tail call void @llvm.dbg.value(metadata !{i32 %3}, i64 0, metadata !15), !dbg !18
  tail call void @llvm.dbg.value(metadata !{float %x}, i64 0, metadata !16), !dbg !18
  %call = tail call arm_aapcscc i32 @_Z1fv() #3, !dbg !19
  %conv = sitofp i32 %call to float, !dbg !19
  tail call void @llvm.dbg.value(metadata !{float %conv}, i64 0, metadata !16), !dbg !19
  tail call arm_aapcscc void @_Z1gf(float %conv) #3, !dbg !19
  ret void, !dbg !20
}

declare arm_aapcscc void @_Z1gf(float)

declare arm_aapcscc i32 @_Z1fv()

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata) #2

attributes #0 = { nounwind  }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17, !21}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 (trunk 190804) (llvm/trunk 190797)", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [//<unknown>] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"/<unknown>", metadata !""}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !5, metadata !6, metadata !"h", metadata !"h", metadata !"_Z1hiiiif", i32 3, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, void (i32, i32, i32, i32, float)* @_Z1hiiiif, null, null, metadata !11, i32 3} ; [ DW_TAG_subprogram ] [line 3] [def] [h]
!5 = metadata !{metadata !"/arm.cpp", metadata !""}
!6 = metadata !{i32 786473, metadata !5}          ; [ DW_TAG_file_type ] [//arm.cpp]
!7 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null, metadata !9, metadata !9, metadata !9, metadata !9, metadata !10}
!9 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{i32 786468, null, null, metadata !"float", i32 0, i64 32, i64 32, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ] [float] [line 0, size 32, align 32, offset 0, enc DW_ATE_float]
!11 = metadata !{metadata !12, metadata !13, metadata !14, metadata !15, metadata !16}
!12 = metadata !{i32 786689, metadata !4, metadata !"", metadata !6, i32 16777219, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [line 3]
!13 = metadata !{i32 786689, metadata !4, metadata !"", metadata !6, i32 33554435, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [line 3]
!14 = metadata !{i32 786689, metadata !4, metadata !"", metadata !6, i32 50331651, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [line 3]
!15 = metadata !{i32 786689, metadata !4, metadata !"", metadata !6, i32 67108867, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [line 3]
!16 = metadata !{i32 786689, metadata !4, metadata !"x", metadata !6, i32 83886083, metadata !10, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [x] [line 3]
!17 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!18 = metadata !{i32 3, i32 0, metadata !4, null}
!19 = metadata !{i32 4, i32 0, metadata !4, null}
!20 = metadata !{i32 5, i32 0, metadata !4, null}
!21 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
