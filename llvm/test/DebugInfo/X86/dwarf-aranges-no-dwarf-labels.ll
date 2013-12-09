; RUN: llc < %s | FileCheck %s

; CHECK: .short  2 # DWARF Arange version number
; CHECK: # Segment Size
; CHECK-NOT: debug_loc
; CHECK: .quad global
; CHECK-NOT: debug_loc
; CHECK: # ARange terminator

; --- Source code ---
; Generated with "clang -g -O1 -S -emit-llvm"

; int global = 2;
; int foo(int bar) { return bar; }
; int foo2(int bar2) { return bar2; }

; int main() {
;   return foo(2) + foo2(1) + global;
; }


; ModuleID = 'tmp/debug_ranges/a.cc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@global = global i32 2, align 4

; Function Attrs: nounwind readnone uwtable
define i32 @_Z3fooi(i32 %bar) #0 {
entry:
  tail call void @llvm.dbg.value(metadata !{i32 %bar}, i64 0, metadata !10), !dbg !20
  ret i32 %bar, !dbg !20
}

; Function Attrs: nounwind readnone uwtable
define i32 @_Z4foo2i(i32 %bar2) #0 {
entry:
  tail call void @llvm.dbg.value(metadata !{i32 %bar2}, i64 0, metadata !13), !dbg !21
  ret i32 %bar2, !dbg !21
}

; Function Attrs: nounwind readonly uwtable
define i32 @main() #1 {
entry:
  %call = tail call i32 @_Z3fooi(i32 2), !dbg !22
  %call1 = tail call i32 @_Z4foo2i(i32 1), !dbg !22
  %add = add nsw i32 %call1, %call, !dbg !22
  %0 = load i32* @global, align 4, !dbg !22, !tbaa !23
  %add2 = add nsw i32 %add, %0, !dbg !22
  ret i32 %add2, !dbg !22
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata) #2

attributes #0 = { nounwind readnone uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readonly uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!19, !26}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 (191881)", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !17, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/tmp/debug_ranges/a.cc] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"tmp/debug_ranges/a.cc", metadata !"/"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4, metadata !11, metadata !14}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"foo", metadata !"foo", metadata !"_Z3fooi", i32 2, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 (i32)* @_Z3fooi, null, null, metadata !9, i32 2} ; [ DW_TAG_subprogram ] [line 2] [def] [foo]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/tmp/debug_ranges/a.cc]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !8}
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{metadata !10}
!10 = metadata !{i32 786689, metadata !4, metadata !"bar", metadata !5, i32 16777218, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [bar] [line 2]
!11 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"foo2", metadata !"foo2", metadata !"_Z4foo2i", i32 3, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 (i32)* @_Z4foo2i, null, null, metadata !12, i32 3} ; [ DW_TAG_subprogram ] [line 3] [def] [foo2]
!12 = metadata !{metadata !13}
!13 = metadata !{i32 786689, metadata !11, metadata !"bar2", metadata !5, i32 16777219, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [bar2] [line 3]
!14 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"main", metadata !"main", metadata !"", i32 5, metadata !15, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 ()* @main, null, null, metadata !2, i32 5} ; [ DW_TAG_subprogram ] [line 5] [def] [main]
!15 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !16, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = metadata !{metadata !8}
!17 = metadata !{metadata !18}
!18 = metadata !{i32 786484, i32 0, null, metadata !"global", metadata !"global", metadata !"", metadata !5, i32 1, metadata !8, i32 0, i32 1, i32* @global, null} ; [ DW_TAG_variable ] [global] [line 1] [def]
!19 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!20 = metadata !{i32 2, i32 0, metadata !4, null}
!21 = metadata !{i32 3, i32 0, metadata !11, null}
!22 = metadata !{i32 6, i32 0, metadata !14, null}
!23 = metadata !{metadata !"int", metadata !24}
!24 = metadata !{metadata !"omnipotent char", metadata !25}
!25 = metadata !{metadata !"Simple C/C++ TBAA"}
!26 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
