; CHECK: [ DW_TAG_class_type ] [A]
; CHECK: [ DW_TAG_class_type ] [Base]
; CHECK: [ DW_TAG_class_type ] [B]
; CHECK-NOT: DW_TAG_class_type
; Content of header files:
; 
; class Base;
; class A : Base {
;   int x;
; };
; 
; class A;
; class Base {
;   int b;
; };
; 
; class B {
;   int bb;
;   A *a;
; };
; Content of foo.cpp:
; 
; #include "b.hpp"
; #include "a.hpp"
; 
; void f(int a) {
;   A t;
; }
; Content of bar.cpp:
; 
; #include "b.hpp"
; #include "a.hpp"
; void g(int a) {
;   B t;
; }
; 
; void f(int);
; int main() {
;   A a;
;   f(0);
;   g(1);
;   return 0;
; }
; ModuleID = 'foo.cpp'

%class.A = type { %class.Base, i32 }
%class.Base = type { i32 }

; Function Attrs: nounwind ssp uwtable
define void @_Z1fi(i32 %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  %t = alloca %class.A, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %a.addr}, metadata !20), !dbg !21
  call void @llvm.dbg.declare(metadata !{%class.A* %t}, metadata !22), !dbg !23
  ret void, !dbg !24
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!19, !25}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 (http://llvm.org/git/clang.git f54e02f969d02d640103db73efc30c45439fceab) (http://llvm.org/git/llvm.git 284353b55896cb1babfaa7add7c0a363245342d2)", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !14, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/Users/mren/c_testing/type_unique_air/inher/foo.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"foo.cpp", metadata !"/Users/mren/c_testing/type_unique_air/inher"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4, metadata !8}
!4 = metadata !{i32 786434, metadata !5, null, metadata !"A", i32 3, i64 64, i64 32, i32 0, i32 0, null, metadata !6, i32 0, null, null, metadata !"_ZTS1A"} ; [ DW_TAG_class_type ] [A] [line 3, size 64, align 32, offset 0] [def] [from ]
!5 = metadata !{metadata !"./a.hpp", metadata !"/Users/mren/c_testing/type_unique_air/inher"}
!6 = metadata !{metadata !7, metadata !13}
!7 = metadata !{i32 786460, null, metadata !"_ZTS1A", null, i32 0, i64 0, i64 0, i64 0, i32 1, metadata !8} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 0] [private] [from Base]
!8 = metadata !{i32 786434, metadata !9, null, metadata !"Base", i32 3, i64 32, i64 32, i32 0, i32 0, null, metadata !10, i32 0, null, null, metadata !"_ZTS4Base"} ; [ DW_TAG_class_type ] [Base] [line 3, size 32, align 32, offset 0] [def] [from ]
!9 = metadata !{metadata !"./b.hpp", metadata !"/Users/mren/c_testing/type_unique_air/inher"}
!10 = metadata !{metadata !11}
!11 = metadata !{i32 786445, metadata !9, metadata !"_ZTS4Base", metadata !"b", i32 4, i64 32, i64 32, i64 0, i32 1, metadata !12} ; [ DW_TAG_member ] [b] [line 4, size 32, align 32, offset 0] [private] [from int]
!12 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!13 = metadata !{i32 786445, metadata !5, metadata !"_ZTS1A", metadata !"x", i32 4, i64 32, i64 32, i64 32, i32 1, metadata !12} ; [ DW_TAG_member ] [x] [line 4, size 32, align 32, offset 32] [private] [from int]
!14 = metadata !{metadata !15}
!15 = metadata !{i32 786478, metadata !1, metadata !16, metadata !"f", metadata !"f", metadata !"_Z1fi", i32 5, metadata !17, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i32)* @_Z1fi, null, null, metadata !2, i32 5} ; [ DW_TAG_subprogram ] [line 5] [def] [f]
!16 = metadata !{i32 786473, metadata !1}         ; [ DW_TAG_file_type ] [/Users/mren/c_testing/type_unique_air/inher/foo.cpp]
!17 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !18, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = metadata !{null, metadata !12}
!19 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!20 = metadata !{i32 786689, metadata !15, metadata !"a", metadata !16, i32 16777221, metadata !12, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [a] [line 5]
!21 = metadata !{i32 5, i32 0, metadata !15, null}
!22 = metadata !{i32 786688, metadata !15, metadata !"t", metadata !16, i32 6, metadata !4, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [t] [line 6]
!23 = metadata !{i32 6, i32 0, metadata !15, null}
!24 = metadata !{i32 7, i32 0, metadata !15, null}
!25 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
