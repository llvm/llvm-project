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
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !20, metadata !{!"0x102"}), !dbg !21
  call void @llvm.dbg.declare(metadata %class.A* %t, metadata !22, metadata !{!"0x102"}), !dbg !23
  ret void, !dbg !24
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!19, !25}

!0 = !{!"0x11\004\00clang version 3.4 (http://llvm.org/git/clang.git f54e02f969d02d640103db73efc30c45439fceab) (http://llvm.org/git/llvm.git 284353b55896cb1babfaa7add7c0a363245342d2)\000\00\000\00\000", !1, !2, !3, !14, !2, !2} ; [ DW_TAG_compile_unit ] [/Users/mren/c_testing/type_unique_air/inher/foo.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"foo.cpp", !"/Users/mren/c_testing/type_unique_air/inher"}
!2 = !{i32 0}
!3 = !{!4, !8}
!4 = !{!"0x2\00A\003\0064\0032\000\000\000", !5, null, null, !6, null, null, !"_ZTS1A"} ; [ DW_TAG_class_type ] [A] [line 3, size 64, align 32, offset 0] [def] [from ]
!5 = !{!"./a.hpp", !"/Users/mren/c_testing/type_unique_air/inher"}
!6 = !{!7, !13}
!7 = !{!"0x1c\00\000\000\000\000\001", null, !"_ZTS1A", !8} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 0] [private] [from Base]
!8 = !{!"0x2\00Base\003\0032\0032\000\000\000", !9, null, null, !10, null, null, !"_ZTS4Base"} ; [ DW_TAG_class_type ] [Base] [line 3, size 32, align 32, offset 0] [def] [from ]
!9 = !{!"./b.hpp", !"/Users/mren/c_testing/type_unique_air/inher"}
!10 = !{!11}
!11 = !{!"0xd\00b\004\0032\0032\000\001", !9, !"_ZTS4Base", !12} ; [ DW_TAG_member ] [b] [line 4, size 32, align 32, offset 0] [private] [from int]
!12 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!13 = !{!"0xd\00x\004\0032\0032\0032\001", !5, !"_ZTS1A", !12} ; [ DW_TAG_member ] [x] [line 4, size 32, align 32, offset 32] [private] [from int]
!14 = !{!15}
!15 = !{!"0x2e\00f\00f\00_Z1fi\005\000\001\000\006\00256\000\005", !1, !16, !17, null, void (i32)* @_Z1fi, null, null, !2} ; [ DW_TAG_subprogram ] [line 5] [def] [f]
!16 = !{!"0x29", !1}         ; [ DW_TAG_file_type ] [/Users/mren/c_testing/type_unique_air/inher/foo.cpp]
!17 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !18, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = !{null, !12}
!19 = !{i32 2, !"Dwarf Version", i32 2}
!20 = !{!"0x101\00a\0016777221\000", !15, !16, !12} ; [ DW_TAG_arg_variable ] [a] [line 5]
!21 = !MDLocation(line: 5, scope: !15)
!22 = !{!"0x100\00t\006\000", !15, !16, !4} ; [ DW_TAG_auto_variable ] [t] [line 6]
!23 = !MDLocation(line: 6, scope: !15)
!24 = !MDLocation(line: 7, scope: !15)
!25 = !{i32 1, !"Debug Info Version", i32 2}
