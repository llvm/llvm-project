; REQUIRES: object-emission

; RUN: llvm-link %s %p/type-unique-simple-b.ll -S -o %t
; RUN: cat %t | FileCheck %s -check-prefix=LINK
; RUN: %llc_dwarf -filetype=obj -O0 < %t > %t2
; RUN: llvm-dwarfdump -debug-dump=info %t2 | FileCheck %s

; Make sure the backend generates a single DIE and uses ref_addr.
; CHECK: 0x[[BASE:.*]]: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}} = "Base"
; CHECK-NOT: DW_TAG_structure_type
; CHECK: 0x[[INT:.*]]: DW_TAG_base_type
; CHECK-NEXT: DW_AT_name {{.*}} = "int"
; CHECK-NOT: DW_TAG_base_type

; CHECK: DW_TAG_compile_unit
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_AT_type [DW_FORM_ref_addr] {{.*}}[[INT]])
; CHECK: DW_TAG_variable
; CHECK: DW_AT_type [DW_FORM_ref_addr] {{.*}}[[BASE]])

; Make sure llvm-link only generates a single copy of the struct.
; LINK: DW_TAG_structure_type
; LINK-NOT: DW_TAG_structure_type
; Content of header files:
; struct Base {
;   int a;
; };
; Content of foo.cpp:
; 
; #include "a.hpp"
; void f(int a) {
;   Base t;
; }
; Content of bar.cpp:
; 
; #include "a.hpp"
; void f(int);
; void g(int a) {
;   Base t;
; }
; int main() {
;   f(0);
;   g(1);
;   return 0;
; }
; ModuleID = 'foo.cpp'

%struct.Base = type { i32 }

; Function Attrs: nounwind ssp uwtable
define void @_Z1fi(i32 %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  %t = alloca %struct.Base, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !15, metadata !{!"0x102"}), !dbg !16
  call void @llvm.dbg.declare(metadata %struct.Base* %t, metadata !17, metadata !{!"0x102"}), !dbg !18
  ret void, !dbg !19
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14, !20}

!0 = !{!"0x11\004\00clang version 3.4 (http://llvm.org/git/clang.git c23b1db6268c8e7ce64026d57d1510c1aac200a0) (http://llvm.org/git/llvm.git 09b98fe3978eddefc2145adc1056cf21580ce945)\000\00\000\00\000", !1, !2, !3, !9, !2, !2} ; [ DW_TAG_compile_unit ] [/Users/mren/c_testing/type_unique_air/simple/foo.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"foo.cpp", !"/Users/mren/c_testing/type_unique_air/simple"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x13\00Base\001\0032\0032\000\000\000", !5, null, null, !6, null, null, !"_ZTS4Base"} ; [ DW_TAG_structure_type ] [Base] [line 1, size 32, align 32, offset 0] [def] [from ]
!5 = !{!"./a.hpp", !"/Users/mren/c_testing/type_unique_air/simple"}
!6 = !{!7}
!7 = !{!"0xd\00a\002\0032\0032\000\000", !5, !"_ZTS4Base", !8} ; [ DW_TAG_member ] [a] [line 2, size 32, align 32, offset 0] [from int]
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{!10}
!10 = !{!"0x2e\00f\00f\00_Z1fi\003\000\001\000\006\00256\000\003", !1, !11, !12, null, void (i32)* @_Z1fi, null, null, !2} ; [ DW_TAG_subprogram ] [line 3] [def] [f]
!11 = !{!"0x29", !1}         ; [ DW_TAG_file_type ] [/Users/mren/c_testing/type_unique_air/simple/foo.cpp]
!12 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !13, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!13 = !{null, !8}
!14 = !{i32 2, !"Dwarf Version", i32 2}
!15 = !{!"0x101\00a\0016777219\000", !10, !11, !8} ; [ DW_TAG_arg_variable ] [a] [line 3]
!16 = !MDLocation(line: 3, scope: !10)
!17 = !{!"0x100\00t\004\000", !10, !11, !4} ; [ DW_TAG_auto_variable ] [t] [line 4]
!18 = !MDLocation(line: 4, scope: !10)
!19 = !MDLocation(line: 5, scope: !10)
!20 = !{i32 1, !"Debug Info Version", i32 2}
