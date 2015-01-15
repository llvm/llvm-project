; RUN: llc < %s -march=x86-64 -mtriple=x86_64-unknown-unknown -mcpu=generic | FileCheck %s
; Both functions should produce the same code. The presence of debug values
; should not affect the scheduling strategy.
; Generated from:
; char argc;
; class C {
; public:
;   int test(char ,char ,char ,...);
; };
; void foo() {
;   C c;
;   char lc = argc;
;   c.test(0,argc,0,lc);
;   c.test(0,argc,0,lc);
; }
;
; with
; clang -O2 -c test.cpp -emit-llvm -S
; clang -O2 -c test.cpp -emit-llvm -S -g
;


%class.C = type { i8 }

@argc = global i8 0, align 1

declare i32 @test_function(%class.C*, i8 signext, i8 signext, i8 signext, ...)

; CHECK-LABEL: test_without_debug
; CHECK: movl [[A:%[a-z]+]], [[B:%[a-z]+]]
; CHECK-NEXT: movl [[A]], [[C:%[a-z]+]]
define void @test_without_debug() {
entry:
  %c = alloca %class.C, align 1
  %0 = load i8* @argc, align 1
  %conv = sext i8 %0 to i32
  %call = call i32 (%class.C*, i8, i8, i8, ...)* @test_function(%class.C* %c, i8 signext 0, i8 signext %0, i8 signext 0, i32 %conv)
  %1 = load i8* @argc, align 1
  %call2 = call i32 (%class.C*, i8, i8, i8, ...)* @test_function(%class.C* %c, i8 signext 0, i8 signext %1, i8 signext 0, i32 %conv)
  ret void
}

; CHECK-LABEL: test_with_debug
; CHECK: movl [[A]], [[B]]
; CHECK-NEXT: movl [[A]], [[C]]
define void @test_with_debug() {
entry:
  %c = alloca %class.C, align 1
  %0 = load i8* @argc, align 1
  tail call void @llvm.dbg.value(metadata i8 %0, i64 0, metadata !19, metadata !29)
  %conv = sext i8 %0 to i32
  tail call void @llvm.dbg.value(metadata %class.C* %c, i64 0, metadata !18, metadata !29)
  %call = call i32 (%class.C*, i8, i8, i8, ...)* @test_function(%class.C* %c, i8 signext 0, i8 signext %0, i8 signext 0, i32 %conv)
  %1 = load i8* @argc, align 1
  call void @llvm.dbg.value(metadata %class.C* %c, i64 0, metadata !18, metadata !29)
  %call2 = call i32 (%class.C*, i8, i8, i8, ...)* @test_function(%class.C* %c, i8 signext 0, i8 signext %1, i8 signext 0, i32 %conv)
  ret void
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22, !23}

!0 = !{!"", !1, !2, !3, !12, !20, !2} ; [ DW_TAG_compile_unit ] [test.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"test.cpp", !""}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2\00C\002\008\008\000\000\000", !1, null, null, !5, null, null, !"_ZTS1C"} ; [ DW_TAG_class_type ] [C] [line 2, size 8, align 8, offset 0] [def] [from ]
!5 = !{!6}
!6 = !{!"", !1, !"_ZTS1C", !7, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 4] [public] [test]
!7 = !{!"", null, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9, !10, !11, !11, !11, null}
!9 = !{!"", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = !{!"", null, null, !"_ZTS1C"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1C]
!11 = !{!"0x24\00char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ] [char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]
!12 = !{!13}
!13 = !{!"0x2e\00test_with_debug\00test_with_debug\00test_with_debug\006\000\001\000\000\00256\001\006", !1, !14, !15, null, void ()* @test_with_debug, null, null, !17} ; [ DW_TAG_subprogram ] [line 6] [def] [test_with_debug]
!14 = !{!"0x29", !1}
!15 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !16, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = !{null}
!17 = !{!18, !19}
!18 = !{!"0x100\00c\007\000", !13, !14, !"_ZTS1C"} ; [ DW_TAG_auto_variable ] [c] [line 7]
!19 = !{!"0x100\00lc\008\000", !13, !14, !11} ; [ DW_TAG_auto_variable ] [lc] [line 8]
!20 = !{!21}
!21 = !{!"0x34\00argc\00argc\00\001\000\001", null, !14, !11, i8* @argc, null} ; [ DW_TAG_variable ] [argc] [line 1] [def]
!22 = !{i32 2, !"Dwarf Version", i32 4}
!23 = !{i32 2, !"Debug Info Version", i32 2}
!25 = !MDLocation(line: 8, column: 3, scope: !13)
!29 = !{!"0x102"}               ; [ DW_TAG_expression ]
