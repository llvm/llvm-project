; REQUIRES: object-emission
; This test is failing for powerpc64, because a location list for the
; variable 'c' is not generated at all. Temporary marking this test as XFAIL 
; for powerpc, until PR21881 is fixed.
; XFAIL: powerpc64

; RUN: %llc_dwarf -O2  -dwarf-version 2 -filetype=obj < %s | llvm-dwarfdump - | FileCheck %s  --check-prefix=DWARF23
; RUN: %llc_dwarf -O2  -dwarf-version 3 -filetype=obj < %s | llvm-dwarfdump - | FileCheck %s  --check-prefix=DWARF23
; RUN: %llc_dwarf -O2  -dwarf-version 4 -filetype=obj < %s | llvm-dwarfdump - | FileCheck %s  --check-prefix=DWARF4

; This is a test for PR21176.
; DW_OP_const <const> doesn't describe a constant value, but a value at a constant address. 
; The proper way to describe a constant value is DW_OP_constu <const>, DW_OP_stack_value.

; Generated with clang -S -emit-llvm -g -O2 test.cpp

; extern int func();
; 
; int main()
; {
;   volatile int c = 13;
;   c = func();
;   return c;
; }

; DWARF23: Location description: 10 0d {{$}}
; DWARF4: Location description: 10 0d 9f

; Function Attrs: uwtable
define i32 @main() #0 {
entry:
  %c = alloca i32, align 4
  tail call void @llvm.dbg.value(metadata i32 13, i64 0, metadata !10, metadata !16), !dbg !17
  store volatile i32 13, i32* %c, align 4, !dbg !18
  %call = tail call i32 @_Z4funcv(), !dbg !19
  tail call void @llvm.dbg.value(metadata i32 %call, i64 0, metadata !10, metadata !16), !dbg !17
  store volatile i32 %call, i32* %c, align 4, !dbg !19
  tail call void @llvm.dbg.value(metadata i32* %c, i64 0, metadata !10, metadata !16), !dbg !17
  %c.0.c.0. = load volatile i32* %c, align 4, !dbg !20
  ret i32 %c.0.c.0., !dbg !20
}

declare i32 @_Z4funcv() #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13}
!llvm.ident = !{!14}

!0 = !{!"0x11\004\00clang version 3.6.0 (trunk 223522)\001\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/home/kromanova/ngh/ToT_latest/llvm/test/DebugInfo/test.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"test.cpp", !"/home/kromanova/ngh/ToT_latest/llvm/test/DebugInfo"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00main\00main\00\003\000\001\000\000\00256\001\004", !1, !5, !6, null, i32 ()* @main, null, null, !9} ; [ DW_TAG_subprogram ] [line 3] [def] [scope 4] [main]
!5 = !{!"0x29", !1}    ; [ DW_TAG_file_type ] [/home/kromanova/ngh/ToT_latest/llvm/test/DebugInfo/test.cpp]
!6 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!8}
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{!10}
!10 = !{!"0x100\00c\005\000", !4, !5, !11} ; [ DW_TAG_auto_variable ] [c] [line 5]
!11 = !{!"0x35\00\000\000\000\000\000", null, null, !8} ; [ DW_TAG_volatile_type ] [line 0, size 0, align 0, offset 0] [from int]
!12 = !{i32 2, !"Dwarf Version", i32 2}
!13 = !{i32 2, !"Debug Info Version", i32 2}
!14 = !{!"clang version 3.6.0 (trunk 223522)"}
!15 = !{i32 13}
!16 = !{!"0x102"}               ; [ DW_TAG_expression ]
!17 = !MDLocation(line: 5, column: 16, scope: !4)
!18 = !MDLocation(line: 5, column: 3, scope: !4)
!19 = !MDLocation(line: 6, column: 7, scope: !4)
!20 = !MDLocation(line: 7, column: 3, scope: !4)

