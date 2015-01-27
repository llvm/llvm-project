; RUN: llc -mtriple=mips-linux-gnu -filetype=asm -asm-verbose=0 -O0 < %s | FileCheck %s
; RUN: llc -mtriple=mips-linux-gnu -filetype=obj -O0 < %s | llvm-dwarfdump -debug-dump=line - | FileCheck %s --check-prefix=INT

; Mips used to generate 'jumpy' debug line info around calls. The address
; calculation for each call to f1() would share the same line info so it would
; emit output of the form:
;   .loc $first_call_location
;   .. address calculation ..
;   .. function call ..
;   .. address calculation ..
;   .loc $second_call_location
;   .. function call ..
;   .loc $first_call_location
;   .. address calculation ..
;   .loc $third_call_location
;   .. function call ..
;   ...
; which would cause confusing stepping behaviour for the end user.
;
; This test checks that we emit more user friendly debug line info of the form:
;   .loc $first_call_location
;   .. address calculation ..
;   .. function call ..
;   .loc $second_call_location
;   .. address calculation ..
;   .. function call ..
;   .loc $third_call_location
;   .. address calculation ..
;   .. function call ..
;   ...
;
; Generated with clang from fn-call-line.c:
; void f1();
; void f2() {
;   f1();
;   f1();
; }

; CHECK: .loc	1 3 3
; CHECK-NOT: .loc
; CHECK: %call16(f1) 
; CHECK-NOT: .loc
; CHECK: .loc	1 4 3
; CHECK-NOT: .loc
; CHECK: %call16(f1) 

; INT: {{^}}Address
; INT: -----
; INT-NEXT: 2 0 1 0 0 is_stmt{{$}}
; INT-NEXT: 3 3 1 0 0 is_stmt prologue_end{{$}}
; INT-NEXT: 4 3 1 0 0 is_stmt{{$}}


; Function Attrs: nounwind uwtable
define void @f2() #0 {
entry:
  call void (...)* @f1(), !dbg !11
  call void (...)* @f1(), !dbg !12
  ret void, !dbg !13
}

declare void @f1(...) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = !{!"0x11\0012\00clang version 3.7.0 (trunk 226641)\000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/fn-call-line.c] [DW_LANG_C99]
!1 = !{!"fn-call-line.c", !"/tmp/dbginfo"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00f2\00f2\00\002\000\001\000\000\000\000\002", !1, !5, !6, null, void ()* @f2, null, null, !2} ; [ DW_TAG_subprogram ] [line 2] [def] [f2]
!5 = !{!"0x29", !1}                               ; [ DW_TAG_file_type ] [/tmp/dbginfo/fn-call-line.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null}
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 2}
!10 = !{!"clang version 3.7.0 (trunk 226641)"}
!11 = !MDLocation(line: 3, column: 3, scope: !4)
!12 = !MDLocation(line: 4, column: 3, scope: !4)
!13 = !MDLocation(line: 5, column: 1, scope: !4)
