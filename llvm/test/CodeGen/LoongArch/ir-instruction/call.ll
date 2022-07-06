; RUN: llc --mtriple=loongarch32 < %s | FileCheck --check-prefix=LA32 %s
; RUN: llc --mtriple=loongarch64 < %s | FileCheck --check-prefix=LA64 %s

;; FIXME: prologue and epilogue insertion must be implemented to complete this
;; test

declare i32 @external_function(i32)

define i32 @test_call_external(i32 %a) nounwind {
; LA32-LABEL: test_call_external:
; LA32:       # %bb.0:
; LA32-NEXT:    st.w $ra, $sp, 12 # 4-byte Folded Spill
; LA32-NEXT:    bl external_function
; LA32-NEXT:    ld.w $ra, $sp, 12 # 4-byte Folded Reload
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: test_call_external:
; LA64:       # %bb.0:
; LA64-NEXT:    st.d $ra, $sp, 8 # 8-byte Folded Spill
; LA64-NEXT:    bl external_function
; LA64-NEXT:    ld.d $ra, $sp, 8 # 8-byte Folded Reload
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = call i32 @external_function(i32 %a)
  ret i32 %1
}

define i32 @defined_function(i32 %a) nounwind {
; LA32-LABEL: defined_function:
; LA32:       # %bb.0:
; LA32-NEXT:    addi.w $a0, $a0, 1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: defined_function:
; LA64:       # %bb.0:
; LA64-NEXT:    addi.d $a0, $a0, 1
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = add i32 %a, 1
  ret i32 %1
}

define i32 @test_call_defined(i32 %a) nounwind {
; LA32-LABEL: test_call_defined:
; LA32:       # %bb.0:
; LA32-NEXT:    st.w $ra, $sp, 12 # 4-byte Folded Spill
; LA32-NEXT:    bl defined_function
; LA32-NEXT:    ld.w $ra, $sp, 12 # 4-byte Folded Reload
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: test_call_defined:
; LA64:       # %bb.0:
; LA64-NEXT:    st.d $ra, $sp, 8 # 8-byte Folded Spill
; LA64-NEXT:    bl defined_function
; LA64-NEXT:    ld.d $ra, $sp, 8 # 8-byte Folded Reload
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = call i32 @defined_function(i32 %a) nounwind
  ret i32 %1
}

define i32 @test_call_indirect(ptr %a, i32 %b) nounwind {
; LA32-LABEL: test_call_indirect:
; LA32:       # %bb.0:
; LA32-NEXT:    st.w $ra, $sp, 12 # 4-byte Folded Spill
; LA32-NEXT:    move $a2, $a0
; LA32-NEXT:    move $a0, $a1
; LA32-NEXT:    jirl $ra, $a2, 0
; LA32-NEXT:    ld.w $ra, $sp, 12 # 4-byte Folded Reload
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: test_call_indirect:
; LA64:       # %bb.0:
; LA64-NEXT:    st.d $ra, $sp, 8 # 8-byte Folded Spill
; LA64-NEXT:    move $a2, $a0
; LA64-NEXT:    move $a0, $a1
; LA64-NEXT:    jirl $ra, $a2, 0
; LA64-NEXT:    ld.d $ra, $sp, 8 # 8-byte Folded Reload
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = call i32 %a(i32 %b)
  ret i32 %1
}
