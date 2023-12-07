# REQUIRES: asserts
# RUN: llvm-mc --triple=loongarch64-linux-gnu --filetype=obj -o %t %s
# RUN: llvm-jitlink --noexec --phony-externals --debug-only=jitlink %t 2>&1 | \
# RUN:   FileCheck %s

## Check that splitting of eh-frame sections works.

# CHECK: DWARFRecordSectionSplitter: Processing .eh_frame...
# CHECK:  Processing block at
# CHECK:    Processing CFI record at
# CHECK:      Extracted {{.*}} section = .eh_frame
# CHECK:    Processing CFI record at
# CHECK:      Extracted {{.*}} section = .eh_frame
# CHECK: EHFrameEdgeFixer: Processing .eh_frame in "{{.*}}"...
# CHECK:   Processing block at
# CHECK:     Processing CFI record at
# CHECK:       Record is CIE
# CHECK:   Processing block at
# CHECK:     Processing CFI record at
# CHECK:       Record is FDE
# CHECK:         Adding edge at {{.*}} to CIE at: {{.*}}
# CHECK:         Existing edge at {{.*}} to PC begin at {{.*}}
# CHECK:         Adding keep-alive edge from target at {{.*}} to FDE at {{.*}}
# CHECK:   Processing block at
# CHECK:     Processing CFI record at
# CHECK:       Record is FDE
# CHECK:         Adding edge at {{.*}} to CIE at: {{.*}}
# CHECK:         Existing edge at {{.*}} to PC begin at {{.*}}
# CHECK:         Adding keep-alive edge from target at {{.*}} to FDE at {{.*}}

 .text
 .globl main
 .p2align 2
 .type main,@function
main:
 .cfi_startproc
 addi.d $sp, $sp, -16
 .cfi_def_cfa_offset 16
 st.d $ra, $sp, 8
 .cfi_offset 1, -8
 ori $a0, $zero, 4
 bl %plt(__cxa_allocate_exception)
 ori $a1, $zero, 5
 st.w $a1, $a0, 0
 pcalau12i $a1, %got_pc_hi20(_ZTIi)
 ld.d $a1, $a1, %got_pc_lo12(_ZTIi)
 move $a2, $zero
 bl %plt(__cxa_throw)
.main_end:
 .size main, .main_end-main
 .cfi_endproc

 .globl dup
 .p2align 2
 .type main,@function
dup:
 .cfi_startproc
 addi.d $sp, $sp, -16
 .cfi_def_cfa_offset 16
 st.d $ra, $sp, 8
 .cfi_offset 1, -8
 ori $a0, $zero, 4
 bl %plt(__cxa_allocate_exception)
 ori $a1, $zero, 5
 st.w $a1, $a0, 0
 pcalau12i $a1, %got_pc_hi20(_ZTIi)
 ld.d $a1, $a1, %got_pc_lo12(_ZTIi)
 move $a2, $zero
 bl %plt(__cxa_throw)
.dup_end:
 .size main, .dup_end-dup
 .cfi_endproc
