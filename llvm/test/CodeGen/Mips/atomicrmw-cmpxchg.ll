; RUN: llc -mtriple=mipsel -mcpu=mips32r2 -verify-machineinstrs < %s | FileCheck %s --check-prefix=NATIVE
; RUN: llc -mtriple=mipsel -mcpu=mips32r6 -O0 -verify-machineinstrs < %s | FileCheck %s --check-prefix=NATIVE
; RUN: llc -mtriple=mips64el -mcpu=mips64r2 -verify-machineinstrs < %s | FileCheck %s --check-prefix=NATIVE
; RUN: llc -mtriple=mips64el -mcpu=mips64r6 -O0 -verify-machineinstrs < %s | FileCheck %s --check-prefix=NATIVE
; RUN: llc -mtriple=mipsel -mattr=+mips16 -verify-machineinstrs < %s | FileCheck %s --check-prefix=MIPS16

define i8 @uinc_wrap_i8(ptr %ptr, i8 %value) {
; NATIVE-LABEL: uinc_wrap_i8:
; NATIVE: [[RMW:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; NATIVE: [[LLSC:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; NATIVE: {{^[ \t]+}}ll $[[OLD:[0-9]+]], 0($[[PTR:[0-9]+]])
; NATIVE-NEXT: {{[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP:[0-9]+]], [[CMPDONE:(\$BB|\.LBB)[0-9]+_[0-9]+]]
; NATIVE-NOT: {{^[ \t]+s[bhwd]([lr]|c[0-3])?[ \t]}}
; NATIVE: {{^[ \t]+}}sc $[[SC:[0-9]+]], 0($[[PTR]])
; NATIVE-NEXT: {{[ \t]+}}beqz{{c?}} $[[SC]], [[LLSC]]
; NATIVE: [[CMPDONE]]:
; NATIVE: {{^[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP]], [[RMW]]
; MIPS16-LABEL: uinc_wrap_i8:
; MIPS16: {{^[ \t]+}}jal __sync_val_compare_and_swap_1
; MIPS16: {{^[ \t]+}}b [[COMPUTE:\$BB[0-9]+_[0-9]+]]
; MIPS16: [[RETRY:\$BB[0-9]+_[0-9]+]]:
; MIPS16: {{^[ \t]+}}jal __sync_val_compare_and_swap_1
; MIPS16: {{^[ \t]+}}cmp ${{[0-9]+}}, ${{[0-9]+}}
; MIPS16: {{^[ \t]+}}bteqz [[DONE:\$BB[0-9]+_[0-9]+]]
; MIPS16: [[COMPUTE]]:
; MIPS16: {{^[ \t]+}}b [[RETRY]]
; MIPS16: [[DONE]]:
  %old = atomicrmw uinc_wrap ptr %ptr, i8 %value seq_cst
  ret i8 %old
}

define i8 @udec_wrap_i8(ptr %ptr, i8 %value) {
; NATIVE-LABEL: udec_wrap_i8:
; NATIVE: [[RMW:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; NATIVE: [[LLSC:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; NATIVE: {{^[ \t]+}}ll $[[OLD:[0-9]+]], 0($[[PTR:[0-9]+]])
; NATIVE-NEXT: {{[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP:[0-9]+]], [[CMPDONE:(\$BB|\.LBB)[0-9]+_[0-9]+]]
; NATIVE-NOT: {{^[ \t]+s[bhwd]([lr]|c[0-3])?[ \t]}}
; NATIVE: {{^[ \t]+}}sc $[[SC:[0-9]+]], 0($[[PTR]])
; NATIVE-NEXT: {{[ \t]+}}beqz{{c?}} $[[SC]], [[LLSC]]
; NATIVE: [[CMPDONE]]:
; NATIVE: {{^[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP]], [[RMW]]
; MIPS16-LABEL: udec_wrap_i8:
; MIPS16: {{^[ \t]+}}jal __sync_val_compare_and_swap_1
; MIPS16: {{^[ \t]+}}b [[COMPUTE:\$BB[0-9]+_[0-9]+]]
; MIPS16: [[RETRY:\$BB[0-9]+_[0-9]+]]:
; MIPS16: {{^[ \t]+}}jal __sync_val_compare_and_swap_1
; MIPS16: {{^[ \t]+}}cmp ${{[0-9]+}}, ${{[0-9]+}}
; MIPS16: {{^[ \t]+}}bteqz [[DONE:\$BB[0-9]+_[0-9]+]]
; MIPS16: [[COMPUTE]]:
; MIPS16: {{^[ \t]+}}b [[RETRY]]
; MIPS16: [[DONE]]:
  %old = atomicrmw udec_wrap ptr %ptr, i8 %value seq_cst
  ret i8 %old
}

define i8 @usub_cond_i8(ptr %ptr, i8 %value) {
; NATIVE-LABEL: usub_cond_i8:
; NATIVE: [[RMW:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; NATIVE: [[LLSC:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; NATIVE: {{^[ \t]+}}ll $[[OLD:[0-9]+]], 0($[[PTR:[0-9]+]])
; NATIVE-NEXT: {{[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP:[0-9]+]], [[CMPDONE:(\$BB|\.LBB)[0-9]+_[0-9]+]]
; NATIVE-NOT: {{^[ \t]+s[bhwd]([lr]|c[0-3])?[ \t]}}
; NATIVE: {{^[ \t]+}}sc $[[SC:[0-9]+]], 0($[[PTR]])
; NATIVE-NEXT: {{[ \t]+}}beqz{{c?}} $[[SC]], [[LLSC]]
; NATIVE: [[CMPDONE]]:
; NATIVE: {{^[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP]], [[RMW]]
; MIPS16-LABEL: usub_cond_i8:
; MIPS16: {{^[ \t]+}}jal __sync_val_compare_and_swap_1
; MIPS16: {{^[ \t]+}}b [[COMPUTE:\$BB[0-9]+_[0-9]+]]
; MIPS16: {{^\$BB[0-9]+_[0-9]+:}}
; MIPS16: [[RETRY:\$BB[0-9]+_[0-9]+]]:
; MIPS16: {{^[ \t]+}}jal __sync_val_compare_and_swap_1
; MIPS16: {{^[ \t]+}}cmp ${{[0-9]+}}, ${{[0-9]+}}
; MIPS16: {{^[ \t]+}}bteqz [[DONE:\$BB[0-9]+_[0-9]+]]
; MIPS16: [[COMPUTE]]:
; MIPS16: {{^[ \t]+}}b [[RETRY]]
; MIPS16: [[DONE]]:
  %old = atomicrmw usub_cond ptr %ptr, i8 %value seq_cst
  ret i8 %old
}

define i8 @usub_sat_i8(ptr %ptr, i8 %value) {
; NATIVE-LABEL: usub_sat_i8:
; NATIVE: [[RMW:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; NATIVE: [[LLSC:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; NATIVE: {{^[ \t]+}}ll $[[OLD:[0-9]+]], 0($[[PTR:[0-9]+]])
; NATIVE-NEXT: {{[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP:[0-9]+]], [[CMPDONE:(\$BB|\.LBB)[0-9]+_[0-9]+]]
; NATIVE-NOT: {{^[ \t]+s[bhwd]([lr]|c[0-3])?[ \t]}}
; NATIVE: {{^[ \t]+}}sc $[[SC:[0-9]+]], 0($[[PTR]])
; NATIVE-NEXT: {{[ \t]+}}beqz{{c?}} $[[SC]], [[LLSC]]
; NATIVE: [[CMPDONE]]:
; NATIVE: {{^[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP]], [[RMW]]
; MIPS16-LABEL: usub_sat_i8:
; MIPS16: {{^[ \t]+}}jal __sync_val_compare_and_swap_1
; MIPS16: {{^[ \t]+}}b [[COMPUTE:\$BB[0-9]+_[0-9]+]]
; MIPS16: [[RETRY:\$BB[0-9]+_[0-9]+]]:
; MIPS16: {{^[ \t]+}}jal __sync_val_compare_and_swap_1
; MIPS16: {{^[ \t]+}}cmp ${{[0-9]+}}, ${{[0-9]+}}
; MIPS16: {{^[ \t]+}}bteqz [[DONE:\$BB[0-9]+_[0-9]+]]
; MIPS16: [[COMPUTE]]:
; MIPS16: {{^[ \t]+}}b [[RETRY]]
; MIPS16: [[DONE]]:
  %old = atomicrmw usub_sat ptr %ptr, i8 %value seq_cst
  ret i8 %old
}

define i16 @uinc_wrap_i16(ptr %ptr, i16 %value) {
; NATIVE-LABEL: uinc_wrap_i16:
; NATIVE: [[RMW:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; NATIVE: [[LLSC:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; NATIVE: {{^[ \t]+}}ll $[[OLD:[0-9]+]], 0($[[PTR:[0-9]+]])
; NATIVE-NEXT: {{[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP:[0-9]+]], [[CMPDONE:(\$BB|\.LBB)[0-9]+_[0-9]+]]
; NATIVE-NOT: {{^[ \t]+s[bhwd]([lr]|c[0-3])?[ \t]}}
; NATIVE: {{^[ \t]+}}sc $[[SC:[0-9]+]], 0($[[PTR]])
; NATIVE-NEXT: {{[ \t]+}}beqz{{c?}} $[[SC]], [[LLSC]]
; NATIVE: [[CMPDONE]]:
; NATIVE: {{^[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP]], [[RMW]]
; MIPS16-LABEL: uinc_wrap_i16:
; MIPS16: {{^[ \t]+}}jal __sync_val_compare_and_swap_2
; MIPS16: {{^[ \t]+}}b [[COMPUTE:\$BB[0-9]+_[0-9]+]]
; MIPS16: [[RETRY:\$BB[0-9]+_[0-9]+]]:
; MIPS16: {{^[ \t]+}}jal __sync_val_compare_and_swap_2
; MIPS16: {{^[ \t]+}}cmp ${{[0-9]+}}, ${{[0-9]+}}
; MIPS16: {{^[ \t]+}}bteqz [[DONE:\$BB[0-9]+_[0-9]+]]
; MIPS16: [[COMPUTE]]:
; MIPS16: {{^[ \t]+}}b [[RETRY]]
; MIPS16: [[DONE]]:
  %old = atomicrmw uinc_wrap ptr %ptr, i16 %value seq_cst
  ret i16 %old
}

define i16 @udec_wrap_i16(ptr %ptr, i16 %value) {
; NATIVE-LABEL: udec_wrap_i16:
; NATIVE: [[RMW:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; NATIVE: [[LLSC:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; NATIVE: {{^[ \t]+}}ll $[[OLD:[0-9]+]], 0($[[PTR:[0-9]+]])
; NATIVE-NEXT: {{[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP:[0-9]+]], [[CMPDONE:(\$BB|\.LBB)[0-9]+_[0-9]+]]
; NATIVE-NOT: {{^[ \t]+s[bhwd]([lr]|c[0-3])?[ \t]}}
; NATIVE: {{^[ \t]+}}sc $[[SC:[0-9]+]], 0($[[PTR]])
; NATIVE-NEXT: {{[ \t]+}}beqz{{c?}} $[[SC]], [[LLSC]]
; NATIVE: [[CMPDONE]]:
; NATIVE: {{^[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP]], [[RMW]]
; MIPS16-LABEL: udec_wrap_i16:
; MIPS16: {{^[ \t]+}}jal __sync_val_compare_and_swap_2
; MIPS16: {{^[ \t]+}}b [[COMPUTE:\$BB[0-9]+_[0-9]+]]
; MIPS16: [[RETRY:\$BB[0-9]+_[0-9]+]]:
; MIPS16: {{^[ \t]+}}jal __sync_val_compare_and_swap_2
; MIPS16: {{^[ \t]+}}cmp ${{[0-9]+}}, ${{[0-9]+}}
; MIPS16: {{^[ \t]+}}bteqz [[DONE:\$BB[0-9]+_[0-9]+]]
; MIPS16: [[COMPUTE]]:
; MIPS16: {{^[ \t]+}}b [[RETRY]]
; MIPS16: [[DONE]]:
  %old = atomicrmw udec_wrap ptr %ptr, i16 %value seq_cst
  ret i16 %old
}

define i16 @usub_cond_i16(ptr %ptr, i16 %value) {
; NATIVE-LABEL: usub_cond_i16:
; NATIVE: [[RMW:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; NATIVE: [[LLSC:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; NATIVE: {{^[ \t]+}}ll $[[OLD:[0-9]+]], 0($[[PTR:[0-9]+]])
; NATIVE-NEXT: {{[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP:[0-9]+]], [[CMPDONE:(\$BB|\.LBB)[0-9]+_[0-9]+]]
; NATIVE-NOT: {{^[ \t]+s[bhwd]([lr]|c[0-3])?[ \t]}}
; NATIVE: {{^[ \t]+}}sc $[[SC:[0-9]+]], 0($[[PTR]])
; NATIVE-NEXT: {{[ \t]+}}beqz{{c?}} $[[SC]], [[LLSC]]
; NATIVE: [[CMPDONE]]:
; NATIVE: {{^[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP]], [[RMW]]
; MIPS16-LABEL: usub_cond_i16:
; MIPS16: {{^[ \t]+}}jal __sync_val_compare_and_swap_2
; MIPS16: {{^[ \t]+}}b [[COMPUTE:\$BB[0-9]+_[0-9]+]]
; MIPS16: {{^\$BB[0-9]+_[0-9]+:}}
; MIPS16: [[RETRY:\$BB[0-9]+_[0-9]+]]:
; MIPS16: {{^[ \t]+}}jal __sync_val_compare_and_swap_2
; MIPS16: {{^[ \t]+}}cmp ${{[0-9]+}}, ${{[0-9]+}}
; MIPS16: {{^[ \t]+}}bteqz [[DONE:\$BB[0-9]+_[0-9]+]]
; MIPS16: [[COMPUTE]]:
; MIPS16: {{^[ \t]+}}b [[RETRY]]
; MIPS16: [[DONE]]:
  %old = atomicrmw usub_cond ptr %ptr, i16 %value seq_cst
  ret i16 %old
}

define i16 @usub_sat_i16(ptr %ptr, i16 %value) {
; NATIVE-LABEL: usub_sat_i16:
; NATIVE: [[RMW:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; NATIVE: [[LLSC:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; NATIVE: {{^[ \t]+}}ll $[[OLD:[0-9]+]], 0($[[PTR:[0-9]+]])
; NATIVE-NEXT: {{[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP:[0-9]+]], [[CMPDONE:(\$BB|\.LBB)[0-9]+_[0-9]+]]
; NATIVE-NOT: {{^[ \t]+s[bhwd]([lr]|c[0-3])?[ \t]}}
; NATIVE: {{^[ \t]+}}sc $[[SC:[0-9]+]], 0($[[PTR]])
; NATIVE-NEXT: {{[ \t]+}}beqz{{c?}} $[[SC]], [[LLSC]]
; NATIVE: [[CMPDONE]]:
; NATIVE: {{^[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP]], [[RMW]]
; MIPS16-LABEL: usub_sat_i16:
; MIPS16: {{^[ \t]+}}jal __sync_val_compare_and_swap_2
; MIPS16: {{^[ \t]+}}b [[COMPUTE:\$BB[0-9]+_[0-9]+]]
; MIPS16: [[RETRY:\$BB[0-9]+_[0-9]+]]:
; MIPS16: {{^[ \t]+}}jal __sync_val_compare_and_swap_2
; MIPS16: {{^[ \t]+}}cmp ${{[0-9]+}}, ${{[0-9]+}}
; MIPS16: {{^[ \t]+}}bteqz [[DONE:\$BB[0-9]+_[0-9]+]]
; MIPS16: [[COMPUTE]]:
; MIPS16: {{^[ \t]+}}b [[RETRY]]
; MIPS16: [[DONE]]:
  %old = atomicrmw usub_sat ptr %ptr, i16 %value seq_cst
  ret i16 %old
}

define i32 @uinc_wrap_i32(ptr %ptr, i32 %value) {
; NATIVE-LABEL: uinc_wrap_i32:
; NATIVE: [[RMW:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; NATIVE: [[LLSC:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; NATIVE: {{^[ \t]+}}ll $[[OLD:[0-9]+]], 0($[[PTR:[0-9]+]])
; NATIVE-NEXT: {{[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP:[0-9]+]], [[CMPDONE:(\$BB|\.LBB)[0-9]+_[0-9]+]]
; NATIVE-NOT: {{^[ \t]+s[bhwd]([lr]|c[0-3])?[ \t]}}
; NATIVE: {{^[ \t]+}}sc $[[SC:[0-9]+]], 0($[[PTR]])
; NATIVE-NEXT: {{[ \t]+}}beqz{{c?}} $[[SC]], [[LLSC]]
; NATIVE: [[CMPDONE]]:
; NATIVE: {{^[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP]], [[RMW]]
; MIPS16-LABEL: uinc_wrap_i32:
; MIPS16: {{^[ \t]+}}jal __sync_val_compare_and_swap_4
; MIPS16: {{^[ \t]+}}b [[COMPUTE:\$BB[0-9]+_[0-9]+]]
; MIPS16: [[RETRY:\$BB[0-9]+_[0-9]+]]:
; MIPS16: {{^[ \t]+}}jal __sync_val_compare_and_swap_4
; MIPS16: {{^[ \t]+}}cmp ${{[0-9]+}}, ${{[0-9]+}}
; MIPS16: {{^[ \t]+}}bteqz [[DONE:\$BB[0-9]+_[0-9]+]]
; MIPS16: [[COMPUTE]]:
; MIPS16: {{^[ \t]+}}b [[RETRY]]
; MIPS16: [[DONE]]:
  %old = atomicrmw uinc_wrap ptr %ptr, i32 %value seq_cst
  ret i32 %old
}

define i32 @udec_wrap_i32(ptr %ptr, i32 %value) {
; NATIVE-LABEL: udec_wrap_i32:
; NATIVE: [[RMW:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; NATIVE: [[LLSC:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; NATIVE: {{^[ \t]+}}ll $[[OLD:[0-9]+]], 0($[[PTR:[0-9]+]])
; NATIVE-NEXT: {{[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP:[0-9]+]], [[CMPDONE:(\$BB|\.LBB)[0-9]+_[0-9]+]]
; NATIVE-NOT: {{^[ \t]+s[bhwd]([lr]|c[0-3])?[ \t]}}
; NATIVE: {{^[ \t]+}}sc $[[SC:[0-9]+]], 0($[[PTR]])
; NATIVE-NEXT: {{[ \t]+}}beqz{{c?}} $[[SC]], [[LLSC]]
; NATIVE: [[CMPDONE]]:
; NATIVE: {{^[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP]], [[RMW]]
; MIPS16-LABEL: udec_wrap_i32:
; MIPS16: {{^[ \t]+}}jal __sync_val_compare_and_swap_4
; MIPS16: {{^[ \t]+}}b [[COMPUTE:\$BB[0-9]+_[0-9]+]]
; MIPS16: [[RETRY:\$BB[0-9]+_[0-9]+]]:
; MIPS16: {{^[ \t]+}}jal __sync_val_compare_and_swap_4
; MIPS16: {{^[ \t]+}}cmp ${{[0-9]+}}, ${{[0-9]+}}
; MIPS16: {{^[ \t]+}}bteqz [[DONE:\$BB[0-9]+_[0-9]+]]
; MIPS16: [[COMPUTE]]:
; MIPS16: {{^[ \t]+}}b [[RETRY]]
; MIPS16: [[DONE]]:
  %old = atomicrmw udec_wrap ptr %ptr, i32 %value seq_cst
  ret i32 %old
}

define i32 @usub_cond_i32(ptr %ptr, i32 %value) {
; NATIVE-LABEL: usub_cond_i32:
; NATIVE: [[RMW:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; NATIVE: [[LLSC:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; NATIVE: {{^[ \t]+}}ll $[[OLD:[0-9]+]], 0($[[PTR:[0-9]+]])
; NATIVE-NEXT: {{[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP:[0-9]+]], [[CMPDONE:(\$BB|\.LBB)[0-9]+_[0-9]+]]
; NATIVE-NOT: {{^[ \t]+s[bhwd]([lr]|c[0-3])?[ \t]}}
; NATIVE: {{^[ \t]+}}sc $[[SC:[0-9]+]], 0($[[PTR]])
; NATIVE-NEXT: {{[ \t]+}}beqz{{c?}} $[[SC]], [[LLSC]]
; NATIVE: [[CMPDONE]]:
; NATIVE: {{^[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP]], [[RMW]]
; MIPS16-LABEL: usub_cond_i32:
; MIPS16: {{^[ \t]+}}jal __sync_val_compare_and_swap_4
; MIPS16: {{^[ \t]+}}b [[COMPUTE:\$BB[0-9]+_[0-9]+]]
; MIPS16: {{^\$BB[0-9]+_[0-9]+:}}
; MIPS16: [[RETRY:\$BB[0-9]+_[0-9]+]]:
; MIPS16: {{^[ \t]+}}jal __sync_val_compare_and_swap_4
; MIPS16: {{^[ \t]+}}cmp ${{[0-9]+}}, ${{[0-9]+}}
; MIPS16: {{^[ \t]+}}bteqz [[DONE:\$BB[0-9]+_[0-9]+]]
; MIPS16: [[COMPUTE]]:
; MIPS16: {{^[ \t]+}}b [[RETRY]]
; MIPS16: [[DONE]]:
  %old = atomicrmw usub_cond ptr %ptr, i32 %value seq_cst
  ret i32 %old
}

define i32 @usub_sat_i32(ptr %ptr, i32 %value) {
; NATIVE-LABEL: usub_sat_i32:
; NATIVE: [[RMW:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; NATIVE: [[LLSC:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; NATIVE: {{^[ \t]+}}ll $[[OLD:[0-9]+]], 0($[[PTR:[0-9]+]])
; NATIVE-NEXT: {{[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP:[0-9]+]], [[CMPDONE:(\$BB|\.LBB)[0-9]+_[0-9]+]]
; NATIVE-NOT: {{^[ \t]+s[bhwd]([lr]|c[0-3])?[ \t]}}
; NATIVE: {{^[ \t]+}}sc $[[SC:[0-9]+]], 0($[[PTR]])
; NATIVE-NEXT: {{[ \t]+}}beqz{{c?}} $[[SC]], [[LLSC]]
; NATIVE: [[CMPDONE]]:
; NATIVE: {{^[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP]], [[RMW]]
; MIPS16-LABEL: usub_sat_i32:
; MIPS16: {{^[ \t]+}}jal __sync_val_compare_and_swap_4
; MIPS16: {{^[ \t]+}}b [[COMPUTE:\$BB[0-9]+_[0-9]+]]
; MIPS16: [[RETRY:\$BB[0-9]+_[0-9]+]]:
; MIPS16: {{^[ \t]+}}jal __sync_val_compare_and_swap_4
; MIPS16: {{^[ \t]+}}cmp ${{[0-9]+}}, ${{[0-9]+}}
; MIPS16: {{^[ \t]+}}bteqz [[DONE:\$BB[0-9]+_[0-9]+]]
; MIPS16: [[COMPUTE]]:
; MIPS16: {{^[ \t]+}}b [[RETRY]]
; MIPS16: [[DONE]]:
  %old = atomicrmw usub_sat ptr %ptr, i32 %value seq_cst
  ret i32 %old
}
