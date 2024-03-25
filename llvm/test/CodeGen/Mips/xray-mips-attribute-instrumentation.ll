; RUN: llc -mtriple=mips-unknown-linux-gnu < %s | FileCheck --check-prefix=CHECK --check-prefix=CHECK-MIPS32 %s
; RUN: llc -mtriple=mipsel-unknown-linux-gnu < %s | FileCheck --check-prefix=CHECK --check-prefix=CHECK-MIPS32 %s
; RUN: llc -mtriple=mips64-unknown-linux-gnu < %s | FileCheck --check-prefix=CHECK --check-prefix=CHECK-MIPS64 %s
; RUN: llc -mtriple=mips64el-unknown-linux-gnu < %s | FileCheck --check-prefix=CHECK --check-prefix=CHECK-MIPS64 %s

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK:       .p2align 2
; CHECK-MIPS64-LABEL: .Lxray_sled_0:
; CHECK-MIPS32-LABEL: $xray_sled_0:
; CHECK-NEXT:  b [[TMP:(\.L|\$)tmp[0-9]+]]
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-MIPS64:  nop
; CHECK-MIPS64:  nop
; CHECK-MIPS64:  nop
; CHECK-MIPS64:  nop
; CHECK-NEXT:  [[TMP]]:
; CHECK-MIPS32-NEXT:  addiu $25, $25, 52
  ret i32 0
; CHECK:       .p2align 2
; CHECK-MIPS64-LABEL: .Lxray_sled_1:
; CHECK-MIPS32-LABEL: $xray_sled_1:
; CHECK-NEXT:  b [[TMP:(\.L|\$)tmp[0-9]+]]
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-MIPS64:  nop
; CHECK-MIPS64:  nop
; CHECK-MIPS64:  nop
; CHECK-MIPS64:  nop
; CHECK:       [[TMP]]:
; CHECK-MIPS32:  addiu $25, $25, 52
}
; CHECK:             .section xray_instr_map,"ao",@progbits,foo
; CHECK-MIPS64:      [[TMP:.Ltmp[0-9]+]]:
; CHECK-MIPS64-NEXT:   .8byte  .Lxray_sled_0-[[TMP]]
; CHECK-MIPS64-NEXT:   .8byte  .Lfunc_begin0-([[TMP]]+8)
; CHECK-MIPS32:      [[TMP:\$tmp[0-9]+]]:
; CHECK-MIPS32-NEXT:   .4byte  ($xray_sled_0)-([[TMP]])
; CHECK-MIPS32-NEXT:   .4byte  ($func_begin0)-(([[TMP]])+4)

; We test multiple returns in a single function to make sure we're getting all
; of them with XRay instrumentation.
define i32 @bar(i32 %i) nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK:       .p2align 2
; CHECK-MIPS64-LABEL: .Lxray_sled_2:
; CHECK-MIPS32-LABEL: $xray_sled_2:
; CHECK-NEXT:  b [[TMP:(\.L|\$)tmp[0-9]+]]
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-MIPS64:  nop
; CHECK-MIPS64:  nop
; CHECK-MIPS64:  nop
; CHECK-MIPS64:  nop
; CHECK:       [[TMP]]:
; CHECK-MIPS32:  addiu $25, $25, 52
Test:
  %cond = icmp eq i32 %i, 0
  br i1 %cond, label %IsEqual, label %NotEqual
IsEqual:
  ret i32 0
; CHECK:       .p2align 2
; CHECK-MIPS64-LABEL: .Lxray_sled_3:
; CHECK-MIPS32-LABEL: $xray_sled_3:
; CHECK-NEXT:  b [[TMP:(\.L|\$)tmp[0-9]+]]
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-MIPS64:  nop
; CHECK-MIPS64:  nop
; CHECK-MIPS64:  nop
; CHECK-MIPS64:  nop
; CHECK-NEXT:    [[TMP]]:
; CHECK-MIPS32:  addiu $25, $25, 52 
NotEqual:
  ret i32 1
; CHECK:       .p2align 2
; CHECK-MIPS64-LABEL: .Lxray_sled_4:
; CHECK-MIPS32-LABEL: $xray_sled_4:
; CHECK-NEXT:  b [[TMP:(\.L|\$)tmp[0-9]+]]
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-MIPS64:  nop
; CHECK-MIPS64:  nop
; CHECK-MIPS64:  nop
; CHECK-MIPS64:  nop
; CHECK-NEXT:    [[TMP]]:
; CHECK-MIPS32:  addiu $25, $25, 52
}
; CHECK: .section xray_instr_map,"ao",@progbits,bar
; CHECK-MIPS64: .8byte  .Lxray_sled_2
; CHECK-MIPS64: .8byte  .Lxray_sled_3
; CHECK-MIPS64: .8byte  .Lxray_sled_4
; CHECK-MIPS32:      [[TMP:\$tmp[0-9]+]]:
; CHECK-MIPS32-NEXT: .4byte	($xray_sled_2)-([[TMP]])
; CHECK-MIPS32:      [[TMP:\$tmp[0-9]+]]:
; CHECK-MIPS32-NEXT: .4byte	($xray_sled_3)-([[TMP]])
; CHECK-MIPS32:      [[TMP:\$tmp[0-9]+]]:
; CHECK-MIPS32-NEXT: .4byte	($xray_sled_4)-([[TMP]])
