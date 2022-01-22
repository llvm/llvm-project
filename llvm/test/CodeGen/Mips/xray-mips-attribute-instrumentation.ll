; RUN: llc -mtriple=mips-unknown-linux-gnu < %s | FileCheck --check-prefix=CHECK --check-prefix=CHECK-MIPS32 %s
; RUN: llc -mtriple=mipsel-unknown-linux-gnu < %s | FileCheck --check-prefix=CHECK --check-prefix=CHECK-MIPS32 %s
; RUN: llc -mtriple=mips64-unknown-linux-gnu < %s | FileCheck --check-prefix=CHECK --check-prefix=CHECK-MIPS64 %s
; RUN: llc -mtriple=mips64el-unknown-linux-gnu < %s | FileCheck --check-prefix=CHECK --check-prefix=CHECK-MIPS64 %s

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK:       .p2align 2
; CHECK-MIPS64-LABEL: .Lxray_sled_0:
; CHECK-MIPS32-LABEL: $xray_sled_0:
; CHECK-MIPS64:  b .Ltmp1
; CHECK-MIPS32:  b $tmp1
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
; CHECK-MIPS64-LABEL: .Ltmp1:
; CHECK-MIPS32-LABEL: $tmp1:
; CHECK-MIPS32:  addiu $25, $25, 52
  ret i32 0
; CHECK:       .p2align 2
; CHECK-MIPS64-LABEL: .Lxray_sled_1:
; CHECK-MIPS64-NEXT:   b .Ltmp2
; CHECK-MIPS32-LABEL: $xray_sled_1:
; CHECK-MIPS32-NEXT:   b $tmp2
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
; CHECK-MIPS64-LABEL: .Ltmp2:
; CHECK-MIPS32-LABEL: $tmp2:
; CHECK-MIPS32:  addiu $25, $25, 52
}
; CHECK:             .section xray_instr_map,"ao",@progbits,foo
; CHECK-MIPS64:      .Ltmp3:
; CHECK-MIPS64-NEXT:   .8byte  .Lxray_sled_0-.Ltmp3
; CHECK-MIPS64-NEXT:   .8byte  .Lfunc_begin0-(.Ltmp3+8)
; CHECK-MIPS32:      $tmp3:
; CHECK-MIPS32-NEXT:   .4byte  ($xray_sled_0)-($tmp3)
; CHECK-MIPS32-NEXT:   .4byte  ($func_begin0)-(($tmp3)+4)

; We test multiple returns in a single function to make sure we're getting all
; of them with XRay instrumentation.
define i32 @bar(i32 %i) nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK:       .p2align 2
; CHECK-MIPS64-LABEL: .Lxray_sled_2:
; CHECK-MIPS64-NEXT:   b .Ltmp6
; CHECK-MIPS32-LABEL: $xray_sled_2:
; CHECK-MIPS32-NEXT:   b $tmp6
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
; CHECK-MIPS64-LABEL: .Ltmp6:
; CHECK-MIPS32-LABEL: $tmp6:
; CHECK-MIPS32:  addiu $25, $25, 52
Test:
  %cond = icmp eq i32 %i, 0
  br i1 %cond, label %IsEqual, label %NotEqual
IsEqual:
  ret i32 0
; CHECK:       .p2align 2
; CHECK-MIPS64-LABEL: .Lxray_sled_3:
; CHECK-MIPS64-NEXT:   b .Ltmp7
; CHECK-MIPS32-LABEL: $xray_sled_3:
; CHECK-MIPS32-NEXT:   b $tmp7
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
; CHECK-MIPS64-LABEL: .Ltmp7:
; CHECK-MIPS32-LABEL: $tmp7:
; CHECK-MIPS32:  addiu $25, $25, 52 
NotEqual:
  ret i32 1
; CHECK:       .p2align 2
; CHECK-MIPS64-LABEL: .Lxray_sled_4:
; CHECK-MIPS64-NEXT:   b .Ltmp8
; CHECK-MIPS32-LABEL: $xray_sled_4:
; CHECK-MIPS32-NEXT:   b $tmp8
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
; CHECK-MIPS64-LABEL: .Ltmp8:
; CHECK-MIPS32-LABEL: $tmp8:
; CHECK-MIPS32:  addiu $25, $25, 52
}
; CHECK: .section xray_instr_map,{{.*}}
; CHECK-MIPS64: .8byte  .Lxray_sled_2
; CHECK-MIPS64: .8byte  .Lxray_sled_3
; CHECK-MIPS64: .8byte  .Lxray_sled_4
; CHECK-MIPS32: .4byte	($xray_sled_2)-($tmp9)
; CHECK-MIPS32: .4byte	($xray_sled_3)-($tmp10)
; CHECK-MIPS32: .4byte	($xray_sled_4)-($tmp11)
