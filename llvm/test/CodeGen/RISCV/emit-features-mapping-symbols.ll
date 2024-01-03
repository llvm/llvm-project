; RUN: llc -filetype=obj -mtriple=riscv32 < %s -o %t
; RUN: llvm-readelf -s %t | FileCheck %s --check-prefixes=SEC

; SEC: [[#%x,]]: [[#%x,]]     0 NOTYPE  LOCAL  DEFAULT     2 $xrv32i2p1_f2p2_d2p2_c2p0_zicsr2p0.0
; SEC: [[#%x,]]: [[#%x,]]     0 NOTYPE  LOCAL  DEFAULT     2 $xrv32i2p1_f2p2_d2p2_zicsr2p0.1
; SEC: [[#%x,]]: [[#%x,]]     0 NOTYPE  LOCAL  DEFAULT     2 $xrv32i2p1_a2p1_f2p2_d2p2_c2p0_zicsr2p0.2
; SEC: [[#%x,]]: [[#%x,]]     0 NOTYPE  LOCAL  DEFAULT     2 $xrv32i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0.3
; SEC: [[#%x,]]: [[#%x,]]     0 NOTYPE  LOCAL  DEFAULT     2 $xrv32i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0.4
; SEC: [[#%x,]]: [[#%x,]]     0 NOTYPE  LOCAL  DEFAULT     2 $x.5

target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-unknown"

; Function Attrs: nounwind
define dso_local void @testAttr0() #0 {
entry:
  ret void
}

; Function Attrs: nounwind
define dso_local void @testAttr1() #1 {
entry:
  ret void
}

; Function Attrs: nounwind
define dso_local void @testAttr2() #2 {
entry:
  ret void
}

; Function Attrs: nounwind
define dso_local void @testAttr3() #3 {
entry:
  ret void
}

; Function Attrs: nounwind
define dso_local void @testAttr4() #4 {
entry:
  ret void
}

; Function Attrs: nounwind
define dso_local void @testAttrDefault() {
entry:
  ret void
}

attributes #0 = { nounwind "target-cpu"="generic-rv32" "target-features"="+32bit,+c,+d" }
attributes #1 = { nounwind "target-cpu"="generic-rv32" "target-features"="+32bit,+d" }
attributes #2 = { nounwind "target-cpu"="generic-rv32" "target-features"="+32bit,+a,+c,+d" }
attributes #3 = { nounwind "target-cpu"="generic-rv32" "target-features"="+32bit,+a,+c,+d,+f,+m,+zicsr" }
attributes #4 = { nounwind "target-cpu"="generic-rv32" "target-features"="+32bit,+a,+c,+d,+f,+m,+v" }



