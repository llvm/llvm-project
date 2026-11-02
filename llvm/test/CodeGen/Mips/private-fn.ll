; RUN: llc -mtriple=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-O32
; RUN: llc -mtriple=mipsel -relocation-model=pic -O0 -fast-isel=false < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-O32
; RUN: llc -mtriple=mips64el -relocation-model=pic -O0 -fast-isel-abort=3 < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-N64
; RUN: llc -mtriple=mips64el -relocation-model=pic -O0 -fast-isel=false < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-N64
; RUN: llc -mtriple=mips64el -target-abi n32 -relocation-model=pic -O0 -fast-isel-abort=3 < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-N32
; RUN: llc -mtriple=mips64el -target-abi n32 -relocation-model=pic -O0 -fast-isel=false < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-N32

define private void @priv() {
  ret void
}

define ptr @addr() {
; CHECK-LABEL: addr:
; CHECK-O32:       lw $[[R:[0-9]+]], %got($priv)(${{[0-9]+}})
; CHECK-O32-NEXT:  addiu ${{[0-9]+}}, $[[R]], %lo($priv)
; CHECK-N64:       ld $[[R:[0-9]+]], %got_page(.Lpriv)(${{[0-9]+}})
; CHECK-N64-NEXT:  daddiu ${{[0-9]+}}, $[[R]], %got_ofst(.Lpriv)
; CHECK-N32:       lw $[[R:[0-9]+]], %got_page(.Lpriv)(${{[0-9]+}})
; CHECK-N32-NEXT:  addiu ${{[0-9]+}}, $[[R]], %got_ofst(.Lpriv)
  ret ptr @priv
}

define void @call() {
; CHECK-LABEL: call:
; CHECK-O32:       lw $[[R:[0-9]+]], %got($priv)(${{[0-9]+}})
; CHECK-O32-NEXT:  addiu $25, $[[R]], %lo($priv)
; CHECK-O32-NEXT:  .reloc {{.*}}, R_MIPS_JALR, $priv
; CHECK-N64:       ld $[[R:[0-9]+]], %got_page(.Lpriv)(${{[0-9]+}})
; CHECK-N64-NEXT:  daddiu ${{[0-9]+}}, $[[R]], %got_ofst(.Lpriv)
; CHECK-N64-NEXT:  .reloc {{.*}}, R_MIPS_JALR, .Lpriv
; CHECK-N32:       lw $[[R:[0-9]+]], %got_page(.Lpriv)(${{[0-9]+}})
; CHECK-N32-NEXT:  addiu ${{[0-9]+}}, $[[R]], %got_ofst(.Lpriv)
; CHECK-N32-NEXT:  .reloc {{.*}}, R_MIPS_JALR, .Lpriv
  call void @priv()
  ret void
}
