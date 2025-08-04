# Xqciint - Qualcomm uC Custom CSRs
# RUN: not llvm-mc -triple riscv32 -mattr=-experimental-xqciint < %s 2>&1 \
# RUN:         | FileCheck -check-prefixes=CHECK-FEATURE %s

csrrs t2, qc.mmcr, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mmcr' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mntvec, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mntvec' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mstktopaddr, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mstktopaddr' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mstkbottomaddr, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mstkbottomaddr' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mthreadptr, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mthreadptr' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mcause, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mcause' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicip0, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicip0' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicip1, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicip1' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicip2, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicip2' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicip3, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicip3' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicip4, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicip4' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicip5, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicip5' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicip6, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicip6' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicip7, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicip7' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicie0, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicie0' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicie1, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicie1' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicie2, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicie2' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicie3, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicie3' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicie4, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicie4' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicie5, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicie5' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicie6, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicie6' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicie7, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicie7' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl00, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl00' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl01, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl01' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl02, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl02' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl03, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl03' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl04, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl04' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl05, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl05' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl06, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl06' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl07, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl07' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl08, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl08' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl09, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl09' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl10, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl10' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl11, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl11' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl12, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl12' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl13, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl13' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl14, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl14' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl15, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl15' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl16, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl16' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl17, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl17' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl18, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl18' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl19, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl19' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl20, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl20' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl21, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl21' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl22, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl22' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl23, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl23' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl24, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl24' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl25, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl25' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl26, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl26' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl27, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl27' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl28, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl28' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl29, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl29' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl30, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl30' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mclicilvl31, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mclicilvl31' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mwpstartaddr0, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mwpstartaddr0' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mwpstartaddr1, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mwpstartaddr1' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mwpstartaddr2, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mwpstartaddr2' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mwpstartaddr3, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mwpstartaddr3' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mwpendaddr0, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mwpendaddr0' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mwpendaddr1, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mwpendaddr1' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mwpendaddr2, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mwpendaddr2' requires 'experimental-xqciint' to be enabled

csrrs t2, qc.mwpendaddr3, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'qc.mwpendaddr3' requires 'experimental-xqciint' to be enabled

