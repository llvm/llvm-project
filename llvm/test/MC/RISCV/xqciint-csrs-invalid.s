# Xqciint - Qualcomm uC Custom CSRs
# RUN: llvm-mc --verify -triple riscv32 -mattr=-xqciint < %s

csrrs t2, qc.mmcr, zero // expected-error:11 {{system register 'qc.mmcr' requires 'xqciint' to be enabled}}

csrrs t2, qc.mntvec, zero // expected-error:11 {{system register 'qc.mntvec' requires 'xqciint' to be enabled}}

csrrs t2, qc.mstktopaddr, zero // expected-error:11 {{system register 'qc.mstktopaddr' requires 'xqciint' to be enabled}}

csrrs t2, qc.mstkbottomaddr, zero // expected-error:11 {{system register 'qc.mstkbottomaddr' requires 'xqciint' to be enabled}}

csrrs t2, qc.mthreadptr, zero // expected-error:11 {{system register 'qc.mthreadptr' requires 'xqciint' to be enabled}}

csrrs t2, qc.mcause, zero // expected-error:11 {{system register 'qc.mcause' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicip0, zero // expected-error:11 {{system register 'qc.mclicip0' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicip1, zero // expected-error:11 {{system register 'qc.mclicip1' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicip2, zero // expected-error:11 {{system register 'qc.mclicip2' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicip3, zero // expected-error:11 {{system register 'qc.mclicip3' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicip4, zero // expected-error:11 {{system register 'qc.mclicip4' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicip5, zero // expected-error:11 {{system register 'qc.mclicip5' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicip6, zero // expected-error:11 {{system register 'qc.mclicip6' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicip7, zero // expected-error:11 {{system register 'qc.mclicip7' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicie0, zero // expected-error:11 {{system register 'qc.mclicie0' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicie1, zero // expected-error:11 {{system register 'qc.mclicie1' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicie2, zero // expected-error:11 {{system register 'qc.mclicie2' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicie3, zero // expected-error:11 {{system register 'qc.mclicie3' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicie4, zero // expected-error:11 {{system register 'qc.mclicie4' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicie5, zero // expected-error:11 {{system register 'qc.mclicie5' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicie6, zero // expected-error:11 {{system register 'qc.mclicie6' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicie7, zero // expected-error:11 {{system register 'qc.mclicie7' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl00, zero // expected-error:11 {{system register 'qc.mclicilvl00' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl01, zero // expected-error:11 {{system register 'qc.mclicilvl01' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl02, zero // expected-error:11 {{system register 'qc.mclicilvl02' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl03, zero // expected-error:11 {{system register 'qc.mclicilvl03' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl04, zero // expected-error:11 {{system register 'qc.mclicilvl04' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl05, zero // expected-error:11 {{system register 'qc.mclicilvl05' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl06, zero // expected-error:11 {{system register 'qc.mclicilvl06' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl07, zero // expected-error:11 {{system register 'qc.mclicilvl07' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl08, zero // expected-error:11 {{system register 'qc.mclicilvl08' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl09, zero // expected-error:11 {{system register 'qc.mclicilvl09' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl10, zero // expected-error:11 {{system register 'qc.mclicilvl10' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl11, zero // expected-error:11 {{system register 'qc.mclicilvl11' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl12, zero // expected-error:11 {{system register 'qc.mclicilvl12' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl13, zero // expected-error:11 {{system register 'qc.mclicilvl13' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl14, zero // expected-error:11 {{system register 'qc.mclicilvl14' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl15, zero // expected-error:11 {{system register 'qc.mclicilvl15' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl16, zero // expected-error:11 {{system register 'qc.mclicilvl16' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl17, zero // expected-error:11 {{system register 'qc.mclicilvl17' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl18, zero // expected-error:11 {{system register 'qc.mclicilvl18' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl19, zero // expected-error:11 {{system register 'qc.mclicilvl19' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl20, zero // expected-error:11 {{system register 'qc.mclicilvl20' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl21, zero // expected-error:11 {{system register 'qc.mclicilvl21' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl22, zero // expected-error:11 {{system register 'qc.mclicilvl22' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl23, zero // expected-error:11 {{system register 'qc.mclicilvl23' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl24, zero // expected-error:11 {{system register 'qc.mclicilvl24' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl25, zero // expected-error:11 {{system register 'qc.mclicilvl25' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl26, zero // expected-error:11 {{system register 'qc.mclicilvl26' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl27, zero // expected-error:11 {{system register 'qc.mclicilvl27' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl28, zero // expected-error:11 {{system register 'qc.mclicilvl28' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl29, zero // expected-error:11 {{system register 'qc.mclicilvl29' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl30, zero // expected-error:11 {{system register 'qc.mclicilvl30' requires 'xqciint' to be enabled}}

csrrs t2, qc.mclicilvl31, zero // expected-error:11 {{system register 'qc.mclicilvl31' requires 'xqciint' to be enabled}}

csrrs t2, qc.mwpstartaddr0, zero // expected-error:11 {{system register 'qc.mwpstartaddr0' requires 'xqciint' to be enabled}}

csrrs t2, qc.mwpstartaddr1, zero // expected-error:11 {{system register 'qc.mwpstartaddr1' requires 'xqciint' to be enabled}}

csrrs t2, qc.mwpstartaddr2, zero // expected-error:11 {{system register 'qc.mwpstartaddr2' requires 'xqciint' to be enabled}}

csrrs t2, qc.mwpstartaddr3, zero // expected-error:11 {{system register 'qc.mwpstartaddr3' requires 'xqciint' to be enabled}}

csrrs t2, qc.mwpendaddr0, zero // expected-error:11 {{system register 'qc.mwpendaddr0' requires 'xqciint' to be enabled}}

csrrs t2, qc.mwpendaddr1, zero // expected-error:11 {{system register 'qc.mwpendaddr1' requires 'xqciint' to be enabled}}

csrrs t2, qc.mwpendaddr2, zero // expected-error:11 {{system register 'qc.mwpendaddr2' requires 'xqciint' to be enabled}}

csrrs t2, qc.mwpendaddr3, zero // expected-error:11 {{system register 'qc.mwpendaddr3' requires 'xqciint' to be enabled}}

