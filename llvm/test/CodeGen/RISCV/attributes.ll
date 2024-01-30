;; Generate ELF attributes from llc.

; RUN: llc -mtriple=riscv32 %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+m %s -o - | FileCheck --check-prefixes=CHECK,RV32M %s
; RUN: llc -mtriple=riscv32 -mattr=+zmmul %s -o - | FileCheck --check-prefixes=CHECK,RV32ZMMUL %s
; RUN: llc -mtriple=riscv32 -mattr=+m,+zmmul %s -o - | FileCheck --check-prefixes=CHECK,RV32MZMMUL %s
; RUN: llc -mtriple=riscv32 -mattr=+a %s -o - | FileCheck --check-prefixes=CHECK,RV32A %s
; RUN: llc -mtriple=riscv32 -mattr=+f %s -o - | FileCheck --check-prefixes=CHECK,RV32F %s
; RUN: llc -mtriple=riscv32 -mattr=+d %s -o - | FileCheck --check-prefixes=CHECK,RV32D %s
; RUN: llc -mtriple=riscv32 -mattr=+c %s -o - | FileCheck --check-prefixes=CHECK,RV32C %s
; RUN: llc -mtriple=riscv32 -mattr=+zihintpause %s -o - | FileCheck --check-prefixes=CHECK,RV32ZIHINTPAUSE %s
; RUN: llc -mtriple=riscv32 -mattr=+zihintntl %s -o - | FileCheck --check-prefixes=CHECK,RV32ZIHINTNTL %s
; RUN: llc -mtriple=riscv32 -mattr=+zfhmin %s -o - | FileCheck --check-prefixes=CHECK,RV32ZFHMIN %s
; RUN: llc -mtriple=riscv32 -mattr=+zfh %s -o - | FileCheck --check-prefixes=CHECK,RV32ZFH %s
; RUN: llc -mtriple=riscv32 -mattr=+zba %s -o - | FileCheck --check-prefixes=CHECK,RV32ZBA %s
; RUN: llc -mtriple=riscv32 -mattr=+zbb %s -o - | FileCheck --check-prefixes=CHECK,RV32ZBB %s
; RUN: llc -mtriple=riscv32 -mattr=+zbc %s -o - | FileCheck --check-prefixes=CHECK,RV32ZBC %s
; RUN: llc -mtriple=riscv32 -mattr=+zbs %s -o - | FileCheck --check-prefixes=CHECK,RV32ZBS %s
; RUN: llc -mtriple=riscv32 -mattr=+v %s -o - | FileCheck --check-prefixes=CHECK,RV32V %s
; RUN: llc -mtriple=riscv32 -mattr=+h %s -o - | FileCheck --check-prefixes=CHECK,RV32H %s
; RUN: llc -mtriple=riscv32 -mattr=+zbb,+zfh,+v,+f %s -o - | FileCheck --check-prefixes=CHECK,RV32COMBINED %s
; RUN: llc -mtriple=riscv32 -mattr=+zbkb %s -o - | FileCheck --check-prefixes=CHECK,RV32ZBKB %s
; RUN: llc -mtriple=riscv32 -mattr=+zbkc %s -o - | FileCheck --check-prefixes=CHECK,RV32ZBKC %s
; RUN: llc -mtriple=riscv32 -mattr=+zbkx %s -o - | FileCheck --check-prefixes=CHECK,RV32ZBKX %s
; RUN: llc -mtriple=riscv32 -mattr=+zknd %s -o - | FileCheck --check-prefixes=CHECK,RV32ZKND %s
; RUN: llc -mtriple=riscv32 -mattr=+zkne %s -o - | FileCheck --check-prefixes=CHECK,RV32ZKNE %s
; RUN: llc -mtriple=riscv32 -mattr=+zknh %s -o - | FileCheck --check-prefixes=CHECK,RV32ZKNH %s
; RUN: llc -mtriple=riscv32 -mattr=+zksed %s -o - | FileCheck --check-prefixes=CHECK,RV32ZKSED %s
; RUN: llc -mtriple=riscv32 -mattr=+zksh %s -o - | FileCheck --check-prefixes=CHECK,RV32ZKSH %s
; RUN: llc -mtriple=riscv32 -mattr=+zkr %s -o - | FileCheck --check-prefixes=CHECK,RV32ZKR %s
; RUN: llc -mtriple=riscv32 -mattr=+zkn %s -o - | FileCheck --check-prefixes=CHECK,RV32ZKN %s
; RUN: llc -mtriple=riscv32 -mattr=+zks %s -o - | FileCheck --check-prefixes=CHECK,RV32ZKS %s
; RUN: llc -mtriple=riscv32 -mattr=+zkt %s -o - | FileCheck --check-prefixes=CHECK,RV32ZKT %s
; RUN: llc -mtriple=riscv32 -mattr=+zk %s -o - | FileCheck --check-prefixes=CHECK,RV32ZK %s
; RUN: llc -mtriple=riscv32 -mattr=+zkn,+zkr,+zkt %s -o - | FileCheck --check-prefixes=CHECK,RV32COMBINEINTOZK %s
; RUN: llc -mtriple=riscv32 -mattr=+zbkb,+zbkc,+zbkx,+zkne,+zknd,+zknh %s -o - | FileCheck --check-prefixes=CHECK,RV32COMBINEINTOZKN %s
; RUN: llc -mtriple=riscv32 -mattr=+zbkb,+zbkc,+zbkx,+zksed,+zksh %s -o - | FileCheck --check-prefixes=CHECK,RV32COMBINEINTOZKS %s
; RUN: llc -mtriple=riscv32 -mattr=+zicbom %s -o - | FileCheck --check-prefixes=CHECK,RV32ZICBOM %s
; RUN: llc -mtriple=riscv32 -mattr=+zicboz %s -o - | FileCheck --check-prefixes=CHECK,RV32ZICBOZ %s
; RUN: llc -mtriple=riscv32 -mattr=+zicbop %s -o - | FileCheck --check-prefixes=CHECK,RV32ZICBOP %s
; RUN: llc -mtriple=riscv32 -mattr=+svnapot %s -o - | FileCheck --check-prefixes=CHECK,RV32SVNAPOT %s
; RUN: llc -mtriple=riscv32 -mattr=+svpbmt %s -o - | FileCheck --check-prefixes=CHECK,RV32SVPBMT %s
; RUN: llc -mtriple=riscv32 -mattr=+svinval %s -o - | FileCheck --check-prefixes=CHECK,RV32SVINVAL %s
; RUN: llc -mtriple=riscv32 -mattr=+xcvalu %s -o - | FileCheck --check-prefix=RV32XCVALU %s
; RUN: llc -mtriple=riscv32 -mattr=+xcvbitmanip %s -o - | FileCheck --check-prefix=RV32XCVBITMANIP %s
; RUN: llc -mtriple=riscv32 -mattr=+xcvelw %s -o - | FileCheck --check-prefix=RV32XCVELW %s
; RUN: llc -mtriple=riscv32 -mattr=+xcvmac %s -o - | FileCheck --check-prefix=RV32XCVMAC %s
; RUN: llc -mtriple=riscv32 -mattr=+xcvmem %s -o - | FileCheck --check-prefix=RV32XCVMEM %s
; RUN: llc -mtriple=riscv32 -mattr=+xcvsimd %s -o - | FileCheck --check-prefix=RV32XCVSIMD %s
; RUN: llc -mtriple=riscv32 -mattr=+xcvbi %s -o - | FileCheck --check-prefix=RV32XCVBI %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfvfwmaccqqq %s -o - | FileCheck --check-prefix=RV32XSFVFWMACCQQQ %s
; RUN: llc -mtriple=riscv32 -mattr=+xtheadcmo %s -o - | FileCheck --check-prefix=RV32XTHEADCMO %s
; RUN: llc -mtriple=riscv32 -mattr=+xtheadcondmov %s -o - | FileCheck --check-prefix=RV32XTHEADCONDMOV %s
; RUN: llc -mtriple=riscv32 -mattr=+xtheadfmemidx %s -o - | FileCheck --check-prefix=RV32XTHEADFMEMIDX %s
; RUN: llc -mtriple=riscv32 -mattr=+xtheadmac %s -o - | FileCheck --check-prefixes=CHECK,RV32XTHEADMAC %s
; RUN: llc -mtriple=riscv32 -mattr=+xtheadmemidx %s -o - | FileCheck --check-prefix=RV32XTHEADMEMIDX %s
; RUN: llc -mtriple=riscv32 -mattr=+xtheadmempair %s -o - | FileCheck --check-prefix=RV32XTHEADMEMPAIR %s
; RUN: llc -mtriple=riscv32 -mattr=+xtheadsync %s -o - | FileCheck --check-prefix=RV32XTHEADSYNC %s
; RUN: llc -mtriple=riscv32 -mattr=+zca %s -o - | FileCheck --check-prefixes=CHECK,RV32ZCA %s
; RUN: llc -mtriple=riscv32 -mattr=+zcb %s -o - | FileCheck --check-prefixes=CHECK,RV32ZCB %s
; RUN: llc -mtriple=riscv32 -mattr=+zcd %s -o - | FileCheck --check-prefixes=CHECK,RV32ZCD %s
; RUN: llc -mtriple=riscv32 -mattr=+zcf %s -o - | FileCheck --check-prefixes=CHECK,RV32ZCF %s
; RUN: llc -mtriple=riscv32 -mattr=+zcmp %s -o - | FileCheck --check-prefixes=CHECK,RV32ZCMP %s
; RUN: llc -mtriple=riscv32 -mattr=+zcmt %s -o - | FileCheck --check-prefixes=CHECK,RV32ZCMT %s
; RUN: llc -mtriple=riscv32 -mattr=+zicsr %s -o - | FileCheck --check-prefixes=CHECK,RV32ZICSR %s
; RUN: llc -mtriple=riscv32 -mattr=+zifencei %s -o - | FileCheck --check-prefixes=CHECK,RV32ZIFENCEI %s
; RUN: llc -mtriple=riscv32 -mattr=+zicntr %s -o - | FileCheck --check-prefixes=CHECK,RV32ZICNTR %s
; RUN: llc -mtriple=riscv32 -mattr=+zihpm %s -o - | FileCheck --check-prefixes=CHECK,RV32ZIHPM %s
; RUN: llc -mtriple=riscv32 -mattr=+zfa %s -o - | FileCheck --check-prefixes=CHECK,RV32ZFA %s
; RUN: llc -mtriple=riscv32 -mattr=+zve32x -mattr=+zvbb %s -o - | FileCheck --check-prefix=RV32ZVBB %s
; RUN: llc -mtriple=riscv32 -mattr=+zve64x -mattr=+zvbc %s -o - | FileCheck --check-prefix=RV32ZVBC %s
; RUN: llc -mtriple=riscv32 -mattr=+zve32x -mattr=+zvkb %s -o - | FileCheck --check-prefix=RV32ZVKB %s
; RUN: llc -mtriple=riscv32 -mattr=+zve32x -mattr=+zvkg %s -o - | FileCheck --check-prefix=RV32ZVKG %s
; RUN: llc -mtriple=riscv32 -mattr=+zve64x -mattr=+zvkn %s -o - | FileCheck --check-prefix=RV32ZVKN %s
; RUN: llc -mtriple=riscv32 -mattr=+zve64x -mattr=+zvknc %s -o - | FileCheck --check-prefix=RV32ZVKNC %s
; RUN: llc -mtriple=riscv32 -mattr=+zve32x -mattr=+zvkned %s -o - | FileCheck --check-prefix=RV32ZVKNED %s
; RUN: llc -mtriple=riscv32 -mattr=+zve64x -mattr=+zvkng %s -o - | FileCheck --check-prefix=RV32ZVKNG %s
; RUN: llc -mtriple=riscv32 -mattr=+zve32x -mattr=+zvknha %s -o - | FileCheck --check-prefix=RV32ZVKNHA %s
; RUN: llc -mtriple=riscv32 -mattr=+zve64x -mattr=+zvknhb %s -o - | FileCheck --check-prefix=RV32ZVKNHB %s
; RUN: llc -mtriple=riscv32 -mattr=+zve64x -mattr=+zvks %s -o - | FileCheck --check-prefix=RV32ZVKS %s
; RUN: llc -mtriple=riscv32 -mattr=+zve64x -mattr=+zvksc %s -o - | FileCheck --check-prefix=RV32ZVKSC %s
; RUN: llc -mtriple=riscv32 -mattr=+zve32x -mattr=+zvksed %s -o - | FileCheck --check-prefix=RV32ZVKSED %s
; RUN: llc -mtriple=riscv32 -mattr=+zve64x -mattr=+zvksg %s -o - | FileCheck --check-prefix=RV32ZVKSG %s
; RUN: llc -mtriple=riscv32 -mattr=+zve32x -mattr=+zvksh %s -o - | FileCheck --check-prefix=RV32ZVKSH %s
; RUN: llc -mtriple=riscv32 -mattr=+zve32x -mattr=+zvkt %s -o - | FileCheck --check-prefix=RV32ZVKT %s
; RUN: llc -mtriple=riscv32 -mattr=+zvfh %s -o - | FileCheck --check-prefix=RV32ZVFH %s
; RUN: llc -mtriple=riscv32 -mattr=+zicond %s -o - | FileCheck --check-prefix=RV32ZICOND %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-zimop %s -o - | FileCheck --check-prefix=RV32ZIMOP %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-zcmop %s -o - | FileCheck --check-prefix=RV32ZCMOP %s
; RUN: llc -mtriple=riscv32 -mattr=+smaia %s -o - | FileCheck --check-prefixes=CHECK,RV32SMAIA %s
; RUN: llc -mtriple=riscv32 -mattr=+ssaia %s -o - | FileCheck --check-prefixes=CHECK,RV32SSAIA %s
; RUN: llc -mtriple=riscv32 -mattr=+smepmp %s -o - | FileCheck --check-prefixes=CHECK,RV32SMEPMP %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-zfbfmin %s -o - | FileCheck --check-prefixes=CHECK,RV32ZFBFMIN %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-zvfbfmin %s -o - | FileCheck --check-prefixes=CHECK,RV32ZVFBFMIN %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-zvfbfwma %s -o - | FileCheck --check-prefixes=CHECK,RV32ZVFBFWMA %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-zacas %s -o - | FileCheck --check-prefix=RV32ZACAS %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-zicfilp %s -o - | FileCheck --check-prefix=RV32ZICFILP %s

; RUN: llc -mtriple=riscv64 %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+m %s -o - | FileCheck --check-prefixes=CHECK,RV64M %s
; RUN: llc -mtriple=riscv64 -mattr=+zmmul %s -o - | FileCheck --check-prefixes=CHECK,RV64ZMMUL %s
; RUN: llc -mtriple=riscv64 -mattr=+m,+zmmul %s -o - | FileCheck --check-prefixes=CHECK,RV64MZMMUL %s
; RUN: llc -mtriple=riscv64 -mattr=+a %s -o - | FileCheck --check-prefixes=CHECK,RV64A %s
; RUN: llc -mtriple=riscv64 -mattr=+f %s -o - | FileCheck --check-prefixes=CHECK,RV64F %s
; RUN: llc -mtriple=riscv64 -mattr=+d %s -o - | FileCheck --check-prefixes=CHECK,RV64D %s
; RUN: llc -mtriple=riscv64 -mattr=+c %s -o - | FileCheck --check-prefixes=CHECK,RV64C %s
; RUN: llc -mtriple=riscv64 -mattr=+zihintpause %s -o - | FileCheck --check-prefixes=CHECK,RV64ZIHINTPAUSE %s
; RUN: llc -mtriple=riscv64 -mattr=+zihintntl %s -o - | FileCheck --check-prefixes=CHECK,RV64ZIHINTNTL %s
; RUN: llc -mtriple=riscv64 -mattr=+zfhmin %s -o - | FileCheck --check-prefixes=CHECK,RV64ZFHMIN %s
; RUN: llc -mtriple=riscv64 -mattr=+zfh %s -o - | FileCheck --check-prefixes=CHECK,RV64ZFH %s
; RUN: llc -mtriple=riscv64 -mattr=+zba %s -o - | FileCheck --check-prefixes=CHECK,RV64ZBA %s
; RUN: llc -mtriple=riscv64 -mattr=+zbb %s -o - | FileCheck --check-prefixes=CHECK,RV64ZBB %s
; RUN: llc -mtriple=riscv64 -mattr=+zbc %s -o - | FileCheck --check-prefixes=CHECK,RV64ZBC %s
; RUN: llc -mtriple=riscv64 -mattr=+zbs %s -o - | FileCheck --check-prefixes=CHECK,RV64ZBS %s
; RUN: llc -mtriple=riscv64 -mattr=+v %s -o - | FileCheck --check-prefixes=CHECK,RV64V %s
; RUN: llc -mtriple=riscv64 -mattr=+h %s -o - | FileCheck --check-prefixes=CHECK,RV64H %s
; RUN: llc -mtriple=riscv64 -mattr=+zbb,+zfh,+v,+f %s -o - | FileCheck --check-prefixes=CHECK,RV64COMBINED %s
; RUN: llc -mtriple=riscv64 -mattr=+zbkb %s -o - | FileCheck --check-prefixes=CHECK,RV64ZBKB %s
; RUN: llc -mtriple=riscv64 -mattr=+zbkc %s -o - | FileCheck --check-prefixes=CHECK,RV64ZBKC %s
; RUN: llc -mtriple=riscv64 -mattr=+zbkx %s -o - | FileCheck --check-prefixes=CHECK,RV64ZBKX %s
; RUN: llc -mtriple=riscv64 -mattr=+zknd %s -o - | FileCheck --check-prefixes=CHECK,RV64ZKND %s
; RUN: llc -mtriple=riscv64 -mattr=+zkne %s -o - | FileCheck --check-prefixes=CHECK,RV64ZKNE %s
; RUN: llc -mtriple=riscv64 -mattr=+zknh %s -o - | FileCheck --check-prefixes=CHECK,RV64ZKNH %s
; RUN: llc -mtriple=riscv64 -mattr=+zksed %s -o - | FileCheck --check-prefixes=CHECK,RV64ZKSED %s
; RUN: llc -mtriple=riscv64 -mattr=+zksh %s -o - | FileCheck --check-prefixes=CHECK,RV64ZKSH %s
; RUN: llc -mtriple=riscv64 -mattr=+zkr %s -o - | FileCheck --check-prefixes=CHECK,RV64ZKR %s
; RUN: llc -mtriple=riscv64 -mattr=+zkn %s -o - | FileCheck --check-prefixes=CHECK,RV64ZKN %s
; RUN: llc -mtriple=riscv64 -mattr=+zks %s -o - | FileCheck --check-prefixes=CHECK,RV64ZKS %s
; RUN: llc -mtriple=riscv64 -mattr=+zkt %s -o - | FileCheck --check-prefixes=CHECK,RV64ZKT %s
; RUN: llc -mtriple=riscv64 -mattr=+zk %s -o - | FileCheck --check-prefixes=CHECK,RV64ZK %s
; RUN: llc -mtriple=riscv64 -mattr=+zkn,+zkr,+zkt %s -o - | FileCheck --check-prefixes=CHECK,RV64COMBINEINTOZK %s
; RUN: llc -mtriple=riscv64 -mattr=+zbkb,+zbkc,+zbkx,+zkne,+zknd,+zknh %s -o - | FileCheck --check-prefixes=CHECK,RV64COMBINEINTOZKN %s
; RUN: llc -mtriple=riscv64 -mattr=+zbkb,+zbkc,+zbkx,+zksed,+zksh %s -o - | FileCheck --check-prefixes=CHECK,RV64COMBINEINTOZKS %s
; RUN: llc -mtriple=riscv64 -mattr=+zic64b %s -o - | FileCheck --check-prefixes=CHECK,RV64ZIC64B %s
; RUN: llc -mtriple=riscv64 -mattr=+zicbom %s -o - | FileCheck --check-prefixes=CHECK,RV64ZICBOM %s
; RUN: llc -mtriple=riscv64 -mattr=+zicboz %s -o - | FileCheck --check-prefixes=CHECK,RV64ZICBOZ %s
; RUN: llc -mtriple=riscv64 -mattr=+zicbop %s -o - | FileCheck --check-prefixes=CHECK,RV64ZICBOP %s
; RUN: llc -mtriple=riscv64 -mattr=+svnapot %s -o - | FileCheck --check-prefixes=CHECK,RV64SVNAPOT %s
; RUN: llc -mtriple=riscv64 -mattr=+svpbmt %s -o - | FileCheck --check-prefixes=CHECK,RV64SVPBMT %s
; RUN: llc -mtriple=riscv64 -mattr=+svinval %s -o - | FileCheck --check-prefixes=CHECK,RV64SVINVAL %s
; RUN: llc -mtriple=riscv64 -mattr=+xventanacondops %s -o - | FileCheck --check-prefixes=CHECK,RV64XVENTANACONDOPS %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfvfwmaccqqq %s -o - | FileCheck --check-prefix=RV64XSFVFWMACCQQQ %s
; RUN: llc -mtriple=riscv64 -mattr=+xtheadba %s -o - | FileCheck --check-prefixes=CHECK,RV64XTHEADBA %s
; RUN: llc -mtriple=riscv64 -mattr=+xtheadbb %s -o - | FileCheck --check-prefixes=CHECK,RV64XTHEADBB %s
; RUN: llc -mtriple=riscv64 -mattr=+xtheadbs %s -o - | FileCheck --check-prefixes=CHECK,RV64XTHEADBS %s
; RUN: llc -mtriple=riscv64 -mattr=+xtheadcmo %s -o - | FileCheck --check-prefix=RV64XTHEADCMO %s
; RUN: llc -mtriple=riscv64 -mattr=+xtheadcondmov %s -o - | FileCheck --check-prefix=RV64XTHEADCONDMOV %s
; RUN: llc -mtriple=riscv64 -mattr=+xtheadfmemidx %s -o - | FileCheck --check-prefix=RV64XTHEADFMEMIDX %s
; RUN: llc -mtriple=riscv64 -mattr=+xtheadmac %s -o - | FileCheck --check-prefixes=CHECK,RV64XTHEADMAC %s
; RUN: llc -mtriple=riscv64 -mattr=+xtheadmemidx %s -o - | FileCheck --check-prefix=RV64XTHEADMEMIDX %s
; RUN: llc -mtriple=riscv64 -mattr=+xtheadmempair %s -o - | FileCheck --check-prefix=RV64XTHEADMEMPAIR %s
; RUN: llc -mtriple=riscv64 -mattr=+xtheadsync %s -o - | FileCheck --check-prefix=RV64XTHEADSYNC %s
; RUN: llc -mtriple=riscv64 -mattr=+xtheadvdot %s -o - | FileCheck --check-prefixes=CHECK,RV64XTHEADVDOT %s
; RUN: llc -mtriple=riscv64 -mattr=+za64rs %s -o - | FileCheck --check-prefixes=CHECK,RV64ZA64RS %s
; RUN: llc -mtriple=riscv64 -mattr=+za128rs %s -o - | FileCheck --check-prefixes=CHECK,RV64ZA128RS %s
; RUN: llc -mtriple=riscv64 -mattr=+zawrs %s -o - | FileCheck --check-prefixes=CHECK,RV64ZAWRS %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-ztso %s -o - | FileCheck --check-prefixes=CHECK,RV64ZTSO %s
; RUN: llc -mtriple=riscv64 -mattr=+zca %s -o - | FileCheck --check-prefixes=CHECK,RV64ZCA %s
; RUN: llc -mtriple=riscv64 -mattr=+zcb %s -o - | FileCheck --check-prefixes=CHECK,RV64ZCB %s
; RUN: llc -mtriple=riscv64 -mattr=+zcd %s -o - | FileCheck --check-prefixes=CHECK,RV64ZCD %s
; RUN: llc -mtriple=riscv64 -mattr=+zcmp %s -o - | FileCheck --check-prefixes=CHECK,RV64ZCMP %s
; RUN: llc -mtriple=riscv64 -mattr=+zcmt %s -o - | FileCheck --check-prefixes=CHECK,RV64ZCMT %s
; RUN: llc -mtriple=riscv64 -mattr=+ziccamoa %s -o - | FileCheck --check-prefixes=CHECK,RV64ZICCAMOA %s
; RUN: llc -mtriple=riscv64 -mattr=+ziccif %s -o - | FileCheck --check-prefixes=CHECK,RV64ZICCIF %s
; RUN: llc -mtriple=riscv64 -mattr=+zicclsm %s -o - | FileCheck --check-prefixes=CHECK,RV64ZICCLSM %s
; RUN: llc -mtriple=riscv64 -mattr=+ziccrse %s -o - | FileCheck --check-prefixes=CHECK,RV64ZICCRSE %s
; RUN: llc -mtriple=riscv64 -mattr=+zicsr %s -o - | FileCheck --check-prefixes=CHECK,RV64ZICSR %s
; RUN: llc -mtriple=riscv64 -mattr=+zifencei %s -o - | FileCheck --check-prefixes=CHECK,RV64ZIFENCEI %s
; RUN: llc -mtriple=riscv64 -mattr=+zicntr %s -o - | FileCheck --check-prefixes=CHECK,RV64ZICNTR %s
; RUN: llc -mtriple=riscv64 -mattr=+zihpm %s -o - | FileCheck --check-prefixes=CHECK,RV64ZIHPM %s
; RUN: llc -mtriple=riscv64 -mattr=+zfa %s -o - | FileCheck --check-prefixes=CHECK,RV64ZFA %s
; RUN: llc -mtriple=riscv64 -mattr=+zve32x -mattr=+zvbb %s -o - | FileCheck --check-prefix=RV64ZVBB %s
; RUN: llc -mtriple=riscv64 -mattr=+zve64x -mattr=+zvbc %s -o - | FileCheck --check-prefix=RV64ZVBC %s
; RUN: llc -mtriple=riscv64 -mattr=+zve32x -mattr=+zvkb %s -o - | FileCheck --check-prefix=RV64ZVKB %s
; RUN: llc -mtriple=riscv64 -mattr=+zve32x -mattr=+zvkg %s -o - | FileCheck --check-prefix=RV64ZVKG %s
; RUN: llc -mtriple=riscv64 -mattr=+zve64x -mattr=+zvkn %s -o - | FileCheck --check-prefix=RV64ZVKN %s
; RUN: llc -mtriple=riscv64 -mattr=+zve64x -mattr=+zvknc %s -o - | FileCheck --check-prefix=RV64ZVKNC %s
; RUN: llc -mtriple=riscv64 -mattr=+zve32x -mattr=+zvkned %s -o - | FileCheck --check-prefix=RV64ZVKNED %s
; RUN: llc -mtriple=riscv64 -mattr=+zve64x -mattr=+zvkng %s -o - | FileCheck --check-prefix=RV64ZVKNG %s
; RUN: llc -mtriple=riscv64 -mattr=+zve32x -mattr=+zvknha %s -o - | FileCheck --check-prefix=RV64ZVKNHA %s
; RUN: llc -mtriple=riscv64 -mattr=+zve64x -mattr=+zvknhb %s -o - | FileCheck --check-prefix=RV64ZVKNHB %s
; RUN: llc -mtriple=riscv64 -mattr=+zve32x -mattr=+zvks %s -o - | FileCheck --check-prefix=RV64ZVKS %s
; RUN: llc -mtriple=riscv64 -mattr=+zve64x -mattr=+zvksc %s -o - | FileCheck --check-prefix=RV64ZVKSC %s
; RUN: llc -mtriple=riscv64 -mattr=+zve32x -mattr=+zvksed %s -o - | FileCheck --check-prefix=RV64ZVKSED %s
; RUN: llc -mtriple=riscv64 -mattr=+zve32x -mattr=+zvksg %s -o - | FileCheck --check-prefix=RV64ZVKSG %s
; RUN: llc -mtriple=riscv64 -mattr=+zve32x -mattr=+zvksh %s -o - | FileCheck --check-prefix=RV64ZVKSH %s
; RUN: llc -mtriple=riscv64 -mattr=+zve32x -mattr=+zvkt %s -o - | FileCheck --check-prefix=RV64ZVKT %s
; RUN: llc -mtriple=riscv64 -mattr=+zvfh %s -o - | FileCheck --check-prefix=RV64ZVFH %s
; RUN: llc -mtriple=riscv64 -mattr=+zicond %s -o - | FileCheck --check-prefix=RV64ZICOND %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zimop %s -o - | FileCheck --check-prefix=RV64ZIMOP %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zcmop %s -o - | FileCheck --check-prefix=RV64ZCMOP %s
; RUN: llc -mtriple=riscv64 -mattr=+smaia %s -o - | FileCheck --check-prefixes=CHECK,RV64SMAIA %s
; RUN: llc -mtriple=riscv64 -mattr=+ssaia %s -o - | FileCheck --check-prefixes=CHECK,RV64SSAIA %s
; RUN: llc -mtriple=riscv64 -mattr=+smepmp %s -o - | FileCheck --check-prefixes=CHECK,RV64SMEPMP %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zfbfmin %s -o - | FileCheck --check-prefixes=CHECK,RV64ZFBFMIN %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zvfbfmin %s -o - | FileCheck --check-prefixes=CHECK,RV64ZVFBFMIN %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zvfbfwma %s -o - | FileCheck --check-prefixes=CHECK,RV64ZVFBFWMA %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zacas %s -o - | FileCheck --check-prefix=RV64ZACAS %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zicfilp %s -o - | FileCheck --check-prefix=RV64ZICFILP %s

; CHECK: .attribute 4, 16

; RV32M: .attribute 5, "rv32i2p1_m2p0"
; RV32ZMMUL: .attribute 5, "rv32i2p1_zmmul1p0"
; RV32MZMMUL: .attribute 5, "rv32i2p1_m2p0_zmmul1p0"
; RV32A: .attribute 5, "rv32i2p1_a2p1"
; RV32F: .attribute 5, "rv32i2p1_f2p2_zicsr2p0"
; RV32D: .attribute 5, "rv32i2p1_f2p2_d2p2_zicsr2p0"
; RV32C: .attribute 5, "rv32i2p1_c2p0"
; RV32ZIHINTPAUSE: .attribute 5, "rv32i2p1_zihintpause2p0"
; RV32ZIHINTNTL: .attribute 5, "rv32i2p1_zihintntl1p0"
; RV32ZFHMIN: .attribute 5, "rv32i2p1_f2p2_zicsr2p0_zfhmin1p0"
; RV32ZFH: .attribute 5, "rv32i2p1_f2p2_zicsr2p0_zfh1p0_zfhmin1p0"
; RV32ZBA: .attribute 5, "rv32i2p1_zba1p0"
; RV32ZBB: .attribute 5, "rv32i2p1_zbb1p0"
; RV32ZBC: .attribute 5, "rv32i2p1_zbc1p0"
; RV32ZBS: .attribute 5, "rv32i2p1_zbs1p0"
; RV32V: .attribute 5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
; RV32H: .attribute 5, "rv32i2p1_h1p0"
; RV32COMBINED: .attribute 5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zfh1p0_zfhmin1p0_zbb1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
; RV32ZBKB: .attribute 5, "rv32i2p1_zbkb1p0"
; RV32ZBKC: .attribute 5, "rv32i2p1_zbkc1p0"
; RV32ZBKX: .attribute 5, "rv32i2p1_zbkx1p0"
; RV32ZKND: .attribute 5, "rv32i2p1_zknd1p0"
; RV32ZKNE: .attribute 5, "rv32i2p1_zkne1p0"
; RV32ZKNH: .attribute 5, "rv32i2p1_zknh1p0"
; RV32ZKSED: .attribute 5, "rv32i2p1_zksed1p0"
; RV32ZKSH: .attribute 5, "rv32i2p1_zksh1p0"
; RV32ZKR: .attribute 5, "rv32i2p1_zkr1p0"
; RV32ZKN: .attribute 5, "rv32i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0"
; RV32ZKS: .attribute 5, "rv32i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zks1p0_zksed1p0_zksh1p0"
; RV32ZKT: .attribute 5, "rv32i2p1_zkt1p0"
; RV32ZK: .attribute 5, "rv32i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zk1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0_zkr1p0_zkt1p0"
; RV32COMBINEINTOZK: .attribute 5, "rv32i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zk1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0_zkr1p0_zkt1p0"
; RV32COMBINEINTOZKN: .attribute 5, "rv32i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0"
; RV32COMBINEINTOZKS: .attribute 5, "rv32i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zks1p0_zksed1p0_zksh1p0"
; RV32ZICBOM: .attribute 5, "rv32i2p1_zicbom1p0"
; RV32ZICBOZ: .attribute 5, "rv32i2p1_zicboz1p0"
; RV32ZICBOP: .attribute 5, "rv32i2p1_zicbop1p0"
; RV32SVNAPOT: .attribute 5, "rv32i2p1_svnapot1p0"
; RV32SVPBMT: .attribute 5, "rv32i2p1_svpbmt1p0"
; RV32SVINVAL: .attribute 5, "rv32i2p1_svinval1p0"
; RV32XCVALU: .attribute 5, "rv32i2p1_xcvalu1p0"
; RV32XCVBITMANIP: .attribute 5, "rv32i2p1_xcvbitmanip1p0"
; RV32XCVELW: .attribute 5, "rv32i2p1_xcvelw1p0"
; RV32XCVMAC: .attribute 5, "rv32i2p1_xcvmac1p0"
; RV32XCVMEM: .attribute 5, "rv32i2p1_xcvmem1p0"
; RV32XCVSIMD: .attribute 5, "rv32i2p1_xcvsimd1p0"
; RV32XCVBI: .attribute 5, "rv32i2p1_xcvbi1p0"
; RV32XSFVFWMACCQQQ: .attribute 5, "rv32i2p1_f2p2_zicsr2p0_zve32f1p0_zve32x1p0_zvfbfmin1p0_zvl32b1p0_xsfvfwmaccqqq1p0"
; RV32XTHEADCMO: .attribute 5, "rv32i2p1_xtheadcmo1p0"
; RV32XTHEADCONDMOV: .attribute 5, "rv32i2p1_xtheadcondmov1p0"
; RV32XTHEADFMEMIDX: .attribute 5, "rv32i2p1_f2p2_zicsr2p0_xtheadfmemidx1p0"
; RV32XTHEADMAC: .attribute 5, "rv32i2p1_xtheadmac1p0"
; RV32XTHEADMEMIDX: .attribute 5, "rv32i2p1_xtheadmemidx1p0"
; RV32XTHEADMEMPAIR: .attribute 5, "rv32i2p1_xtheadmempair1p0"
; RV32XTHEADSYNC: .attribute 5, "rv32i2p1_xtheadsync1p0"
; RV32ZCA: .attribute 5, "rv32i2p1_zca1p0"
; RV32ZCB: .attribute 5, "rv32i2p1_zca1p0_zcb1p0"
; RV32ZCD: .attribute 5, "rv32i2p1_f2p2_d2p2_zicsr2p0_zca1p0_zcd1p0"
; RV32ZCF: .attribute 5, "rv32i2p1_f2p2_zicsr2p0_zca1p0_zcf1p0"
; RV32ZCMP: .attribute 5, "rv32i2p1_zca1p0_zcmp1p0"
; RV32ZCMT: .attribute 5, "rv32i2p1_zicsr2p0_zca1p0_zcmt1p0"
; RV32ZICSR: .attribute 5, "rv32i2p1_zicsr2p0"
; RV32ZIFENCEI: .attribute 5, "rv32i2p1_zifencei2p0"
; RV32ZICNTR: .attribute 5, "rv32i2p1_zicntr2p0_zicsr2p0"
; RV32ZIHPM: .attribute 5, "rv32i2p1_zicsr2p0_zihpm2p0"
; RV32ZFA: .attribute 5, "rv32i2p1_f2p2_zicsr2p0_zfa1p0"
; RV32ZVBB: .attribute 5, "rv32i2p1_zicsr2p0_zvbb1p0_zve32x1p0_zvkb1p0_zvl32b1p0"
; RV32ZVBC: .attribute 5, "rv32i2p1_zicsr2p0_zvbc1p0_zve32x1p0_zve64x1p0_zvl32b1p0_zvl64b1p0"
; RV32ZVKB: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zvkb1p0_zvl32b1p0"
; RV32ZVKG: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zvkg1p0_zvl32b1p0"
; RV32ZVKN: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zve64x1p0_zvkb1p0_zvkn1p0_zvkned1p0_zvknhb1p0_zvkt1p0_zvl32b1p0_zvl64b1p0"
; RV32ZVKNC: .attribute 5, "rv32i2p1_zicsr2p0_zvbc1p0_zve32x1p0_zve64x1p0_zvkb1p0_zvkn1p0_zvknc1p0_zvkned1p0_zvknhb1p0_zvkt1p0_zvl32b1p0_zvl64b1p0"
; RV32ZVKNED: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zvkned1p0_zvl32b1p0"
; RV32ZVKNG: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zve64x1p0_zvkb1p0_zvkg1p0_zvkn1p0_zvkned1p0_zvkng1p0_zvknhb1p0_zvkt1p0_zvl32b1p0_zvl64b1p0"
; RV32ZVKNHA: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zvknha1p0_zvl32b1p0"
; RV32ZVKNHB: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zve64x1p0_zvknhb1p0_zvl32b1p0_zvl64b1p0"
; RV32ZVKS: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zve64x1p0_zvkb1p0_zvks1p0_zvksed1p0_zvksh1p0_zvkt1p0_zvl32b1p0_zvl64b1p0"
; RV32ZVKSC: .attribute 5, "rv32i2p1_zicsr2p0_zvbc1p0_zve32x1p0_zve64x1p0_zvkb1p0_zvks1p0_zvksc1p0_zvksed1p0_zvksh1p0_zvkt1p0_zvl32b1p0_zvl64b1p0"
; RV32ZVKSED: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zvksed1p0_zvl32b1p0"
; RV32ZVKSG: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zve64x1p0_zvkb1p0_zvkg1p0_zvks1p0_zvksed1p0_zvksg1p0_zvksh1p0_zvkt1p0_zvl32b1p0_zvl64b1p0"
; RV32ZVKSH: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zvksh1p0_zvl32b1p0"
; RV32ZVKT: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zvkt1p0_zvl32b1p0"
; RV32ZVFH: .attribute 5, "rv32i2p1_f2p2_zicsr2p0_zfhmin1p0_zve32f1p0_zve32x1p0_zvfh1p0_zvfhmin1p0_zvl32b1p0"
; RV32ZICOND: .attribute 5, "rv32i2p1_zicond1p0"
; RV32ZIMOP: .attribute 5, "rv32i2p1_zimop0p1"
; RV32ZCMOP: .attribute 5, "rv32i2p1_zca1p0_zcmop0p2"
; RV32SMAIA: .attribute 5, "rv32i2p1_smaia1p0"
; RV32SSAIA: .attribute 5, "rv32i2p1_ssaia1p0"
; RV32SMEPMP: .attribute 5, "rv32i2p1_smepmp1p0"
; RV32ZFBFMIN: .attribute 5, "rv32i2p1_f2p2_zicsr2p0_zfbfmin1p0"
; RV32ZVFBFMIN: .attribute 5, "rv32i2p1_f2p2_zicsr2p0_zve32f1p0_zve32x1p0_zvfbfmin1p0_zvl32b1p0"
; RV32ZVFBFWMA: .attribute 5, "rv32i2p1_f2p2_zicsr2p0_zfbfmin1p0_zve32f1p0_zve32x1p0_zvfbfmin1p0_zvfbfwma1p0_zvl32b1p0"
; RV32ZACAS: .attribute 5, "rv32i2p1_a2p1_zacas1p0"
; RV32ZICFILP: .attribute 5, "rv32i2p1_zicfilp0p4"

; RV64M: .attribute 5, "rv64i2p1_m2p0"
; RV64ZMMUL: .attribute 5, "rv64i2p1_zmmul1p0"
; RV64MZMMUL: .attribute 5, "rv64i2p1_m2p0_zmmul1p0"
; RV64A: .attribute 5, "rv64i2p1_a2p1"
; RV64F: .attribute 5, "rv64i2p1_f2p2_zicsr2p0"
; RV64D: .attribute 5, "rv64i2p1_f2p2_d2p2_zicsr2p0"
; RV64C: .attribute 5, "rv64i2p1_c2p0"
; RV64ZIHINTPAUSE: .attribute 5, "rv64i2p1_zihintpause2p0"
; RV64ZIHINTNTL: .attribute 5, "rv64i2p1_zihintntl1p0"
; RV64ZFHMIN: .attribute 5, "rv64i2p1_f2p2_zicsr2p0_zfhmin1p0"
; RV64ZFH: .attribute 5, "rv64i2p1_f2p2_zicsr2p0_zfh1p0_zfhmin1p0"
; RV64ZBA: .attribute 5, "rv64i2p1_zba1p0"
; RV64ZBB: .attribute 5, "rv64i2p1_zbb1p0"
; RV64ZBC: .attribute 5, "rv64i2p1_zbc1p0"
; RV64ZBS: .attribute 5, "rv64i2p1_zbs1p0"
; RV64V: .attribute 5, "rv64i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
; RV64H: .attribute 5, "rv64i2p1_h1p0"
; RV64COMBINED: .attribute 5, "rv64i2p1_f2p2_d2p2_v1p0_zicsr2p0_zfh1p0_zfhmin1p0_zbb1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
; RV64ZBKB: .attribute 5, "rv64i2p1_zbkb1p0"
; RV64ZBKC: .attribute 5, "rv64i2p1_zbkc1p0"
; RV64ZBKX: .attribute 5, "rv64i2p1_zbkx1p0"
; RV64ZKND: .attribute 5, "rv64i2p1_zknd1p0"
; RV64ZKNE: .attribute 5, "rv64i2p1_zkne1p0"
; RV64ZKNH: .attribute 5, "rv64i2p1_zknh1p0"
; RV64ZKSED: .attribute 5, "rv64i2p1_zksed1p0"
; RV64ZKSH: .attribute 5, "rv64i2p1_zksh1p0"
; RV64ZKR: .attribute 5, "rv64i2p1_zkr1p0"
; RV64ZKN: .attribute 5, "rv64i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0"
; RV64ZKS: .attribute 5, "rv64i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zks1p0_zksed1p0_zksh1p0"
; RV64ZKT: .attribute 5, "rv64i2p1_zkt1p0"
; RV64ZK: .attribute 5, "rv64i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zk1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0_zkr1p0_zkt1p0"
; RV64COMBINEINTOZK: .attribute 5, "rv64i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zk1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0_zkr1p0_zkt1p0"
; RV64COMBINEINTOZKN: .attribute 5, "rv64i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0"
; RV64COMBINEINTOZKS: .attribute 5, "rv64i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zks1p0_zksed1p0_zksh1p0"
; RV64ZIC64B: .attribute 5, "rv64i2p1_zic64b1p0"
; RV64ZICBOM: .attribute 5, "rv64i2p1_zicbom1p0"
; RV64ZICBOZ: .attribute 5, "rv64i2p1_zicboz1p0"
; RV64ZA64RS: .attribute 5, "rv64i2p1_za64rs1p0"
; RV64ZA128RS: .attribute 5, "rv64i2p1_za128rs1p0"
; RV64ZAWRS: .attribute 5, "rv64i2p1_zawrs1p0"
; RV64ZICBOP: .attribute 5, "rv64i2p1_zicbop1p0"
; RV64SVNAPOT: .attribute 5, "rv64i2p1_svnapot1p0"
; RV64SVPBMT: .attribute 5, "rv64i2p1_svpbmt1p0"
; RV64SVINVAL: .attribute 5, "rv64i2p1_svinval1p0"
; RV64XVENTANACONDOPS: .attribute 5, "rv64i2p1_xventanacondops1p0"
; RV64XSFVFWMACCQQQ: .attribute 5, "rv64i2p1_f2p2_zicsr2p0_zve32f1p0_zve32x1p0_zvfbfmin1p0_zvl32b1p0_xsfvfwmaccqqq1p0"
; RV64XTHEADBA: .attribute 5, "rv64i2p1_xtheadba1p0"
; RV64XTHEADBB: .attribute 5, "rv64i2p1_xtheadbb1p0"
; RV64XTHEADBS: .attribute 5, "rv64i2p1_xtheadbs1p0"
; RV64XTHEADCMO: .attribute 5, "rv64i2p1_xtheadcmo1p0"
; RV64XTHEADCONDMOV: .attribute 5, "rv64i2p1_xtheadcondmov1p0"
; RV64XTHEADFMEMIDX: .attribute 5, "rv64i2p1_f2p2_zicsr2p0_xtheadfmemidx1p0"
; RV64XTHEADMAC: .attribute 5, "rv64i2p1_xtheadmac1p0"
; RV64XTHEADMEMIDX: .attribute 5, "rv64i2p1_xtheadmemidx1p0"
; RV64XTHEADMEMPAIR: .attribute 5, "rv64i2p1_xtheadmempair1p0"
; RV64XTHEADSYNC: .attribute 5, "rv64i2p1_xtheadsync1p0"
; RV64XTHEADVDOT: .attribute 5, "rv64i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xtheadvdot1p0"
; RV64ZTSO: .attribute 5, "rv64i2p1_ztso0p1"
; RV64ZCA: .attribute 5, "rv64i2p1_zca1p0"
; RV64ZCB: .attribute 5, "rv64i2p1_zca1p0_zcb1p0"
; RV64ZCD: .attribute 5, "rv64i2p1_f2p2_d2p2_zicsr2p0_zca1p0_zcd1p0"
; RV64ZCMP: .attribute 5, "rv64i2p1_zca1p0_zcmp1p0"
; RV64ZCMT: .attribute 5, "rv64i2p1_zicsr2p0_zca1p0_zcmt1p0"
; RV64ZICCAMOA: .attribute 5, "rv64i2p1_ziccamoa1p0"
; RV64ZICCIF: .attribute 5, "rv64i2p1_ziccif1p0"
; RV64ZICCLSM: .attribute 5, "rv64i2p1_zicclsm1p0"
; RV64ZICCRSE: .attribute 5, "rv64i2p1_ziccrse1p0"
; RV64ZICSR: .attribute 5, "rv64i2p1_zicsr2p0"
; RV64ZIFENCEI: .attribute 5, "rv64i2p1_zifencei2p0"
; RV64ZICNTR: .attribute 5, "rv64i2p1_zicntr2p0_zicsr2p0"
; RV64ZIHPM: .attribute 5, "rv64i2p1_zicsr2p0_zihpm2p0"
; RV64ZFA: .attribute 5, "rv64i2p1_f2p2_zicsr2p0_zfa1p0"
; RV64ZVBB: .attribute 5, "rv64i2p1_zicsr2p0_zvbb1p0_zve32x1p0_zvkb1p0_zvl32b1p0"
; RV64ZVBC: .attribute 5, "rv64i2p1_zicsr2p0_zvbc1p0_zve32x1p0_zve64x1p0_zvl32b1p0_zvl64b1p0"
; RV64ZVKB: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zvkb1p0_zvl32b1p0"
; RV64ZVKG: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zvkg1p0_zvl32b1p0"
; RV64ZVKN: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zve64x1p0_zvkb1p0_zvkn1p0_zvkned1p0_zvknhb1p0_zvkt1p0_zvl32b1p0_zvl64b1p0"
; RV64ZVKNC: .attribute 5, "rv64i2p1_zicsr2p0_zvbc1p0_zve32x1p0_zve64x1p0_zvkb1p0_zvkn1p0_zvknc1p0_zvkned1p0_zvknhb1p0_zvkt1p0_zvl32b1p0_zvl64b1p0"
; RV64ZVKNED: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zvkned1p0_zvl32b1p0"
; RV64ZVKNG: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zve64x1p0_zvkb1p0_zvkg1p0_zvkn1p0_zvkned1p0_zvkng1p0_zvknhb1p0_zvkt1p0_zvl32b1p0_zvl64b1p0"
; RV64ZVKNHA: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zvknha1p0_zvl32b1p0"
; RV64ZVKNHB: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zve64x1p0_zvknhb1p0_zvl32b1p0_zvl64b1p0"
; RV64ZVKS: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zvkb1p0_zvks1p0_zvksed1p0_zvksh1p0_zvkt1p0_zvl32b1p0"
; RV64ZVKSC: .attribute 5, "rv64i2p1_zicsr2p0_zvbc1p0_zve32x1p0_zve64x1p0_zvkb1p0_zvks1p0_zvksc1p0_zvksed1p0_zvksh1p0_zvkt1p0_zvl32b1p0_zvl64b1p0"
; RV64ZVKSED: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zvksed1p0_zvl32b1p0"
; RV64ZVKSG: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zvkb1p0_zvkg1p0_zvks1p0_zvksed1p0_zvksg1p0_zvksh1p0_zvkt1p0_zvl32b1p0"
; RV64ZVKSH: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zvksh1p0_zvl32b1p0"
; RV64ZVKT: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zvkt1p0_zvl32b1p0"
; RV64ZVFH: .attribute 5, "rv64i2p1_f2p2_zicsr2p0_zfhmin1p0_zve32f1p0_zve32x1p0_zvfh1p0_zvfhmin1p0_zvl32b1p0"
; RV64ZICOND: .attribute 5, "rv64i2p1_zicond1p0"
; RV64ZIMOP: .attribute 5, "rv64i2p1_zimop0p1"
; RV64ZCMOP: .attribute 5, "rv64i2p1_zca1p0_zcmop0p2"
; RV64SMAIA: .attribute 5, "rv64i2p1_smaia1p0"
; RV64SSAIA: .attribute 5, "rv64i2p1_ssaia1p0"
; RV64SMEPMP: .attribute 5, "rv64i2p1_smepmp1p0"
; RV64ZFBFMIN: .attribute 5, "rv64i2p1_f2p2_zicsr2p0_zfbfmin1p0"
; RV64ZVFBFMIN: .attribute 5, "rv64i2p1_f2p2_zicsr2p0_zve32f1p0_zve32x1p0_zvfbfmin1p0_zvl32b1p0"
; RV64ZVFBFWMA: .attribute 5, "rv64i2p1_f2p2_zicsr2p0_zfbfmin1p0_zve32f1p0_zve32x1p0_zvfbfmin1p0_zvfbfwma1p0_zvl32b1p0"
; RV64ZACAS: .attribute 5, "rv64i2p1_a2p1_zacas1p0"
; RV64ZICFILP: .attribute 5, "rv64i2p1_zicfilp0p4"

define i32 @addi(i32 %a) {
  %1 = add i32 %a, 1
  ret i32 %1
}
