; RUN: llc -march=mips64 -mcpu=mips64 -target-abi=n32 < %s | FileCheck %s \
; RUN:     -check-prefix=ALL -check-prefix=N32 -check-prefix=HARD-FLOAT
; RUN: llc -march=mips64el -mcpu=mips64 -target-abi=n32 < %s | FileCheck %s \
; RUN:     -check-prefix=ALL -check-prefix=N32 -check-prefix=HARD-FLOAT

; RUN: llc -march=mips64 -mcpu=mips64 -target-abi=n64 < %s | FileCheck %s \
; RUN:     -check-prefix=ALL -check-prefix=N64 -check-prefix=HARD-FLOAT
; RUN: llc -march=mips64el -mcpu=mips64 -target-abi=n64 < %s | FileCheck %s \
; RUN:     -check-prefix=ALL -check-prefix=N64 -check-prefix=HARD-FLOAT

; RUN: llc -march=mips64 -mcpu=mips64 -mattr=+soft-float -target-abi=n32 < %s \
; RUN: | FileCheck %s -check-prefix=ALL -check-prefix=N32 \
; RUN:                -check-prefix=SOFT-FLOAT
; RUN: llc -march=mips64el -mcpu=mips64 -mattr=+soft-float -target-abi=n32 < \
; RUN:     %s | FileCheck %s -check-prefix=ALL -check-prefix=N32 \
; RUN:                       -check-prefix=SOFT-FLOAT

; RUN: llc -march=mips64 -mcpu=mips64 -mattr=+soft-float -target-abi=n64 < %s \
; RUN: | FileCheck %s -check-prefix=ALL -check-prefix=N64 \
; RUN:                -check-prefix=SOFT-FLOAT
; RUN: llc -march=mips64el -mcpu=mips64 -mattr=+soft-float -target-abi=n64 < \
; RUN:     %s | FileCheck %s -check-prefix=ALL -check-prefix=N64 \
; RUN:                       -check-prefix=SOFT-FLOAT

@fp128 = global fp128 zeroinitializer

define void @roundl_call(fp128 %value) {
entry:
; ALL-LABEL: roundl_call:
; N32:          lw      $25, %call16(roundl)($gp)
; N64:          ld      $25, %call16(roundl)($gp)

; SOFT-FLOAT:   sd      $4, 8(${{[0-9]+}})
; SOFT-FLOAT:   sd      $2, 0(${{[0-9]+}})

; HARD-FLOAT:   sdc1    $f2, 8(${{[0-9]+}})
; HARD-FLOAT:   sdc1    $f0, 0(${{[0-9]+}})

  %call = call fp128 @roundl(fp128 %value)
  store fp128 %call, fp128* @fp128
  ret void
}

declare fp128 @roundl(fp128) nounwind readnone
