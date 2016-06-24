; RUN: llc -march=mips64 -mcpu=mips64 -target-abi=n32 -relocation-model=pic < \
; RUN:     %s | FileCheck %s -check-prefixes=ALL,N32,HARD-FLOAT
; RUN: llc -march=mips64el -mcpu=mips64 -target-abi=n32 -relocation-model=pic < \
; RUN:     %s | FileCheck %s -check-prefixes=ALL,N32,HARD-FLOAT

; RUN: llc -march=mips64 -mcpu=mips64 -target-abi=n64 -relocation-model=pic < \
; RUN:     %s | FileCheck %s -check-prefixes=ALL,N64,HARD-FLOAT
; RUN: llc -march=mips64el -mcpu=mips64 -target-abi=n64 -relocation-model=pic < \
; RUN:     %s | FileCheck %s -check-prefixes=ALL,N64,HARD-FLOAT

; RUN: llc -march=mips64 -mcpu=mips64 -mattr=+soft-float -target-abi=n32 \
; RUN:     -relocation-model=pic < %s | FileCheck %s -check-prefixes=ALL,N32,SOFT-FLOAT
; RUN: llc -march=mips64el -mcpu=mips64 -mattr=+soft-float -target-abi=n32 \
; RUN:     -relocation-model=pic < %s | FileCheck %s -check-prefixes=ALL,N32,SOFT-FLOAT

; RUN: llc -march=mips64 -mcpu=mips64 -mattr=+soft-float -target-abi=n64 < %s \
; RUN: | FileCheck %s -check-prefixes=ALL,N64,SOFT-FLOAT
; RUN: llc -march=mips64el -mcpu=mips64 -mattr=+soft-float -target-abi=n64 < \
; RUN:     %s | FileCheck %s -check-prefixes=ALL,N64,SOFT-FLOAT

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
