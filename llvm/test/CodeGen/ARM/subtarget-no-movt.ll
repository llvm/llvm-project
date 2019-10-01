; RUN: llc -mcpu=cortex-a8 -relocation-model=static %s -o - | \
; RUN:   FileCheck -check-prefixes=CHECK,NO-OPTION,NO-OPTION-COMMON %s
; RUN: llc -mcpu=cortex-a8 -relocation-model=static %s -o - -mattr=-no-movt | \
; RUN:   FileCheck -check-prefixes=CHECK,USE-MOVT,USE-MOVT-COMMON %s
; RUN: llc -mcpu=cortex-a8 -relocation-model=static %s -o - -mattr=+no-movt | \
; RUN:   FileCheck -check-prefixes=CHECK,NO-USE-MOVT,NO-USE-MOVT-COMMON %s
; RUN: llc -mcpu=cortex-a8 -relocation-model=static %s -o - -O0 | \
; RUN:   FileCheck -check-prefixes=CHECK,NO-OPTION-O0,NO-OPTION-COMMON %s
; RUN: llc -mcpu=cortex-a8 -relocation-model=static %s -o - -O0 -mattr=-no-movt | \
; RUN:  FileCheck -check-prefixes=CHECK,USE-MOVT-O0,USE-MOVT-COMMON %s
; RUN: llc -mcpu=cortex-a8 -relocation-model=static %s -o - -O0 -mattr=+no-movt | \
; RUN:   FileCheck -check-prefixes=CHECK,NO-USE-MOVT-O0,NO-USE-MOVT-COMMON %s

target triple = "thumb-apple-darwin"

; NO-OPTION-COMMON-LABEL: {{_?}}foo0
; NO-OPTION-COMMON: ldr [[R0:r[0-9]+]], [[L0:.*]]
; NO-OPTION-COMMON: [[L0]]:
; NO-OPTION-COMMON: .long 2296237089

; USE-MOVT-COMMON-LABEL: {{_?}}foo0
; USE-MOVT-COMMON: movw [[R0:r[0-9]+]], #52257
; USE-MOVT-COMMON: movt [[R0]], #35037

; NO-USE-MOVT-COMMON-LABEL: {{_?}}foo0
; NO-USE-MOVT-COMMON: ldr [[R0:r[0-9]+]], [[L0:.*]]
; NO-USE-MOVT-COMMON: [[L0]]:
; NO-USE-MOVT-COMMON: .long 2296237089

define i32 @foo0(i32 %a) #0 {
  %1 = xor i32 -1998730207, %a
  ret i32 %1
}

; NO-OPTION-COMMON-LABEL: {{_?}}foo1
; NO-OPTION-COMMON: movw [[R0:r[0-9]+]], #52257
; NO-OPTION-COMMON: movt [[R0]], #35037

; USE-MOVT-COMMON-LABEL: {{_?}}foo1
; USE-MOVT-COMMON: movw [[R0:r[0-9]+]], #52257
; USE-MOVT-COMMON: movt [[R0]], #35037

; NO-USE-MOVT-COMMON-LABEL: {{_?}}foo1
; NO-USE-MOVT-COMMON: ldr [[R0:r[0-9]+]], [[L0:.*]]
; NO-USE-MOVT-COMMON: [[L0]]:
; NO-USE-MOVT-COMMON: .long 2296237089

define i32 @foo1(i32 %a) {
  %1 = xor i32 -1998730207, %a
  ret i32 %1
}

; NO-OPTION-COMMON-LABEL: {{_?}}foo2
; NO-OPTION:   mov.w	[[R0:r[0-9]+]], #-536813568
; NO-OPTION-O0: movw	[[R0:r[0-9]+]], #57344
; NO-OPTION-O0: movt	[[R0]], #57344

; USE-MOVT-COMMON-LABEL: {{_?}}foo2
; USE-MOVT:     mov.w [[R0:r[0-9]+]], #-536813568
; USE-MOVT-O0:  movw  [[R0:r[0-9]+]], #57344
; USE-MOVT-O0:  movt  [[R0]], #57344

; NO-USE-MOVT-COMMON-LABEL: {{_?}}foo2
; NO-USE-MOVT: mov.w	[[R0:r[0-9]+]], #-536813568
; NO-USE-MOVT-O0: ldr [[R0:r[0-9]+]], [[L0:.*]]
; NO-USE-MOVT-O0: [[L0]]:
; NO-USE-MOVT-O0: .long 3758153728     @ 0xe000e000

define i32 @foo2() {
  %1 = load i32, i32* inttoptr (i32 -536813568 to i32*) ; load from 0xe000e000
  ret i32 %1
}
attributes #0 = { "target-features"="+no-movt" }

define hidden i32 @no_litpool() minsize optsize {
; CHECK-LABEL:  no_litpool:
; CHECK:        mov.w r{{.}}, #65536
; CHECK:        mov.w r{{.}}, #-134217728
; CHECK:        mvn r{{.}}, #-134217728
entry:
  %call0 = tail call i32 @eat_const(i32 65536)
  %call1 = tail call i32 @eat_const(i32 -134217728)
  %call2 = tail call i32 @eat_const(i32 134217727)
  ret i32 %call2
}

define hidden i32 @litpool() minsize optsize {
; CHECK-LABEL:  litpool:
; CHECK:        ldr r0, {{.*}}LCPI{{.*}}
; CHECK-NEXT:   b.w {{.*}}eat_const
entry:
  %call1 = tail call i32 @eat_const(i32 8388601)
  ret i32 %call1
}

declare dso_local i32 @eat_const(i32)

