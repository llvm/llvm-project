; RUN: llc -mtriple=aarch64 -mattr=+sve < %s 2>%t | FileCheck %s --check-prefix=CHECKCG
; RUN: llc -mtriple=aarch64 -mattr=+sve -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=CHECKISEL
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

; CHECKCG-LABEL: foo:
; CHECKCG: addvl   sp, sp, #-1

; CHECKISEL-LABEL: name: foo
; CHECKISEL:       stack:
; CHECKISEL:       id: 0, name: ptr, type: default, offset: 0, size: 16, alignment: 16,
; CHECKISEL-NEXT:  stack-id: sve-vec
define i32 @foo(<vscale x 16 x i8> %val) {
  %ptr = alloca <vscale x 16 x i8>
  %res = call i32 @bar(<vscale x 16 x i8>* %ptr)
  ret i32 %res
}

declare i32 @bar(<vscale x 16 x i8>* %ptr);

; CHECKCG-LABEL: foo2:
; CHECKCG: addvl   sp, sp, #-2

; CHECKISEL-LABEL: name: foo2
; CHECKISEL:       stack:
; CHECKISEL:       id: 0, name: ptr, type: default, offset: 0, size: 32, alignment: 16,
; CHECKISEL-NEXT:  stack-id: sve-vec

define i32 @foo2(<vscale x 32 x i8> %val) {
  %ptr = alloca <vscale x 32 x i8>, align 16
  %res = call i32 @bar2(<vscale x 32 x i8>* %ptr)
  ret i32 %res
}
declare i32 @bar2(<vscale x 32 x i8>* %ptr);
