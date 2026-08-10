; RUN: llvm-as < %s | llvm-dis > %t0
; RUN: opt -S < %s > %t1
; RUN: diff %t0 %t1
; RUN: FileCheck < %t1 %s

; Make sure fast math flags on fcmp instructions are serialized/deserialized properly.

define i1 @foo(float %a, float %b, double %c, double %d) {
  ; CHECK:   %plain = fcmp ueq float %a, %b
  %plain = fcmp ueq float %a, %b
  ; CHECK:   %fast = fcmp fast olt float %a, %b
  %fast = fcmp fast olt float %a, %b
  ; CHECK:   %nsz = fcmp nsz uge float %a, %b
  %nsz = fcmp nsz uge float %a, %b
  ; CHECK:   %nnan = fcmp nnan nsz oge double %c, %d
  %nnan = fcmp nnan nsz oge double %c, %d

  %dce1 = or i1 %plain, %fast
  %dce2 = or i1 %dce1, %nsz
  %dce3 = or i1 %dce2, %nnan

  ret i1 %dce3
}
