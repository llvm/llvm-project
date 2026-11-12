; Check that spills and reloads are not generated for functions that do not
; have HLX arguments or return values.

; RUN: llc -march=hexagon -mattr=+hlx -mcpu=hexagonv85 -enable-misched=false -stop-after=prolog-epilog < %s | FileCheck %s --check-prefix=CHECK-PEI

target triple = "hexagon"

; Scalar function calls scalar function: no spills
; CHECK-PEI-LABEL: name: foo
; CHECK-PEI-NOT: PS_storerl
; CHECK-PEI-NOT: PS_loadrl
define dso_local i32 @foo(i32 noundef %a, i32 noundef %b) local_unnamed_addr {
entry:
  %add = add nsw i32 %b, %a
  %call = tail call i32 @bar(i32 noundef %b)
  %add1 = add nsw i32 %add, %call
  ret i32 %add1
}

declare dso_local i32 @bar(i32 noundef) local_unnamed_addr

; Scalar function calls HLX function: no spills
; CHECK-PEI-LABEL: foo_hlx_callee
; CHECK-PEI-NOT: PS_storerl
; CHECK-PEI-NOT: PS_loadrl
define dso_local i32 @foo_hlx_callee(i32 noundef %a, i32 noundef %b, ptr noundef writeonly captures(none) initializes((0, 2048)) %dest) local_unnamed_addr {
entry:
  %add = add nsw i32 %b, %a
  %call = tail call inreg <1024 x i16> @hlx_bar(i32 noundef %b)
  store <1024 x i16> %call, ptr %dest, align 2048
  ret i32 %add
}

declare dso_local inreg <1024 x i16> @hlx_bar(i32 noundef) local_unnamed_addr

; HLX function calls scalar function: spills, as callee may not preserved HLX CSRs
; CHECK-PEI-LABEL: name: foo_hlx_caller
; CHECK-PEI: PS_storerl {{.*}}, killed $l12
; CHECK-PEI: PS_storerl {{.*}}, killed $l13
; CHECK-PEI: PS_storerl {{.*}}, killed $l14
; CHECK-PEI: PS_storerl {{.*}}, killed $l15
; CHECK-PEI: J2_call @bar, hexagoncsr,
; CHECK-PEI: $l12 = PS_loadrl
; CHECK-PEI: $l13 = PS_loadrl
; CHECK-PEI: $l14 = PS_loadrl
; CHECK-PEI: $l15 = PS_loadrl
define dso_local inreg <1024 x i16> @foo_hlx_caller(i32 noundef %a, i32 noundef %b) local_unnamed_addr {
entry:
  %add = add nsw i32 %b, %a
  %call = tail call i32 @bar(i32 noundef %b)
  %add1 = add nsw i32 %add, %call
  %0 = tail call <1024 x i16> @llvm.hexagon.LX.assign.r(i32 %add1)
  ret <1024 x i16> %0
}

declare <1024 x i16> @llvm.hexagon.LX.assign.r(i32)

; HLX function calls HLX function: no spills, as callee preserves HLX CSRs
; CHECK-PEI-LABEL: name: foo_hlx_caller_callee
; CHECK-PEI-NOT: PS_storerl
; CHECK-PEI-NOT: PS_loadrl
define dso_local inreg <1024 x i16> @foo_hlx_caller_callee(i32 noundef %a, i32 noundef %b) local_unnamed_addr {
entry:
  %add = add nsw i32 %b, %a
  %call = tail call inreg <1024 x i16> @hlx_bar(i32 noundef %b)
  %0 = tail call <1024 x i16> @llvm.hexagon.LX.assign.r(i32 %add)
  %add1 = add <1024 x i16> %0, %call
  ret <1024 x i16> %add1
}
