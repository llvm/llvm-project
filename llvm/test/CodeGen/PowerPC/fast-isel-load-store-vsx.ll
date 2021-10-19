; RUN: llc < %s -O0 -fast-isel -mattr=+vsx -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -ppc-late-peephole=false | FileCheck %s --check-prefix=ELF64VSX

;; The semantics of VSX stores for when R0 is used is different depending on
;; whether it is used as base or offset. If used as base, the effective
;; address computation will use zero regardless of the content of R0. If used as
;; an offset the content will be used in the effective address. We observed that
;; for some constructors, the initialization values were being stored without
;; an offset register being specified which was causing R0 to be used as offset
;; in regions where it contained the value in the link register. This test
;; verifies that R0 is used as base in these situations.

%SomeStruct = type { double }

; ELF64VSX-LABEL: SomeStructCtor
define linkonce_odr void @SomeStructCtor(%SomeStruct* %this, double %V) unnamed_addr align 2 {
entry:
  %this.addr = alloca %SomeStruct*, align 8
  %V.addr = alloca double, align 8
  store %SomeStruct* %this, %SomeStruct** %this.addr, align 8
; ELF64VSX: stfd {{[0-9][0-9]?}}, -{{[1-9][0-9]?}}({{[1-9][0-9]?}})
  store double %V, double* %V.addr, align 8
  %this1 = load %SomeStruct*, %SomeStruct** %this.addr
  %Val = getelementptr inbounds %SomeStruct, %SomeStruct* %this1, i32 0, i32 0
; ELF64VSX: stxsdx {{[0-9][0-9]?}}, 0, {{[1-9][0-9]?}}
  %0 = load double, double* %V.addr, align 8
  store double %0, double* %Val, align 8
  ret void
 }
