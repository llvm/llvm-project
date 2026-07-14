; RUN: llc < %s -mtriple=riscv32 -mattr=+experimental-zicfiss \
; RUN:   -verify-machineinstrs -stop-after=prolog-epilog \
; RUN:   | FileCheck %s --check-prefix=RV32
; RUN: llc < %s -mtriple=riscv64 -mattr=+experimental-zicfiss \
; RUN:   -verify-machineinstrs -stop-after=prolog-epilog \
; RUN:   | FileCheck %s --check-prefix=RV64

declare i32 @bar()

define i32 @f() "hw-shadow-stack" {
; RV32-LABEL: name: f
; RV32: frame-setup SSPUSH
; RV32: frame-destroy SSPOPCHK
;
; RV64-LABEL: name: f
; RV64: frame-setup SSPUSH
; RV64: frame-destroy SSPOPCHK
  %res = call i32 @bar()
  %res1 = add i32 %res, 1
  ret i32 %res
}
