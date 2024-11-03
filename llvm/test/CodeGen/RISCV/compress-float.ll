; This test is designed to run three times, once with function attributes, once
; with all target attributes added on the command line, and once with compress
; added with the command line and float added via function attributes, all
; three of these should result in the same output.
;
; RUN: cat %s > %t.tgtattr
; RUN: echo 'attributes #0 = { nounwind }' >> %t.tgtattr
; RUN: llc -mtriple=riscv32 -target-abi ilp32f -mattr=+c,+f -filetype=obj \
; RUN:   -disable-block-placement < %t.tgtattr \
; RUN:   | llvm-objdump -d --triple=riscv32 --mattr=+c,+f -M no-aliases - \
; RUN:   | FileCheck -check-prefix=RV32IFDC %s
;
; RUN: cat %s > %t.fnattr
; RUN: echo 'attributes #0 = { nounwind "target-features"="+c,+f" }' >> %t.fnattr
; RUN: llc -mtriple=riscv32 -target-abi ilp32f -filetype=obj \
; RUN:   -disable-block-placement < %t.fnattr \
; RUN:   | llvm-objdump -d --triple=riscv32 --mattr=+c,+f -M no-aliases - \
; RUN:   | FileCheck -check-prefix=RV32IFDC %s
;
; RUN: cat %s > %t.mixedattr
; RUN: echo 'attributes #0 = { nounwind "target-features"="+f" }' >> %t.mixedattr
; RUN: llc -mtriple=riscv32 -target-abi ilp32f -mattr=+c -filetype=obj \
; RUN:   -disable-block-placement < %t.mixedattr \
; RUN:   | llvm-objdump -d --triple=riscv32 --mattr=+c,+f -M no-aliases - \
; RUN:   | FileCheck -check-prefix=RV32IFDC %s
;
; RUN: cat %s > %t.tgtattr
; RUN: echo 'attributes #0 = { nounwind }' >> %t.tgtattr
; RUN: llc -mtriple=riscv32 -target-abi ilp32f -mattr=+zcf,+f -filetype=obj \
; RUN:   -disable-block-placement < %t.tgtattr \
; RUN:   | llvm-objdump -d --triple=riscv32 --mattr=+zcf,+f -M no-aliases - \
; RUN:   | FileCheck -check-prefix=RV32IFDC %s
;
; RUN: cat %s > %t.fnattr
; RUN: echo 'attributes #0 = { nounwind "target-features"="+zcf,+f" }' >> %t.fnattr
; RUN: llc -mtriple=riscv32 -target-abi ilp32f -filetype=obj \
; RUN:   -disable-block-placement < %t.fnattr \
; RUN:   | llvm-objdump -d --triple=riscv32 --mattr=+zcf,+f -M no-aliases - \
; RUN:   | FileCheck -check-prefix=RV32IFDC %s
;
; RUN: cat %s > %t.mixedattr
; RUN: echo 'attributes #0 = { nounwind "target-features"="+f" }' >> %t.mixedattr
; RUN: llc -mtriple=riscv32 -target-abi ilp32f -mattr=+zcf -filetype=obj \
; RUN:   -disable-block-placement < %t.mixedattr \
; RUN:   | llvm-objdump -d --triple=riscv32 --mattr=+zcf,+f -M no-aliases - \
; RUN:   | FileCheck -check-prefix=RV32IFDC %s
;
; RUN: cat %s > %t.tgtattr
; RUN: echo 'attributes #0 = { nounwind }' >> %t.tgtattr
; RUN: llc -mtriple=riscv32 -target-abi ilp32f -mattr=+zce,+f -filetype=obj \
; RUN:   -disable-block-placement < %t.tgtattr \
; RUN:   | llvm-objdump -d --triple=riscv32 --mattr=+zce,+f -M no-aliases - \
; RUN:   | FileCheck -check-prefix=RV32IFDC %s
;
; RUN: cat %s > %t.fnattr
; RUN: echo 'attributes #0 = { nounwind "target-features"="+zce,+f" }' >> %t.fnattr
; RUN: llc -mtriple=riscv32 -target-abi ilp32f -filetype=obj \
; RUN:   -disable-block-placement < %t.fnattr \
; RUN:   | llvm-objdump -d --triple=riscv32 --mattr=+zce,+f -M no-aliases - \
; RUN:   | FileCheck -check-prefix=RV32IFDC %s
;
; RUN: cat %s > %t.mixedattr
; RUN: echo 'attributes #0 = { nounwind "target-features"="+f" }' >> %t.mixedattr
; RUN: llc -mtriple=riscv32 -target-abi ilp32f -mattr=+zce -filetype=obj \
; RUN:   -disable-block-placement < %t.mixedattr \
; RUN:   | llvm-objdump -d --triple=riscv32 --mattr=+zce,+f -M no-aliases - \
; RUN:   | FileCheck -check-prefix=RV32IFDC %s

; This acts as a basic correctness check for the codegen instruction compression
; path, verifying that the assembled file contains compressed instructions when
; expected. Handling of the compressed ISA is implemented so the same
; transformation patterns should be used whether compressing an input .s file or
; compressing codegen output. This file contains basic functionality tests using
; instructions which also require one of the floating point extensions.

define float @float_load(ptr %a) #0 {
; RV32IFDC-LABEL: <float_load>:
; RV32IFDC:         c.flw fa0, 0(a0)
; RV32IFDC-NEXT:    c.jr ra
  %1 = load volatile float, ptr %a
  ret float %1
}
