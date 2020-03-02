; This test is designed to run twice, once with function attributes and once
; with target attributes added on the command line.
;
; RUN: cat %s > %t.tgtattr
; RUN: echo 'attributes #0 = { nounwind }' >> %t.tgtattr
; RUN: llc -mtriple=riscv32 -mattr=+c -filetype=obj \
; RUN:   -disable-block-placement < %t.tgtattr \
; RUN:   | llvm-objdump -d -triple=riscv32 -mattr=+c -M no-aliases - \
; RUN:   | FileCheck -check-prefix=RV32IC %s
;
; RUN: cat %s > %t.fnattr
; RUN: echo 'attributes #0 = { nounwind "target-features"="+c" }' >> %t.fnattr
; RUN: llc -mtriple=riscv32 -filetype=obj \
; RUN:   -disable-block-placement < %t.fnattr \
; RUN:   | llvm-objdump -d -triple=riscv32 -mattr=+c -M no-aliases - \
; RUN:   | FileCheck -check-prefix=RV32IC %s

; This acts as a sanity check for the codegen instruction compression path,
; verifying that the assembled file contains compressed instructions when
; expected. Handling of the compressed ISA is implemented so the same
; transformation patterns should be used whether compressing an input .s file or
; compressing codegen output. This file contains sanity checks to ensure that is
; working as expected. Particular care should be taken to test pseudo
; instructions.

; Note: TODOs in this file are only appropriate if they highlight a case where
; a generated instruction that can be compressed by an existing pattern isn't.
; It may be useful to have tests that indicate where better compression would be
; possible if alternative codegen choices were made, but they belong in a
; different test file.

define i32 @simple_arith(i32 %a, i32 %b) #0 {
; RV32IC-LABEL: simple_arith:
; RV32IC:         addi a2, a0, 1
; RV32IC-NEXT:    c.andi a2, 11
; RV32IC-NEXT:    c.slli a2, 7
; RV32IC-NEXT:    c.srai a1, 9
; RV32IC-NEXT:    c.add a1, a2
; RV32IC-NEXT:    sub a0, a1, a0
; RV32IC-NEXT:    c.jr ra
  %1 = add i32 %a, 1
  %2 = and i32 %1, 11
  %3 = shl i32 %2, 7
  %4 = ashr i32 %b, 9
  %5 = add i32 %3, %4
  %6 = sub i32 %5, %a
  ret i32 %6
}

define i32 @select(i32 %a, i32 *%b) #0 {
; RV32IC-LABEL: select:
; RV32IC:         c.lw a2, 0(a1)
; RV32IC-NEXT:    c.beqz a2, 4
; RV32IC-NEXT:    c.mv a0, a2
; RV32IC-NEXT:    c.lw a2, 0(a1)
; RV32IC-NEXT:    c.bnez a2, 4
; RV32IC-NEXT:    c.mv a0, a2
; RV32IC-NEXT:    c.lw a2, 0(a1)
; RV32IC-NEXT:    bltu a2, a0, 6
; RV32IC-NEXT:    c.mv a0, a2
; RV32IC-NEXT:    c.lw a2, 0(a1)
; RV32IC-NEXT:    bgeu a0, a2, 6
; RV32IC-NEXT:    c.mv a0, a2
; RV32IC-NEXT:    c.lw a2, 0(a1)
; RV32IC-NEXT:    bltu a0, a2, 6
; RV32IC-NEXT:    c.mv a0, a2
; RV32IC-NEXT:    c.lw a2, 0(a1)
; RV32IC-NEXT:    bgeu a2, a0, 6
; RV32IC-NEXT:    c.mv a0, a2
; RV32IC-NEXT:    c.lw a2, 0(a1)
; RV32IC-NEXT:    blt a2, a0, 6
; RV32IC-NEXT:    c.mv a0, a2
; RV32IC-NEXT:    c.lw a2, 0(a1)
; RV32IC-NEXT:    bge a0, a2, 6
; RV32IC-NEXT:    c.mv a0, a2
; RV32IC-NEXT:    c.lw a2, 0(a1)
; RV32IC-NEXT:    blt a0, a2, 6
; RV32IC-NEXT:    c.mv a0, a2
; RV32IC-NEXT:    c.lw a1, 0(a1)
; RV32IC-NEXT:    bge a1, a0, 6
; RV32IC-NEXT:    c.mv a0, a1
; RV32IC-NEXT:    c.jr ra
  %val1 = load volatile i32, i32* %b
  %tst1 = icmp eq i32 0, %val1
  %val2 = select i1 %tst1, i32 %a, i32 %val1

  %val3 = load volatile i32, i32* %b
  %tst2 = icmp ne i32 0, %val3
  %val4 = select i1 %tst2, i32 %val2, i32 %val3

  %val5 = load volatile i32, i32* %b
  %tst3 = icmp ugt i32 %val4, %val5
  %val6 = select i1 %tst3, i32 %val4, i32 %val5

  %val7 = load volatile i32, i32* %b
  %tst4 = icmp uge i32 %val6, %val7
  %val8 = select i1 %tst4, i32 %val6, i32 %val7

  %val9 = load volatile i32, i32* %b
  %tst5 = icmp ult i32 %val8, %val9
  %val10 = select i1 %tst5, i32 %val8, i32 %val9

  %val11 = load volatile i32, i32* %b
  %tst6 = icmp ule i32 %val10, %val11
  %val12 = select i1 %tst6, i32 %val10, i32 %val11

  %val13 = load volatile i32, i32* %b
  %tst7 = icmp sgt i32 %val12, %val13
  %val14 = select i1 %tst7, i32 %val12, i32 %val13

  %val15 = load volatile i32, i32* %b
  %tst8 = icmp sge i32 %val14, %val15
  %val16 = select i1 %tst8, i32 %val14, i32 %val15

  %val17 = load volatile i32, i32* %b
  %tst9 = icmp slt i32 %val16, %val17
  %val18 = select i1 %tst9, i32 %val16, i32 %val17

  %val19 = load volatile i32, i32* %b
  %tst10 = icmp sle i32 %val18, %val19
  %val20 = select i1 %tst10, i32 %val18, i32 %val19

  ret i32 %val20
}

define i32 @pos_tiny() #0 {
; RV32IC-LABEL: pos_tiny:
; RV32IC:         c.li a0, 18
; RV32IC-NEXT:    c.jr ra
  ret i32 18
}

define i32 @pos_i32() #0 {
; RV32IC-LABEL: pos_i32:
; RV32IC:         lui a0, 423811
; RV32IC-NEXT:    addi a0, a0, -1297
; RV32IC-NEXT:    c.jr ra
  ret i32 1735928559
}

define i32 @pos_i32_half_compressible() #0 {
; RV32IC-LABEL: pos_i32_half_compressible:
; RV32IC:         lui a0, 423810
; RV32IC-NEXT:    c.addi  a0, 28
; RV32IC-NEXT:    c.jr    ra
  ret i32 1735925788
}

define i32 @neg_tiny() #0 {
; RV32IC-LABEL: neg_tiny:
; RV32IC:       c.li a0, -19
; RV32IC-NEXT:  c.jr ra
  ret i32 -19
}

define i32 @neg_i32() #0 {
; RV32IC-LABEL: neg_i32:
; RV32IC:       lui a0, 912092
; RV32IC-NEXT:  addi a0, a0, -273
; RV32IC-NEXT:  c.jr ra
  ret i32 -559038737
}

define i32 @pos_i32_hi20_only() #0 {
; RV32IC-LABEL: pos_i32_hi20_only:
; RV32IC:       c.lui a0, 16
; RV32IC-NEXT:  c.jr ra
  ret i32 65536
}

define i32 @neg_i32_hi20_only() #0 {
; RV32IC-LABEL: neg_i32_hi20_only:
; RV32IC:       c.lui a0, 1048560
; RV32IC-NEXT:  c.jr ra
  ret i32 -65536
}
