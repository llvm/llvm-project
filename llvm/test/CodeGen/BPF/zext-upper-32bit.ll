; RUN: llc -march=bpfel -mcpu=v3 --filetype=obj < %s | llvm-objdump -d - \
; RUN: | FileCheck --check-prefix=ALU32 %s
; RUN: llc -march=bpfel -mcpu=v2 --filetype=obj < %s | llvm-objdump -d - \
; RUN: | FileCheck --check-prefix=NOALU32 %s

define dso_local i64 @test1(i64 %x) {
entry:
  %a = and i64 %x, 4294967295
  ret i64 %a
}
; ALU32:      <test1>:
; ALU32-NEXT: w0 = w1
; ALU32-NEXT: exit

; NOALU32:      <test1>:
; NOALU32-NEXT: r0 = r1
; NOALU32-NEXT: r0 <<= 0x20
; NOALU32-NEXT: r0 >>= 0x20
; NOALU32-NEXT: exit

define dso_local i64 @test2(i32 %x) {
entry:
  %a = zext i32 %x to i64
  ret i64 %a
}
; ALU32:      <test2>:
; ALU32-NEXT: w0 = w1
; ALU32-NEXT: exit

; NOALU32:      <test2>:
; NOALU32-NEXT: r0 = r1
; NOALU32-NEXT: r0 <<= 0x20
; NOALU32-NEXT: r0 >>= 0x20
; NOALU32-NEXT: exit
