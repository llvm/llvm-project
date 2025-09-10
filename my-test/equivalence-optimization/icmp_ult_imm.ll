; target：改 riscv32 也可以
target triple = "riscv64"

define i32 @icmp_ult_imm(i32 %x) {
entry:
  %c = icmp ult i32 %x, 15
  %z = zext i1 %c to i32
  ret i32 %z
}
