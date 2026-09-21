; RUN: llc -mtriple=riscv64 -mcpu=sifive-u74 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mcpu=sifive-u74 -filetype=obj < %s \
; RUN:   | llvm-objdump -d --show-all-symbols --no-show-raw-insn - | FileCheck %s --check-prefix=OBJ

;; TODO: emitTargetFeaturePush does not call setArchString(), so per-function
;; target-features are not reflected in the $x<arch> mapping symbols.
; OBJ-LABEL: Disassembly of section .text:
; OBJ-EMPTY:
; OBJ-NEXT:  0000000000000000 <$xrv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0_zcd1p0>:
; OBJ-NEXT:  0000000000000000 <test1>:
; OBJ-NEXT:         0:      	ret
; OBJ-EMPTY:
; OBJ-NEXT:  0000000000000002 <test2>:
; OBJ-NEXT:         2:      	ret
; OBJ-EMPTY:
; OBJ-NEXT:  0000000000000004 <test3>:
; OBJ-NEXT:         4:      	ret
; OBJ-EMPTY:
; OBJ-NEXT:  0000000000000006 <test4>:
; OBJ-NEXT:         6:      	ret
; OBJ-EMPTY:
; OBJ-NEXT:  0000000000000008 <test5>:
; OBJ-NEXT:         8:      	ret
; OBJ-NOT:   {{.}}

; CHECK:      .option push
; CHECK-NEXT: .option arch, +v, +zve32f, +zve32x, +zve64d, +zve64f, +zve64x, +zvl128b, +zvl32b, +zvl64b{{$}}
define void @test1() "target-features"="+a,+d,+f,+m,+c,+v,+zifencei,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl128b,+zvl32b,+zvl64b" {
; CHECK-LABEL: test1:
; CHECK:       ret
; CHECK:       .option pop
entry:
  ret void
}

; CHECK-NEXT: .option push
; CHECK-NEXT: .option arch, +zihintntl{{$}}
define void @test2() "target-features"="+a,+d,+f,+m,+zihintntl,+zifencei" {
; CHECK-LABEL: test2:
; CHECK:       ret
; CHECK:       .option pop
entry:
  ret void
}

; CHECK-NEXT: .option push
; CHECK-NEXT: .option arch, -a, -d, -f, -m, -zcd{{$}}
define void @test3() "target-features"="-a,-d,-f,-m" {
; CHECK-LABEL: test3:
; CHECK:       ret
; CHECK:       .option pop
entry:
  ret void
}

; CHECK-NOT: .option push
define void @test4() {
; CHECK-LABEL: test4:
; CHECK:       ret
; CHECK-NOT:   .option pop
entry:
  ret void
}

; CHECK-NOT: .option push
define void @test5() "target-features"="+unaligned-scalar-mem" {
; CHECK-LABEL: test5:
; CHECK:       ret
; CHECK-NOT:   .option pop
entry:
  ret void
}
