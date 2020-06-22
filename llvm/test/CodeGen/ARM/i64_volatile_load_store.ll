; RUN: llc -mtriple=armv5e-arm-none-eabi %s -o - | FileCheck %s --check-prefixes=CHECK-ARMV5TE,CHECK
; RUN: llc -mtriple=thumbv6t2-arm-none-eabi %s -o - | FileCheck %s --check-prefixes=CHECK-T2,CHECK
; RUN: llc -mtriple=armv4t-arm-none-eabi %s -o - | FileCheck %s --check-prefixes=CHECK-ARMV4T,CHECK

@x = common dso_local global i64 0, align 8
@y = common dso_local global i64 0, align 8

define void @test() {
entry:
; CHECK-LABEL: test:
; CHECK-ARMV5TE:      ldr [[ADDR0:r[0-9]+]]
; CHECK-ARMV5TE-NEXT: ldr [[ADDR1:r[0-9]+]]
; CHECK-ARMV5TE-NEXT: ldrd [[R0:r[0-9]+]], [[R1:r[0-9]+]], {{\[}}[[ADDR0]]]
; CHECK-ARMV5TE-NEXT: strd [[R0]], [[R1]], {{\[}}[[ADDR1]]]
; CHECK-T2:           movw [[ADDR0:r[0-9]+]], :lower16:x
; CHECK-T2-NEXT:      movw [[ADDR1:r[0-9]+]], :lower16:y
; CHECK-T2-NEXT:      movt [[ADDR0]], :upper16:x
; CHECK-T2-NEXT:      movt [[ADDR1]], :upper16:y
; CHECK-T2-NEXT:      ldrd [[R0:r[0-9]+]], [[R1:r[0-9]+]], {{\[}}[[ADDR0]]]
; CHECK-T2-NEXT:      strd [[R0]], [[R1]], {{\[}}[[ADDR1]]]
; CHECK-ARMV4T:       ldr [[ADDR0:r[0-9]+]]
; CHECK-ARMV4T-NEXT:  ldr [[ADDR1:r[0-9]+]]
; CHECK-ARMV4T-NEXT:  ldr [[R1:r[0-9]+]], {{\[}}[[ADDR0]]]
; CHECK-ARMV4T-NEXT:  ldr [[R0:r[0-9]+]], {{\[}}[[ADDR0]], #4]
; CHECK-ARMV4T-NEXT:  str [[R0]], {{\[}}[[ADDR1]], #4]
; CHECK-ARMV4T-NEXT:  str [[R1]], {{\[}}[[ADDR1]]]
  %0 = load volatile i64, i64* @x, align 8
  store volatile i64 %0, i64* @y, align 8
  ret void
}

define void @test_offset() {
entry:
; CHECK-LABEL: test_offset:
; CHECK-ARMV5TE:      ldr [[ADDR0:r[0-9]+]]
; CHECK-ARMV5TE-NEXT: ldr [[ADDR1:r[0-9]+]]
; CHECK-ARMV5TE-NEXT: ldrd [[R0:r[0-9]+]], [[R1:r[0-9]+]], {{\[}}[[ADDR0]], #-4]
; CHECK-ARMV5TE-NEXT: strd [[R0]], [[R1]], {{\[}}[[ADDR1]], #-4]
; CHECK-T2:           movw [[ADDR0:r[0-9]+]], :lower16:x
; CHECK-T2-NEXT:      movw [[ADDR1:r[0-9]+]], :lower16:y
; CHECK-T2-NEXT:      movt [[ADDR0]], :upper16:x
; CHECK-T2-NEXT:      movt [[ADDR1]], :upper16:y
; CHECK-T2-NEXT:      ldrd [[R0:r[0-9]+]], [[R1:r[0-9]+]], {{\[}}[[ADDR0]], #-4]
; CHECK-T2-NEXT:      strd [[R0]], [[R1]], {{\[}}[[ADDR1]], #-4]
; CHECK-ARMV4T:       ldr [[ADDR0:r[0-9]+]]
; CHECK-ARMV4T-NEXT:  ldr [[ADDR1:r[0-9]+]]
; CHECK-ARMV4T-NEXT:  ldr [[R0:r[0-9]+]], {{\[}}[[ADDR0]], #-4]
; CHECK-ARMV4T-NEXT:  ldr [[R1:r[0-9]+]], {{\[}}[[ADDR0]]]
; CHECK-ARMV4T-NEXT:  str [[R1]], {{\[}}[[ADDR1]]]
; CHECK-ARMV4T-NEXT:  str [[R0]], {{\[}}[[ADDR1]], #-4]
  %0 = load volatile i64, i64* bitcast (i8* getelementptr (i8, i8* bitcast (i64* @x to i8*), i32 -4) to i64*), align 8
  store volatile i64 %0, i64* bitcast (i8* getelementptr (i8, i8* bitcast (i64* @y to i8*), i32 -4) to i64*), align 8
  ret void
}

define void @test_offset_1() {
; CHECK-LABEL: test_offset_1:
; CHECK-ARMV5TE:      ldr [[ADDR0:r[0-9]+]]
; CHECK-ARMV5TE-NEXT: ldr [[ADDR1:r[0-9]+]]
; CHECK-ARMV5TE-NEXT: ldrd [[R0:r[0-9]+]], [[R1:r[0-9]+]], {{\[}}[[ADDR0]], #255]
; CHECK-ARMV5TE-NEXT: strd [[R0]], [[R1]], {{\[}}[[ADDR1]], #255]
; CHECK-T2:           adds [[ADDR0:r[0-9]+]], #255
; CHECK-T2-NEXT:      adds [[ADDR1:r[0-9]+]], #255
; CHECK-T2-NEXT:      ldrd [[R0:r[0-9]+]], [[R1:r[0-9]+]], {{\[}}[[ADDR0]]]
; CHECK-T2-NEXT:      strd [[R0]], [[R1]], {{\[}}[[ADDR1]]]
; CHECK-ARMV4T:       ldr [[ADDR0:r[0-9]+]]
; CHECK-ARMV4T-NEXT:  ldr [[ADDR1:r[0-9]+]]
; CHECK-ARMV4T-NEXT:  ldr [[R0:r[0-9]+]], {{\[}}[[ADDR0]], #255]
; CHECK-ARMV4T-NEXT:  ldr [[R1:r[0-9]+]], {{\[}}[[ADDR0]], #259]
; CHECK-ARMV4T-NEXT:  str [[R1]], {{\[}}[[ADDR1]], #259]
; CHECK-ARMV4T-NEXT:  str [[R0]], {{\[}}[[ADDR1]], #255]
entry:
  %0 = load volatile i64, i64* bitcast (i8* getelementptr (i8, i8* bitcast (i64* @x to i8*), i32 255) to i64*), align 8
  store volatile i64 %0, i64* bitcast (i8* getelementptr (i8, i8* bitcast (i64* @y to i8*), i32 255) to i64*), align 8
  ret void
}

define void @test_offset_2() {
; CHECK-LABEL: test_offset_2:
; CHECK-ARMV5TE:      ldr [[ADDR0:r[0-9]+]]
; CHECK-ARMV5TE-NEXT: ldr [[ADDR1:r[0-9]+]]
; CHECK-ARMV5TE-NEXT: add [[ADDR0]], [[ADDR0]], #256
; CHECK-ARMV5TE-NEXT: add [[ADDR1]], [[ADDR1]], #256
; CHECK-ARMV5TE-NEXT: ldrd [[R0:r[0-9]+]], [[R1:r[0-9]+]], {{\[}}[[ADDR0]]]
; CHECK-ARMV5TE-NEXT: strd [[R0]], [[R1]], {{\[}}[[ADDR1]]]
; CHECK-T2:           movw [[ADDR0:r[0-9]+]], :lower16:x
; CHECK-T2-NEXT:      movw [[ADDR1:r[0-9]+]], :lower16:y
; CHECK-T2-NEXT:      movt [[ADDR0]], :upper16:x
; CHECK-T2-NEXT:      movt [[ADDR1]], :upper16:y
; CHECK-T2-NEXT:      ldrd [[R0:r[0-9]+]], [[R1:r[0-9]+]], {{\[}}[[ADDR0]], #256]
; CHECK-T2-NEXT:      strd [[R0]], [[R1]], {{\[}}[[ADDR1]], #256]
; CHECK-ARMV4T:       ldr [[ADDR0:r[0-9]+]]
; CHECK-ARMV4T-NEXT:  ldr [[ADDR1:r[0-9]+]]
; CHECK-ARMV4T-NEXT:  ldr [[R0:r[0-9]+]], {{\[}}[[ADDR0]], #256]
; CHECK-ARMV4T-NEXT:  ldr [[R1:r[0-9]+]], {{\[}}[[ADDR0]], #260]
; CHECK-ARMV4T-NEXT:  str [[R1]], {{\[}}[[ADDR1]], #260]
; CHECK-ARMV4T-NEXT:  str [[R0]], {{\[}}[[ADDR1]], #256]
entry:
  %0 = load volatile i64, i64* bitcast (i8* getelementptr (i8, i8* bitcast (i64* @x to i8*), i32 256) to i64*), align 8
  store volatile i64 %0, i64* bitcast (i8* getelementptr (i8, i8* bitcast (i64* @y to i8*), i32 256) to i64*), align 8
  ret void
}

define void @test_offset_3() {
; CHECK-LABEL: test_offset_3:
; CHECK-ARMV5TE:      ldr [[ADDR0:r[0-9]+]]
; CHECK-ARMV5TE-NEXT: ldr [[ADDR1:r[0-9]+]]
; CHECK-ARMV5TE-NEXT: add [[ADDR0]], [[ADDR0]], #1020
; CHECK-ARMV5TE-NEXT: add [[ADDR1]], [[ADDR1]], #1020
; CHECK-ARMV5TE-NEXT: ldrd [[R0:r[0-9]+]], [[R1:r[0-9]+]], {{\[}}[[ADDR0]]]
; CHECK-ARMV5TE-NEXT: strd [[R0]], [[R1]], {{\[}}[[ADDR1]]]
; CHECK-T2:           movw [[ADDR0:r[0-9]+]], :lower16:x
; CHECK-T2-NEXT:      movw [[ADDR1:r[0-9]+]], :lower16:y
; CHECK-T2-NEXT:      movt [[ADDR0]], :upper16:x
; CHECK-T2-NEXT:      movt [[ADDR1]], :upper16:y
; CHECK-T2-NEXT:      ldrd [[R0:r[0-9]+]], [[R1:r[0-9]+]], {{\[}}[[ADDR0]], #1020]
; CHECK-T2-NEXT:      strd [[R0]], [[R1]], {{\[}}[[ADDR1]], #1020]
; CHECK-ARMV4T:       ldr [[ADDR0:r[0-9]+]]
; CHECK-ARMV4T-NEXT:  ldr [[ADDR1:r[0-9]+]]
; CHECK-ARMV4T-NEXT:  ldr [[R0:r[0-9]+]], {{\[}}[[ADDR0]], #1020]
; CHECK-ARMV4T-NEXT:  ldr [[R1:r[0-9]+]], {{\[}}[[ADDR0]], #1024]
; CHECK-ARMV4T-NEXT:  str [[R1]], {{\[}}[[ADDR1]], #1024]
; CHECK-ARMV4T-NEXT:  str [[R0]], {{\[}}[[ADDR1]], #1020]
entry:
  %0 = load volatile i64, i64* bitcast (i8* getelementptr (i8, i8* bitcast (i64* @x to i8*), i32 1020) to i64*), align 8
  store volatile i64 %0, i64* bitcast (i8* getelementptr (i8, i8* bitcast (i64* @y to i8*), i32 1020) to i64*), align 8
  ret void
}

define void @test_offset_4() {
; CHECK-LABEL: test_offset_4:
; CHECK-ARMV5TE:      ldr [[ADDR0:r[0-9]+]]
; CHECK-ARMV5TE:      ldr [[ADDR1:r[0-9]+]]
; CHECK-ARMV5TE-NEXT: add [[ADDR0]], [[ADDR0]], #1024
; CHECK-ARMV5TE-NEXT: add [[ADDR1]], [[ADDR1]], #1024
; CHECK-ARMV5TE-NEXT: ldrd [[R0:r[0-9]+]], [[R1:r[0-9]+]], {{\[}}[[ADDR0]]]
; CHECK-ARMV5TE-NEXT: strd [[R0]], [[R1]], {{\[}}[[ADDR1]]]
; CHECK-T2:           movw [[ADDR1:r[0-9]+]], :lower16:y
; CHECK-T2-NEXT:      movw [[ADDR0:r[0-9]+]], :lower16:x
; CHECK-T2-NEXT:      movt [[ADDR1]], :upper16:y
; CHECK-T2-NEXT:      movt [[ADDR0]], :upper16:x
; CHECK-T2-NEXT:      add.w [[ADDR0]], [[ADDR0]], #1024
; CHECK-T2-NEXT:      add.w [[ADDR1]], [[ADDR1]], #1024
; CHECK-T2-NEXT:      ldrd [[R0:r[0-9]+]], [[R1:r[0-9]+]], {{\[}}[[ADDR0]]]
; CHECK-T2-NEXT:      strd [[R0]], [[R1]], {{\[}}[[ADDR1]]]
; CHECK-ARMV4T:       ldr [[ADDR0:r[0-9]+]]
; CHECK-ARMV4T-NEXT:  ldr [[ADDR1:r[0-9]+]]
; CHECK-ARMV4T-NEXT:  ldr [[R0:r[0-9]+]], {{\[}}[[ADDR0]], #1024]
; CHECK-ARMV4T-NEXT:  ldr [[R1:r[0-9]+]], {{\[}}[[ADDR0]], #1028]
; CHECK-ARMV4T-NEXT:  str [[R1]], {{\[}}[[ADDR1]], #1028]
; CHECK-ARMV4T-NEXT:  str [[R0]], {{\[}}[[ADDR1]], #1024]
entry:
  %0 = load volatile i64, i64* bitcast (i8* getelementptr (i8, i8* bitcast (i64* @x to i8*), i32 1024) to i64*), align 8
  store volatile i64 %0, i64* bitcast (i8* getelementptr (i8, i8* bitcast (i64* @y to i8*), i32 1024) to i64*), align 8
  ret void
}

define i64 @test_stack() {
; CHECK-LABEL: test_stack:
; CHECK-ARMV5TE:      sub sp, sp, #80
; CHECK-ARMV5TE-NEXT: mov [[R0:r[0-9]+]], #0
; CHECK-ARMV5TE-NEXT: mov [[R1:r[0-9]+]], #1
; CHECK-ARMV5TE-NEXT: strd [[R1]], [[R0]], [sp, #8]
; CHECK-ARMV5TE-NEXT: ldrd r0, r1, [sp, #8]
; CHECK-ARMV5TE-NEXT: add sp, sp, #80
; CHECK-ARMV5TE-NEXT: bx lr
; CHECK-T2:      sub sp, #80
; CHECK-T2-NEXT: movs [[R0:r[0-9]+]], #0
; CHECK-T2-NEXT: movs [[R1:r[0-9]+]], #1
; CHECK-T2-NEXT: strd [[R1]], [[R0]], [sp, #8]
; CHECK-T2-NEXT: ldrd r0, r1, [sp, #8]
; CHECK-T2-NEXT: add sp, #80
; CHECK-T2-NEXT: bx lr
; CHECK-ARMV4T:      sub sp, sp, #80
; CHECK-ARMV4T-NEXT: mov [[R0:r[0-9]+]], #0
; CHECK-ARMV4T-NEXT: str [[R0]], [sp, #12]
; CHECK-ARMV4T-NEXT: mov [[R1:r[0-9]+]], #1
; CHECK-ARMV4T-NEXT: str [[R1]], [sp, #8]
; CHECK-ARMV4T-NEXT: ldr r0, [sp, #8]
; CHECK-ARMV4T-NEXT: ldr r1, [sp, #12]
; CHECK-ARMV4T-NEXT: add sp, sp, #80
; CHECK-ARMV4T-NEXT: bx lr
entry:
  %a = alloca [10 x i64], align 8
  %arrayidx = getelementptr inbounds [10 x i64], [10 x i64]* %a, i32 0, i32 1
  store volatile i64 1, i64* %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds [10 x i64], [10 x i64]* %a, i32 0, i32 1
  %0 = load volatile i64, i64* %arrayidx1, align 8
  ret i64 %0
}

