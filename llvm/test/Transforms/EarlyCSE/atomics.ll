; RUN: opt < %s -S -early-cse | FileCheck %s

; CHECK-LABEL: @test12(
define i32 @test12(i1 %B, i32* %P1, i32* %P2) {
  %load0 = load i32, i32* %P1
  %1 = load atomic i32, i32* %P2 seq_cst, align 4
  %load1 = load i32, i32* %P1
  %sel = select i1 %B, i32 %load0, i32 %load1
  ret i32 %sel
  ; CHECK: load i32, i32* %P1
  ; CHECK: load i32, i32* %P1
}

; CHECK-LABEL: @test13(
; atomic to non-atomic forwarding is legal
define i32 @test13(i1 %B, i32* %P1) {
  %a = load atomic i32, i32* %P1 seq_cst, align 4
  %b = load i32, i32* %P1
  %res = sub i32 %a, %b
  ret i32 %res
  ; CHECK: load atomic i32, i32* %P1
  ; CHECK: ret i32 0
}

; CHECK-LABEL: @test14(
; atomic to unordered atomic forwarding is legal
define i32 @test14(i1 %B, i32* %P1) {
  %a = load atomic i32, i32* %P1 seq_cst, align 4
  %b = load atomic i32, i32* %P1 unordered, align 4
  %res = sub i32 %a, %b
  ret i32 %res
  ; CHECK: load atomic i32, i32* %P1 seq_cst
  ; CHECK-NEXT: ret i32 0
}

; CHECK-LABEL: @test15(
; implementation restriction: can't forward to stonger
; than unordered
define i32 @test15(i1 %B, i32* %P1, i32* %P2) {
  %a = load atomic i32, i32* %P1 seq_cst, align 4
  %b = load atomic i32, i32* %P1 seq_cst, align 4
  %res = sub i32 %a, %b
  ret i32 %res
  ; CHECK: load atomic i32, i32* %P1
  ; CHECK: load atomic i32, i32* %P1
}

; CHECK-LABEL: @test16(
; forwarding non-atomic to atomic is wrong! (However,
; it would be legal to use the later value in place of the
; former in this particular example.  We just don't
; do that right now.)
define i32 @test16(i1 %B, i32* %P1, i32* %P2) {
  %a = load i32, i32* %P1, align 4
  %b = load atomic i32, i32* %P1 unordered, align 4
  %res = sub i32 %a, %b
  ret i32 %res
  ; CHECK: load i32, i32* %P1
  ; CHECK: load atomic i32, i32* %P1
}

; Can't DSE across a full fence
define void @fence_seq_cst_store(i1 %B, i32* %P1, i32* %P2) {
; CHECK-LABEL: @fence_seq_cst_store
; CHECK: store
; CHECK: store atomic
; CHECK: store
  store i32 0, i32* %P1, align 4
  store atomic i32 0, i32* %P2 seq_cst, align 4
  store i32 0, i32* %P1, align 4
  ret void
}

; Can't DSE across a full fence
define void @fence_seq_cst(i1 %B, i32* %P1, i32* %P2) {
; CHECK-LABEL: @fence_seq_cst
; CHECK: store
; CHECK: fence seq_cst
; CHECK: store
  store i32 0, i32* %P1, align 4
  fence seq_cst
  store i32 0, i32* %P1, align 4
  ret void
}

; Can't DSE across a full fence
define void @fence_asm_sideeffect(i1 %B, i32* %P1, i32* %P2) {
; CHECK-LABEL: @fence_asm_sideeffect
; CHECK: store
; CHECK: call void asm sideeffect
; CHECK: store
  store i32 0, i32* %P1, align 4
  call void asm sideeffect "", ""()
  store i32 0, i32* %P1, align 4
  ret void
}

; Can't DSE across a full fence
define void @fence_asm_memory(i1 %B, i32* %P1, i32* %P2) {
; CHECK-LABEL: @fence_asm_memory
; CHECK: store
; CHECK: call void asm
; CHECK: store
  store i32 0, i32* %P1, align 4
  call void asm "", "~{memory}"()
  store i32 0, i32* %P1, align 4
  ret void
}

; Can't remove a volatile load
define i32 @volatile_load(i1 %B, i32* %P1, i32* %P2) {
  %a = load i32, i32* %P1, align 4
  %b = load volatile i32, i32* %P1, align 4
  %res = sub i32 %a, %b
  ret i32 %res
  ; CHECK-LABEL: @volatile_load
  ; CHECK: load i32, i32* %P1
  ; CHECK: load volatile i32, i32* %P1
}

; Can't remove redundant volatile loads
define i32 @redundant_volatile_load(i1 %B, i32* %P1, i32* %P2) {
  %a = load volatile i32, i32* %P1, align 4
  %b = load volatile i32, i32* %P1, align 4
  %res = sub i32 %a, %b
  ret i32 %res
  ; CHECK-LABEL: @redundant_volatile_load
  ; CHECK: load volatile i32, i32* %P1
  ; CHECK: load volatile i32, i32* %P1
  ; CHECK: sub
}

; Can't DSE a volatile store
define void @volatile_store(i1 %B, i32* %P1, i32* %P2) {
; CHECK-LABEL: @volatile_store
; CHECK: store volatile
; CHECK: store
  store volatile i32 0, i32* %P1, align 4
  store i32 3, i32* %P1, align 4
  ret void
}

; Can't DSE a redundant volatile store
define void @redundant_volatile_store(i1 %B, i32* %P1, i32* %P2) {
; CHECK-LABEL: @redundant_volatile_store
; CHECK: store volatile
; CHECK: store volatile
  store volatile i32 0, i32* %P1, align 4
  store volatile i32 0, i32* %P1, align 4
  ret void
}

; Can value forward from volatiles
define i32 @test20(i1 %B, i32* %P1, i32* %P2) {
  %a = load volatile i32, i32* %P1, align 4
  %b = load i32, i32* %P1, align 4
  %res = sub i32 %a, %b
  ret i32 %res
  ; CHECK-LABEL: @test20
  ; CHECK: load volatile i32, i32* %P1
  ; CHECK: ret i32 0
}

; Can DSE a non-volatile store in favor of a volatile one
; currently a missed optimization
define void @test21(i1 %B, i32* %P1, i32* %P2) {
; CHECK-LABEL: @test21
; CHECK: store 
; CHECK: store volatile
  store i32 0, i32* %P1, align 4
  store volatile i32 3, i32* %P1, align 4
  ret void
}

; Can DSE a normal store in favor of a unordered one
define void @test22(i1 %B, i32* %P1, i32* %P2) {
; CHECK-LABEL: @test22
; CHECK-NEXT: store atomic
  store i32 0, i32* %P1, align 4
  store atomic i32 3, i32* %P1 unordered, align 4
  ret void
}



