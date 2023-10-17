; RUN: llc < %s -march=avr -mattr=avr6 | FileCheck %s

declare i16 @allocate(i16*, i16*)

; Test taking an address of an alloca with a small offset (adiw)
define i16 @alloca_addressof_small() {
entry:
; CHECK-LABEL: alloca_addressof_small:
; Test that Y is saved
; CHECK: push r28
; CHECK: push r29
; CHECK: movw r24, r28
; CHECK: adiw r24, 17
; CHECK: movw {{.*}}, r28
; CHECK: adiw {{.*}}, 39
; CHECK: movw r22, {{.*}}
; CHECK: pop r29
; CHECK: pop r28
  %p = alloca [18 x i16]
  %k = alloca [14 x i16]
  %arrayidx = getelementptr inbounds [14 x i16], [14 x i16]* %k, i16 0, i16 8
  %arrayidx1 = getelementptr inbounds [18 x i16], [18 x i16]* %p, i16 0, i16 5
  %call = call i16 @allocate(i16* %arrayidx, i16* %arrayidx1)
  ret i16 %call
}

; Test taking an address of an alloca with a big offset (subi/sbci pair)
define i16 @alloca_addressof_big() {
entry:
; CHECK-LABEL: alloca_addressof_big:
; CHECK: movw r24, r28
; CHECK: adiw r24, 17
; CHECK: movw r22, r28
; CHECK: subi r22, 145
; CHECK: sbci r23, 255
  %p = alloca [55 x i16]
  %k = alloca [14 x i16]
  %arrayidx = getelementptr inbounds [14 x i16], [14 x i16]* %k, i16 0, i16 8
  %arrayidx1 = getelementptr inbounds [55 x i16], [55 x i16]* %p, i16 0, i16 41
  %call = call i16 @allocate(i16* %arrayidx, i16* %arrayidx1)
  ret i16 %call
}

; Test writing to an allocated variable with a small and a big offset
define i16 @alloca_write(i16 %x) {
entry:
; CHECK-LABEL: alloca_write:
; Small offset here
; CHECK: std Y+24, {{.*}}
; CHECK: std Y+23, {{.*}}
; Big offset here
; CHECK: adiw r28, 57
; CHECK: std Y+63, {{.*}}
; CHECK: std Y+62, {{.*}}
; CHECK: sbiw r28, 57
  %p = alloca [15 x i16]
  %k = alloca [14 x i16]
  %arrayidx = getelementptr inbounds [15 x i16], [15 x i16]* %p, i16 0, i16 45
  store i16 22, i16* %arrayidx
  %arrayidx1 = getelementptr inbounds [14 x i16], [14 x i16]* %k, i16 0, i16 11
  store i16 42, i16* %arrayidx1
  %arrayidx2 = getelementptr inbounds [14 x i16], [14 x i16]* %k, i16 0, i16 0
  %arrayidx3 = getelementptr inbounds [15 x i16], [15 x i16]* %p, i16 0, i16 0
  %call = call i16 @allocate(i16* %arrayidx2, i16* %arrayidx3)
  ret i16 %call
}

; Test writing to an allocated variable with a huge offset that cant be
; materialized with adiw/sbiw but with a subi/sbci pair.
define void @alloca_write_huge() {
; CHECK-LABEL: alloca_write_huge:
; CHECK: subi r28, 41
; CHECK: sbci r29, 255
; CHECK: std Y+63, {{.*}}
; CHECK: std Y+62, {{.*}}
; CHECK: subi r28, 215
; CHECK: sbci r29, 0
  %k = alloca [140 x i16]
  %arrayidx = getelementptr inbounds [140 x i16], [140 x i16]* %k, i16 0, i16 138
  store i16 22, i16* %arrayidx
  %arraydecay = getelementptr inbounds [140 x i16], [140 x i16]* %k, i16 0, i16 0
  call i16 @allocate(i16* %arraydecay, i16* null)
  ret void
}
