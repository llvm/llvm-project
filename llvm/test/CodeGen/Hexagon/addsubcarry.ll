; RUN: llc -mtriple=hexagon < %s | FileCheck %s

@g = global i128 zeroinitializer, align 8

; CHECK-LABEL: addc:
; CHECK: p[[P0:[0-3]]] = and(p[[P1:[0-9]]],!p[[P1]])
; CHECK: add({{.*}},{{.*}},p[[P0]]):carry
; CHECK: add({{.*}},{{.*}},p[[P0]]):carry
define void @addc(i128 %a0, i128 %a1) #0 {
  %v0 = add i128 %a0, %a1
  store i128 %v0, ptr @g, align 8
  ret void
}

; CHECK-LABEL: subc:
; CHECK: p[[P0:[0-3]]] = or(p[[P1:[0-9]]],!p[[P1]])
; CHECK: sub({{.*}},{{.*}},p[[P0]]):carry
; CHECK: sub({{.*}},{{.*}},p[[P0]]):carry
define void @subc(i128 %a0, i128 %a1) #0 {
  %v0 = sub i128 %a0, %a1
  store i128 %v0, ptr @g, align 8
  ret void
}


