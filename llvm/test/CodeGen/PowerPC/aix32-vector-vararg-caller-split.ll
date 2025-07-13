; RUN: llc -verify-machineinstrs -stop-before=ppc-vsx-copy -vec-extabi \
; RUN:     -mcpu=pwr7  -mtriple powerpc-ibm-aix-xcoff < %s 2>&1 | \
; RUN: FileCheck %s

define void @caller() {
entry:
  %call = tail call <4 x i32> (double, double, double, ...) @split_spill(double 0.000000e+00, double 1.000000e+00, double 2.000000e+00, <4 x i32> <i32 1, i32 2, i32 3, i32 4>)
  ret void
}

declare <4 x i32> @split_spill(double, double, double, ...)

; CHECK:     ADJCALLSTACKDOWN 64, 0, implicit-def dead $r1, implicit $r1
; CHECK:     [[VECCONSTADDR:%[0-9]+]]:gprc = LWZtoc %const.0, $r2 :: (load (s32) from got)
; CHECK:     [[VECCONST:%[0-9]+]]:vsrc = LXVW4X $zero, killed [[VECCONSTADDR]] :: (load (s128) from constant-pool)
; CHECK:     [[STACKOFFSET:%[0-9]+]]:gprc = LI 48
; CHECK:     STXVW4X killed [[VECCONST]], $r1, killed [[STACKOFFSET]] :: (store (s128))
; CHECK-DAG: [[ELEMENT1:%[0-9]+]]:gprc = LWZ 48, $r1 :: (load (s32))
; CHECK-DAG: [[ELEMENT2:%[0-9]+]]:gprc = LWZ 52, $r1 :: (load (s32))
; CHECK:     [[FLOAT1SPLAT:%[0-9]+]]:vrrc = VSPLTISW 1
; CHECK:     [[FLOAT1COPY:%[0-9]+]]:vsrc = COPY killed [[FLOAT1SPLAT]]
; CHECK:     [[DOUBLE1:%[0-9]+]]:vsrc = XVCVSXWDP killed [[FLOAT1COPY]], implicit $rm
; CHECK:     [[DOUBLE1COPY:%[0-9]+]]:vsfrc = COPY killed [[DOUBLE1]]
; CHECK:     [[FLOAT2SPLAT:%[0-9]+]]:vrrc = VSPLTISW 2
; CHECK:     [[FLOAT2COPY:%[0-9]+]]:vsrc = COPY killed [[FLOAT2SPLAT]]
; CHECK:     [[DOUBLE2:%[0-9]+]]:vsrc = XVCVSXWDP killed [[FLOAT2COPY]], implicit $rm
; CHECK:     [[DOUBLE2COPY:%[0-9]+]]:vsfrc = COPY killed [[DOUBLE2]]

; CHECK:     [[DZERO:%[0-9]+]]:vsfrc = XXLXORdpz
; CHECK:     [[DTOI1:%[0-9]+]]:gprc = LIS 16368
; CHECK:     [[DTOI2:%[0-9]+]]:gprc = LIS 16384
; CHECK:     [[IZERO:%[0-9]+]]:gprc = LI 0

; CHECK-DAG: $f1 = COPY [[DZERO]]
; CHECK-DAG: $r3 = COPY [[IZERO]]
; CHECK-DAG: $r4 = COPY [[IZERO]]

; CHECK-DAG: $f2 = COPY [[DOUBLE1COPY]]
; CHECK-DAG: $r5 = COPY [[DTOI1]]
; CHECK-DAG: $r6 = COPY [[IZERO]]

; CHECK-DAG: $f3 = COPY [[DOUBLE2COPY]]
; CHECK-DAG: $r7 = COPY [[DTOI2]]
; CHECK-DAG: $r8 = COPY [[IZERO]]

; CHECK-DAG: $r9 = COPY [[ELEMENT1]]
; CHECK-DAG: $r10 = COPY [[ELEMENT2]]

; CHECK:    BL_NOP <mcsymbol .split_spill[PR]>, csr_aix32_altivec, implicit-def dead $lr, implicit $rm, implicit $f1, implicit $r3, implicit $r4, implicit $f2, implicit $r5, implicit $r6, implicit $f3, implicit $r7, implicit $r8, implicit $r9, implicit $r10, implicit $r2, implicit-def $r1, implicit-def $v2
