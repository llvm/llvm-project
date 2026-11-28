; Check that formal arguments and return values of HVX vector type,
; HLX vector type and scalar types can be lowered correctly when
; compiling with both HVX and HLX enabled.

; RUN: llc -march=hexagon -mcpu=hexagonv85 -mattr=+hlx,+hvx,+hvx-length128b \
; RUN: -stop-after=hexagon-isel -verify-machineinstrs -o - %s | FileCheck %s

target triple = "hexagon"

; Generated from:
; HVX_Vector hvx_cc(HVX_Vector a, HVX_Vector b) {
;   return a + b;
; }
;
; CHECK-LABEL: name: hvx_cc
; CHECK-DAG: %{{[0-9]+}}:hvxvr = COPY $v0
; CHECK-DAG: %{{[0-9]+}}:hvxvr = COPY $v1
; CHECK: $v0 = COPY %{{[0-9]+}}
define dso_local inreg <32 x i32> @hvx_cc(<32 x i32> %a, <32 x i32> %b) local_unnamed_addr {
entry:
  %add = add <32 x i32> %b, %a
  ret <32 x i32> %add
}

; Generated from:
; HLX_Vector hlx_cc(HLX_Vector a, HLX_Vector b) {
;   return a + b;
; }
;
; CHECK-LABEL: name: hlx_cc
; CHECK-DAG: %{{[0-9]+}}:hlxlr = COPY $l0
; CHECK-DAG: %{{[0-9]+}}:hlxlr = COPY $l1
; CHECK: $l0 = COPY %{{[0-9]+}}
define dso_local inreg <1024 x i16> @hlx_cc(<1024 x i16> %a, <1024 x i16> %b) local_unnamed_addr {
entry:
  %add = add <1024 x i16> %b, %a
  ret <1024 x i16> %add
}

; CHECK-LABEL: name: hlx_pair_cc
; CHECK-DAG: %{{[0-9]+}}:hlxllr = COPY $ll0
; CHECK-DAG: %{{[0-9]+}}:hlxllr = COPY $ll1
; CHECK: $ll0 = COPY %{{[0-9]+}}
define dso_local inreg <1024 x i32> @hlx_pair_cc(<1024 x i32> %a, <1024 x i32> %b) local_unnamed_addr {
entry:
  %add = add <1024 x i32> %b, %a
  ret <1024 x i32> %add
}

; Results of calls returning HLX types are read from $l0 and $ll0.
; CHECK-LABEL: name: hlx_call_result
; CHECK: J2_call @hlx_cc, {{.*}}, implicit-def $l0
; CHECK: %{{[0-9]+}}:hlxlr = COPY $l0
define dso_local inreg <1024 x i16> @hlx_call_result(<1024 x i16> %a) local_unnamed_addr {
entry:
  %r = call inreg <1024 x i16> @hlx_cc(<1024 x i16> %a, <1024 x i16> %a)
  ret <1024 x i16> %r
}

; CHECK-LABEL: name: hlx_pair_call_result
; CHECK: J2_call @hlx_pair_cc, {{.*}}, implicit-def $ll0
; CHECK: %{{[0-9]+}}:hlxllr = COPY $ll0
define dso_local inreg <1024 x i32> @hlx_pair_call_result(<1024 x i32> %a) local_unnamed_addr {
entry:
  %r = call inreg <1024 x i32> @hlx_pair_cc(<1024 x i32> %a, <1024 x i32> %a)
  ret <1024 x i32> %r
}

; Generated from:
; int scalar_cc(int a, int b) {
;   return a + b;
; }
;
; CHECK-LABEL: name: scalar_cc
; CHECK-DAG: %{{[0-9]+}}:intregs = COPY $r0
; CHECK-DAG: %{{[0-9]+}}:intregs = COPY $r1
; CHECK: $r0 = COPY %{{[0-9]+}}
define dso_local i32 @scalar_cc(i32 %a, i32 %b) local_unnamed_addr {
entry:
  %add = add nsw i32 %b, %a
  ret i32 %add
}

; Generated from:
; HVX_Vector mix_cc(int a, HVX_Vector b, HLX_Vector c) {
;   int d = a + b[2] + c[4];
;   return Q6_V_vsplat_R(d);
; }
;
; CHECK-LABEL: name: mix_cc
; CHECK-DAG: %{{[0-9]+}}:hlxlr = COPY $l0
; CHECK-DAG: %{{[0-9]+}}:hvxvr = COPY $v0
; CHECK-DAG: %{{[0-9]+}}:intregs = COPY $r0
; CHECK: %[[HVXREG:[0-9]+]]:hvxvr = V6_lvsplatw %{{[0-9]+}}
; CHECK: $v0 = COPY %[[HVXREG]]
define dso_local inreg <32 x i32> @mix_cc(i32 %a, <32 x i32> %b, <1024 x i16> %c) local_unnamed_addr {
entry:
  %vecext = extractelement <32 x i32> %b, i64 2
  %add = add nsw i32 %vecext, %a
  %vecext1 = extractelement <1024 x i16> %c, i64 4
  %conv = sext i16 %vecext1 to i32
  %add2 = add nsw i32 %add, %conv
  %0 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 %add2)
  ret <32 x i32> %0
}
