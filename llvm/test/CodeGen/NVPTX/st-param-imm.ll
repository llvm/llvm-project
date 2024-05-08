; RUN: llc < %s -march=nvptx64 | FileCheck %s
; RUN: llc < %s -march=nvptx | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -verify-machineinstrs | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -verify-machineinstrs | %ptxas-verify %}

%struct.A = type { i8, i16 }
%struct.char2 = type { i8, i8 }
%struct.char4 = type { i8, i8, i8, i8 }
%struct.short2 = type { i16, i16 }
%struct.short4 = type { i16, i16, i16, i16 }
%struct.int2 = type { i32, i32 }
%struct.int4 = type { i32, i32, i32, i32 }
%struct.longlong2 = type { i64, i64 }
%struct.float2 = type { float, float }
%struct.float4 = type { float, float, float, float }
%struct.double2 = type { double, double }

; CHECK-LABEL: st_param_i8_i16
; CHECK: st.param.b8 [param0+0], 1
; CHECK: st.param.b16 [param0+2], 2
define void @st_param_i8_i16() {
  call void @call_i8_i16(%struct.A { i8 1, i16 2 })
  ret void
}

; CHECK-LABEL: st_param_i32
; CHECK: st.param.b32 [param0+0], 3
define void @st_param_i32() {
  call void @call_i32(i32 3)
  ret void
}

; CHECK-LABEL: st_param_i64
; CHECK: st.param.b64 [param0+0], 4
define void @st_param_i64() {
  call void @call_i64(i64 4)
  ret void
}

; CHECK-LABEL: st_param_f32
; CHECK: st.param.f32 [param0+0], 0f40A00000
define void @st_param_f32() {
  call void @call_f32(float 5.0)
  ret void
}

; CHECK-LABEL: st_param_f64
; CHECK: st.param.f64 [param0+0], 0d4018000000000000
define void @st_param_f64() {
  call void @call_f64(double 6.0)
  ret void
}

declare void @call_i8_i16(%struct.A)
declare void @call_i32(i32)
declare void @call_i64(i64)
declare void @call_f32(float)
declare void @call_f64(double)

; CHECK-LABEL: st_param_v2_i8
; CHECK: st.param.v2.b8 [param0+0], {1, 2}
; CHECK: st.param.v2.b8 [param0+0], {1, {{%rs[0-9]+}}}
; CHECK: st.param.v2.b8 [param0+0], {{{%rs[0-9]+}}, 2}
define void @st_param_v2_i8(i8 %val) {
  call void @call_v2_i8(%struct.char2 { i8 1, i8 2 })
  %struct.ir0 = insertvalue %struct.char2 poison, i8 1, 0
  %struct.ir1 = insertvalue %struct.char2 %struct.ir0, i8 %val, 1
  call void @call_v2_i8(%struct.char2 %struct.ir1)
  %struct.ri0 = insertvalue %struct.char2 poison, i8 %val, 0
  %struct.ri1 = insertvalue %struct.char2 %struct.ri0, i8 2, 1
  call void @call_v2_i8(%struct.char2 %struct.ri1)
  ret void
}

; CHECK-LABEL: st_param_v2_i16
; CHECK: st.param.v2.b16 [param0+0], {1, 2}
; CHECK: st.param.v2.b16 [param0+0], {1, {{%rs[0-9]+}}}
; CHECK: st.param.v2.b16 [param0+0], {{{%rs[0-9]+}}, 2}
define void @st_param_v2_i16(i16 %val) {
  call void @call_v2_i16(%struct.short2 { i16 1, i16 2 })
  %struct.ir0 = insertvalue %struct.short2 poison, i16 1, 0
  %struct.ir1 = insertvalue %struct.short2 %struct.ir0, i16 %val, 1
  call void @call_v2_i16(%struct.short2 %struct.ir1)
  %struct.ri0 = insertvalue %struct.short2 poison, i16 %val, 0
  %struct.ri1 = insertvalue %struct.short2 %struct.ri0, i16 2, 1
  call void @call_v2_i16(%struct.short2 %struct.ri1)
  ret void
}

; CHECK-LABEL: st_param_v2_i32
; CHECK: st.param.v2.b32 [param0+0], {1, 2}
; CHECK: st.param.v2.b32 [param0+0], {1, {{%r[0-9]+}}}
; CHECK: st.param.v2.b32 [param0+0], {{{%r[0-9]+}}, 2}
define void @st_param_v2_i32(i32 %val) {
  call void @call_v2_i32(%struct.int2 { i32 1, i32 2 })
  %struct.ir0 = insertvalue %struct.int2 poison, i32 1, 0
  %struct.ir1 = insertvalue %struct.int2 %struct.ir0, i32 %val, 1
  call void @call_v2_i32(%struct.int2 %struct.ir1)
  %struct.ri0 = insertvalue %struct.int2 poison, i32 %val, 0
  %struct.ri1 = insertvalue %struct.int2 %struct.ri0, i32 2, 1
  call void @call_v2_i32(%struct.int2 %struct.ri1)
  ret void
}

; CHECK-LABEL: st_param_v2_i64
; CHECK: st.param.v2.b64 [param0+0], {1, 2}
; CHECK: st.param.v2.b64 [param0+0], {1, {{%rd[0-9]+}}}
; CHECK: st.param.v2.b64 [param0+0], {{{%rd[0-9]+}}, 2}
define void @st_param_v2_i64(i64 %val) {
  call void @call_v2_i64(%struct.longlong2 { i64 1, i64 2 })
  %struct.ir0 = insertvalue %struct.longlong2 poison, i64 1, 0
  %struct.ir1 = insertvalue %struct.longlong2 %struct.ir0, i64 %val, 1
  call void @call_v2_i64(%struct.longlong2 %struct.ir1)
  %struct.ri0 = insertvalue %struct.longlong2 poison, i64 %val, 0
  %struct.ri1 = insertvalue %struct.longlong2 %struct.ri0, i64 2, 1
  call void @call_v2_i64(%struct.longlong2 %struct.ri1)
  ret void
}

; CHECK-LABEL: st_param_v2_f32
; CHECK: st.param.v2.f32 [param0+0], {0f3F800000, 0f40000000}
; CHECK: st.param.v2.f32 [param0+0], {0f3F800000, {{%f[0-9]+}}}
; CHECK: st.param.v2.f32 [param0+0], {{{%f[0-9]+}}, 0f40000000}
define void @st_param_v2_f32(float %val) {
  call void @call_v2_f32(%struct.float2 { float 1.0, float 2.0 })
  %struct.ir0 = insertvalue %struct.float2 poison, float 1.0, 0
  %struct.ir1 = insertvalue %struct.float2 %struct.ir0, float %val, 1
  call void @call_v2_f32(%struct.float2 %struct.ir1)
  %struct.ri0 = insertvalue %struct.float2 poison, float %val, 0
  %struct.ri1 = insertvalue %struct.float2 %struct.ri0, float 2.0, 1
  call void @call_v2_f32(%struct.float2 %struct.ri1)
  ret void
}

; CHECK-LABEL: st_param_v2_f64
; CHECK: st.param.v2.f64 [param0+0], {0d3FF0000000000000, 0d4000000000000000}
; CHECK: st.param.v2.f64 [param0+0], {0d3FF0000000000000, {{%fd[0-9]+}}}
; CHECK: st.param.v2.f64 [param0+0], {{{%fd[0-9]+}}, 0d4000000000000000}
define void @st_param_v2_f64(double %val) {
  call void @call_v2_f64(%struct.double2 { double 1.0, double 2.0 })
  %struct.ir0 = insertvalue %struct.double2 poison, double 1.0, 0
  %struct.ir1 = insertvalue %struct.double2 %struct.ir0, double %val, 1
  call void @call_v2_f64(%struct.double2 %struct.ir1)
  %struct.ri0 = insertvalue %struct.double2 poison, double %val, 0
  %struct.ri1 = insertvalue %struct.double2 %struct.ri0, double 2.0, 1
  call void @call_v2_f64(%struct.double2 %struct.ri1)
  ret void
}

declare void @call_v2_i8(%struct.char2)
declare void @call_v2_i16(%struct.short2)
declare void @call_v2_i32(%struct.int2)
declare void @call_v2_i64(%struct.longlong2)
declare void @call_v2_f32(%struct.float2)
declare void @call_v2_f64(%struct.double2)

; CHECK-LABEL: st_param_v4_i8
; CHECK: st.param.v4.b8 [param0+0], {1, 2, 3, 4}
; CHECK: st.param.v4.b8 [param0+0], {1, {{%rs[0-9]+}}, {{%rs[0-9]+}}, {{%rs[0-9]+}}}
; CHECK: st.param.v4.b8 [param0+0], {{{%rs[0-9]+}}, 2, {{%rs[0-9]+}}, {{%rs[0-9]+}}}
; CHECK: st.param.v4.b8 [param0+0], {{{%rs[0-9]+}}, {{%rs[0-9]+}}, 3, {{%rs[0-9]+}}}
; CHECK: st.param.v4.b8 [param0+0], {{{%rs[0-9]+}}, {{%rs[0-9]+}}, {{%rs[0-9]+}}, 4}
; CHECK: st.param.v4.b8 [param0+0], {1, 2, {{%rs[0-9]+}}, {{%rs[0-9]+}}}
; CHECK: st.param.v4.b8 [param0+0], {1, {{%rs[0-9]+}}, 3, {{%rs[0-9]+}}}
; CHECK: st.param.v4.b8 [param0+0], {1, {{%rs[0-9]+}}, {{%rs[0-9]+}}, 4}
; CHECK: st.param.v4.b8 [param0+0], {{{%rs[0-9]+}}, 2, 3, {{%rs[0-9]+}}}
; CHECK: st.param.v4.b8 [param0+0], {{{%rs[0-9]+}}, 2, {{%rs[0-9]+}}, 4}
; CHECK: st.param.v4.b8 [param0+0], {{{%rs[0-9]+}}, {{%rs[0-9]+}}, 3, 4}
; CHECK: st.param.v4.b8 [param0+0], {1, 2, 3, {{%rs[0-9]+}}}
; CHECK: st.param.v4.b8 [param0+0], {1, 2, {{%rs[0-9]+}}, 4}
; CHECK: st.param.v4.b8 [param0+0], {1, {{%rs[0-9]+}}, 3, 4}
; CHECK: st.param.v4.b8 [param0+0], {{{%rs[0-9]+}}, 2, 3, 4}
define void @st_param_v4_i8(i8 %a, i8 %b, i8 %c, i8 %d) {
  call void @call_v4_i8(%struct.char4 { i8 1, i8 2, i8 3, i8 4 })

  %struct.irrr0 = insertvalue %struct.char4 poison, i8 1, 0
  %struct.irrr1 = insertvalue %struct.char4 %struct.irrr0, i8 %b, 1
  %struct.irrr2 = insertvalue %struct.char4 %struct.irrr1, i8 %c, 2
  %struct.irrr3 = insertvalue %struct.char4 %struct.irrr2, i8 %d, 3
  call void @call_v4_i8(%struct.char4 %struct.irrr3)

  %struct.rirr0 = insertvalue %struct.char4 poison, i8 %a, 0
  %struct.rirr1 = insertvalue %struct.char4 %struct.rirr0, i8 2, 1
  %struct.rirr2 = insertvalue %struct.char4 %struct.rirr1, i8 %c, 2
  %struct.rirr3 = insertvalue %struct.char4 %struct.rirr2, i8 %d, 3
  call void @call_v4_i8(%struct.char4 %struct.rirr3)

  %struct.rrir0 = insertvalue %struct.char4 poison, i8 %a, 0
  %struct.rrir1 = insertvalue %struct.char4 %struct.rrir0, i8 %b, 1
  %struct.rrir2 = insertvalue %struct.char4 %struct.rrir1, i8 3, 2
  %struct.rrir3 = insertvalue %struct.char4 %struct.rrir2, i8 %d, 3
  call void @call_v4_i8(%struct.char4 %struct.rrir3)

  %struct.rrri0 = insertvalue %struct.char4 poison, i8 %a, 0
  %struct.rrri1 = insertvalue %struct.char4 %struct.rrri0, i8 %b, 1
  %struct.rrri2 = insertvalue %struct.char4 %struct.rrri1, i8 %c, 2
  %struct.rrri3 = insertvalue %struct.char4 %struct.rrri2, i8 4, 3
  call void @call_v4_i8(%struct.char4 %struct.rrri3)

  %struct.iirr0 = insertvalue %struct.char4 poison, i8 1, 0
  %struct.iirr1 = insertvalue %struct.char4 %struct.iirr0, i8 2, 1
  %struct.iirr2 = insertvalue %struct.char4 %struct.iirr1, i8 %c, 2
  %struct.iirr3 = insertvalue %struct.char4 %struct.iirr2, i8 %d, 3
  call void @call_v4_i8(%struct.char4 %struct.iirr3)

  %struct.irir0 = insertvalue %struct.char4 poison, i8 1, 0
  %struct.irir1 = insertvalue %struct.char4 %struct.irir0, i8 %b, 1
  %struct.irir2 = insertvalue %struct.char4 %struct.irir1, i8 3, 2
  %struct.irir3 = insertvalue %struct.char4 %struct.irir2, i8 %d, 3
  call void @call_v4_i8(%struct.char4 %struct.irir3)

  %struct.irri0 = insertvalue %struct.char4 poison, i8 1, 0
  %struct.irri1 = insertvalue %struct.char4 %struct.irri0, i8 %b, 1
  %struct.irri2 = insertvalue %struct.char4 %struct.irri1, i8 %c, 2
  %struct.irri3 = insertvalue %struct.char4 %struct.irri2, i8 4, 3
  call void @call_v4_i8(%struct.char4 %struct.irri3)

  %struct.riir0 = insertvalue %struct.char4 poison, i8 %a, 0
  %struct.riir1 = insertvalue %struct.char4 %struct.riir0, i8 2, 1
  %struct.riir2 = insertvalue %struct.char4 %struct.riir1, i8 3, 2
  %struct.riir3 = insertvalue %struct.char4 %struct.riir2, i8 %d, 3
  call void @call_v4_i8(%struct.char4 %struct.riir3)

  %struct.riri0 = insertvalue %struct.char4 poison, i8 %a, 0
  %struct.riri1 = insertvalue %struct.char4 %struct.riri0, i8 2, 1
  %struct.riri2 = insertvalue %struct.char4 %struct.riri1, i8 %c, 2
  %struct.riri3 = insertvalue %struct.char4 %struct.riri2, i8 4, 3
  call void @call_v4_i8(%struct.char4 %struct.riri3)

  %struct.rrii0 = insertvalue %struct.char4 poison, i8 %a, 0
  %struct.rrii1 = insertvalue %struct.char4 %struct.rrii0, i8 %b, 1
  %struct.rrii2 = insertvalue %struct.char4 %struct.rrii1, i8 3, 2
  %struct.rrii3 = insertvalue %struct.char4 %struct.rrii2, i8 4, 3
  call void @call_v4_i8(%struct.char4 %struct.rrii3)

  %struct.iiir0 = insertvalue %struct.char4 poison, i8 1, 0
  %struct.iiir1 = insertvalue %struct.char4 %struct.iiir0, i8 2, 1
  %struct.iiir2 = insertvalue %struct.char4 %struct.iiir1, i8 3, 2
  %struct.iiir3 = insertvalue %struct.char4 %struct.iiir2, i8 %d, 3
  call void @call_v4_i8(%struct.char4 %struct.iiir3)

  %struct.iiri0 = insertvalue %struct.char4 poison, i8 1, 0
  %struct.iiri1 = insertvalue %struct.char4 %struct.iiri0, i8 2, 1
  %struct.iiri2 = insertvalue %struct.char4 %struct.iiri1, i8 %c, 2
  %struct.iiri3 = insertvalue %struct.char4 %struct.iiri2, i8 4, 3
  call void @call_v4_i8(%struct.char4 %struct.iiri3)

  %struct.irii0 = insertvalue %struct.char4 poison, i8 1, 0
  %struct.irii1 = insertvalue %struct.char4 %struct.irii0, i8 %b, 1
  %struct.irii2 = insertvalue %struct.char4 %struct.irii1, i8 3, 2
  %struct.irii3 = insertvalue %struct.char4 %struct.irii2, i8 4, 3
  call void @call_v4_i8(%struct.char4 %struct.irii3)

  %struct.riii0 = insertvalue %struct.char4 poison, i8 %a, 0
  %struct.riii1 = insertvalue %struct.char4 %struct.riii0, i8 2, 1
  %struct.riii2 = insertvalue %struct.char4 %struct.riii1, i8 3, 2
  %struct.riii3 = insertvalue %struct.char4 %struct.riii2, i8 4, 3
  call void @call_v4_i8(%struct.char4 %struct.riii3)
  ret void
}

; CHECK-LABEL: st_param_v4_i16
; CHECK: st.param.v4.b16 [param0+0], {1, 2, 3, 4}
; CHECK: st.param.v4.b16 [param0+0], {1, {{%rs[0-9]+}}, {{%rs[0-9]+}}, {{%rs[0-9]+}}}
; CHECK: st.param.v4.b16 [param0+0], {{{%rs[0-9]+}}, 2, {{%rs[0-9]+}}, {{%rs[0-9]+}}}
; CHECK: st.param.v4.b16 [param0+0], {{{%rs[0-9]+}}, {{%rs[0-9]+}}, 3, {{%rs[0-9]+}}}
; CHECK: st.param.v4.b16 [param0+0], {{{%rs[0-9]+}}, {{%rs[0-9]+}}, {{%rs[0-9]+}}, 4}
; CHECK: st.param.v4.b16 [param0+0], {1, 2, {{%rs[0-9]+}}, {{%rs[0-9]+}}}
; CHECK: st.param.v4.b16 [param0+0], {1, {{%rs[0-9]+}}, 3, {{%rs[0-9]+}}}
; CHECK: st.param.v4.b16 [param0+0], {1, {{%rs[0-9]+}}, {{%rs[0-9]+}}, 4}
; CHECK: st.param.v4.b16 [param0+0], {{{%rs[0-9]+}}, 2, 3, {{%rs[0-9]+}}}
; CHECK: st.param.v4.b16 [param0+0], {{{%rs[0-9]+}}, 2, {{%rs[0-9]+}}, 4}
; CHECK: st.param.v4.b16 [param0+0], {{{%rs[0-9]+}}, {{%rs[0-9]+}}, 3, 4}
; CHECK: st.param.v4.b16 [param0+0], {1, 2, 3, {{%rs[0-9]+}}}
; CHECK: st.param.v4.b16 [param0+0], {1, 2, {{%rs[0-9]+}}, 4}
; CHECK: st.param.v4.b16 [param0+0], {1, {{%rs[0-9]+}}, 3, 4}
; CHECK: st.param.v4.b16 [param0+0], {{{%rs[0-9]+}}, 2, 3, 4}
define void @st_param_v4_i16(i16 %a, i16 %b, i16 %c, i16 %d) {
  call void @call_v4_i16(%struct.short4 { i16 1, i16 2, i16 3, i16 4 })

  %struct.irrr0 = insertvalue %struct.short4 poison, i16 1, 0
  %struct.irrr1 = insertvalue %struct.short4 %struct.irrr0, i16 %b, 1
  %struct.irrr2 = insertvalue %struct.short4 %struct.irrr1, i16 %c, 2
  %struct.irrr3 = insertvalue %struct.short4 %struct.irrr2, i16 %d, 3
  call void @call_v4_i16(%struct.short4 %struct.irrr3)

  %struct.rirr0 = insertvalue %struct.short4 poison, i16 %a, 0
  %struct.rirr1 = insertvalue %struct.short4 %struct.rirr0, i16 2, 1
  %struct.rirr2 = insertvalue %struct.short4 %struct.rirr1, i16 %c, 2
  %struct.rirr3 = insertvalue %struct.short4 %struct.rirr2, i16 %d, 3
  call void @call_v4_i16(%struct.short4 %struct.rirr3)

  %struct.rrir0 = insertvalue %struct.short4 poison, i16 %a, 0
  %struct.rrir1 = insertvalue %struct.short4 %struct.rrir0, i16 %b, 1
  %struct.rrir2 = insertvalue %struct.short4 %struct.rrir1, i16 3, 2
  %struct.rrir3 = insertvalue %struct.short4 %struct.rrir2, i16 %d, 3
  call void @call_v4_i16(%struct.short4 %struct.rrir3)

  %struct.rrri0 = insertvalue %struct.short4 poison, i16 %a, 0
  %struct.rrri1 = insertvalue %struct.short4 %struct.rrri0, i16 %b, 1
  %struct.rrri2 = insertvalue %struct.short4 %struct.rrri1, i16 %c, 2
  %struct.rrri3 = insertvalue %struct.short4 %struct.rrri2, i16 4, 3
  call void @call_v4_i16(%struct.short4 %struct.rrri3)

  %struct.iirr0 = insertvalue %struct.short4 poison, i16 1, 0
  %struct.iirr1 = insertvalue %struct.short4 %struct.iirr0, i16 2, 1
  %struct.iirr2 = insertvalue %struct.short4 %struct.iirr1, i16 %c, 2
  %struct.iirr3 = insertvalue %struct.short4 %struct.iirr2, i16 %d, 3
  call void @call_v4_i16(%struct.short4 %struct.iirr3)

  %struct.irir0 = insertvalue %struct.short4 poison, i16 1, 0
  %struct.irir1 = insertvalue %struct.short4 %struct.irir0, i16 %b, 1
  %struct.irir2 = insertvalue %struct.short4 %struct.irir1, i16 3, 2
  %struct.irir3 = insertvalue %struct.short4 %struct.irir2, i16 %d, 3
  call void @call_v4_i16(%struct.short4 %struct.irir3)

  %struct.irri0 = insertvalue %struct.short4 poison, i16 1, 0
  %struct.irri1 = insertvalue %struct.short4 %struct.irri0, i16 %b, 1
  %struct.irri2 = insertvalue %struct.short4 %struct.irri1, i16 %c, 2
  %struct.irri3 = insertvalue %struct.short4 %struct.irri2, i16 4, 3
  call void @call_v4_i16(%struct.short4 %struct.irri3)

  %struct.riir0 = insertvalue %struct.short4 poison, i16 %a, 0
  %struct.riir1 = insertvalue %struct.short4 %struct.riir0, i16 2, 1
  %struct.riir2 = insertvalue %struct.short4 %struct.riir1, i16 3, 2
  %struct.riir3 = insertvalue %struct.short4 %struct.riir2, i16 %d, 3
  call void @call_v4_i16(%struct.short4 %struct.riir3)

  %struct.riri0 = insertvalue %struct.short4 poison, i16 %a, 0
  %struct.riri1 = insertvalue %struct.short4 %struct.riri0, i16 2, 1
  %struct.riri2 = insertvalue %struct.short4 %struct.riri1, i16 %c, 2
  %struct.riri3 = insertvalue %struct.short4 %struct.riri2, i16 4, 3
  call void @call_v4_i16(%struct.short4 %struct.riri3)

  %struct.rrii0 = insertvalue %struct.short4 poison, i16 %a, 0
  %struct.rrii1 = insertvalue %struct.short4 %struct.rrii0, i16 %b, 1
  %struct.rrii2 = insertvalue %struct.short4 %struct.rrii1, i16 3, 2
  %struct.rrii3 = insertvalue %struct.short4 %struct.rrii2, i16 4, 3
  call void @call_v4_i16(%struct.short4 %struct.rrii3)

  %struct.iiir0 = insertvalue %struct.short4 poison, i16 1, 0
  %struct.iiir1 = insertvalue %struct.short4 %struct.iiir0, i16 2, 1
  %struct.iiir2 = insertvalue %struct.short4 %struct.iiir1, i16 3, 2
  %struct.iiir3 = insertvalue %struct.short4 %struct.iiir2, i16 %d, 3
  call void @call_v4_i16(%struct.short4 %struct.iiir3)

  %struct.iiri0 = insertvalue %struct.short4 poison, i16 1, 0
  %struct.iiri1 = insertvalue %struct.short4 %struct.iiri0, i16 2, 1
  %struct.iiri2 = insertvalue %struct.short4 %struct.iiri1, i16 %c, 2
  %struct.iiri3 = insertvalue %struct.short4 %struct.iiri2, i16 4, 3
  call void @call_v4_i16(%struct.short4 %struct.iiri3)

  %struct.irii0 = insertvalue %struct.short4 poison, i16 1, 0
  %struct.irii1 = insertvalue %struct.short4 %struct.irii0, i16 %b, 1
  %struct.irii2 = insertvalue %struct.short4 %struct.irii1, i16 3, 2
  %struct.irii3 = insertvalue %struct.short4 %struct.irii2, i16 4, 3
  call void @call_v4_i16(%struct.short4 %struct.irii3)

  %struct.riii0 = insertvalue %struct.short4 poison, i16 %a, 0
  %struct.riii1 = insertvalue %struct.short4 %struct.riii0, i16 2, 1
  %struct.riii2 = insertvalue %struct.short4 %struct.riii1, i16 3, 2
  %struct.riii3 = insertvalue %struct.short4 %struct.riii2, i16 4, 3
  call void @call_v4_i16(%struct.short4 %struct.riii3)
  ret void
}

; CHECK-LABEL: st_param_v4_i32
; CHECK: st.param.v4.b32 [param0+0], {1, 2, 3, 4}
; CHECK: st.param.v4.b32 [param0+0], {1, {{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}}
; CHECK: st.param.v4.b32 [param0+0], {{{%r[0-9]+}}, 2, {{%r[0-9]+}}, {{%r[0-9]+}}}
; CHECK: st.param.v4.b32 [param0+0], {{{%r[0-9]+}}, {{%r[0-9]+}}, 3, {{%r[0-9]+}}}
; CHECK: st.param.v4.b32 [param0+0], {{{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}, 4}
; CHECK: st.param.v4.b32 [param0+0], {1, 2, {{%r[0-9]+}}, {{%r[0-9]+}}}
; CHECK: st.param.v4.b32 [param0+0], {1, {{%r[0-9]+}}, 3, {{%r[0-9]+}}}
; CHECK: st.param.v4.b32 [param0+0], {1, {{%r[0-9]+}}, {{%r[0-9]+}}, 4}
; CHECK: st.param.v4.b32 [param0+0], {{{%r[0-9]+}}, 2, 3, {{%r[0-9]+}}}
; CHECK: st.param.v4.b32 [param0+0], {{{%r[0-9]+}}, 2, {{%r[0-9]+}}, 4}
; CHECK: st.param.v4.b32 [param0+0], {{{%r[0-9]+}}, {{%r[0-9]+}}, 3, 4}
; CHECK: st.param.v4.b32 [param0+0], {1, 2, 3, {{%r[0-9]+}}}
; CHECK: st.param.v4.b32 [param0+0], {1, 2, {{%r[0-9]+}}, 4}
; CHECK: st.param.v4.b32 [param0+0], {1, {{%r[0-9]+}}, 3, 4}
; CHECK: st.param.v4.b32 [param0+0], {{{%r[0-9]+}}, 2, 3, 4}
define void @st_param_v4_i32(i32 %a, i32 %b, i32 %c, i32 %d) {
  call void @call_v4_i32(%struct.int4 { i32 1, i32 2, i32 3, i32 4 })

  %struct.irrr0 = insertvalue %struct.int4 poison, i32 1, 0
  %struct.irrr1 = insertvalue %struct.int4 %struct.irrr0, i32 %b, 1
  %struct.irrr2 = insertvalue %struct.int4 %struct.irrr1, i32 %c, 2
  %struct.irrr3 = insertvalue %struct.int4 %struct.irrr2, i32 %d, 3
  call void @call_v4_i32(%struct.int4 %struct.irrr3)

  %struct.rirr0 = insertvalue %struct.int4 poison, i32 %a, 0
  %struct.rirr1 = insertvalue %struct.int4 %struct.rirr0, i32 2, 1
  %struct.rirr2 = insertvalue %struct.int4 %struct.rirr1, i32 %c, 2
  %struct.rirr3 = insertvalue %struct.int4 %struct.rirr2, i32 %d, 3
  call void @call_v4_i32(%struct.int4 %struct.rirr3)

  %struct.rrir0 = insertvalue %struct.int4 poison, i32 %a, 0
  %struct.rrir1 = insertvalue %struct.int4 %struct.rrir0, i32 %b, 1
  %struct.rrir2 = insertvalue %struct.int4 %struct.rrir1, i32 3, 2
  %struct.rrir3 = insertvalue %struct.int4 %struct.rrir2, i32 %d, 3
  call void @call_v4_i32(%struct.int4 %struct.rrir3)

  %struct.rrri0 = insertvalue %struct.int4 poison, i32 %a, 0
  %struct.rrri1 = insertvalue %struct.int4 %struct.rrri0, i32 %b, 1
  %struct.rrri2 = insertvalue %struct.int4 %struct.rrri1, i32 %c, 2
  %struct.rrri3 = insertvalue %struct.int4 %struct.rrri2, i32 4, 3
  call void @call_v4_i32(%struct.int4 %struct.rrri3)

  %struct.iirr0 = insertvalue %struct.int4 poison, i32 1, 0
  %struct.iirr1 = insertvalue %struct.int4 %struct.iirr0, i32 2, 1
  %struct.iirr2 = insertvalue %struct.int4 %struct.iirr1, i32 %c, 2
  %struct.iirr3 = insertvalue %struct.int4 %struct.iirr2, i32 %d, 3
  call void @call_v4_i32(%struct.int4 %struct.iirr3)

  %struct.irir0 = insertvalue %struct.int4 poison, i32 1, 0
  %struct.irir1 = insertvalue %struct.int4 %struct.irir0, i32 %b, 1
  %struct.irir2 = insertvalue %struct.int4 %struct.irir1, i32 3, 2
  %struct.irir3 = insertvalue %struct.int4 %struct.irir2, i32 %d, 3
  call void @call_v4_i32(%struct.int4 %struct.irir3)

  %struct.irri0 = insertvalue %struct.int4 poison, i32 1, 0
  %struct.irri1 = insertvalue %struct.int4 %struct.irri0, i32 %b, 1
  %struct.irri2 = insertvalue %struct.int4 %struct.irri1, i32 %c, 2
  %struct.irri3 = insertvalue %struct.int4 %struct.irri2, i32 4, 3
  call void @call_v4_i32(%struct.int4 %struct.irri3)

  %struct.riir0 = insertvalue %struct.int4 poison, i32 %a, 0
  %struct.riir1 = insertvalue %struct.int4 %struct.riir0, i32 2, 1
  %struct.riir2 = insertvalue %struct.int4 %struct.riir1, i32 3, 2
  %struct.riir3 = insertvalue %struct.int4 %struct.riir2, i32 %d, 3
  call void @call_v4_i32(%struct.int4 %struct.riir3)

  %struct.riri0 = insertvalue %struct.int4 poison, i32 %a, 0
  %struct.riri1 = insertvalue %struct.int4 %struct.riri0, i32 2, 1
  %struct.riri2 = insertvalue %struct.int4 %struct.riri1, i32 %c, 2
  %struct.riri3 = insertvalue %struct.int4 %struct.riri2, i32 4, 3
  call void @call_v4_i32(%struct.int4 %struct.riri3)

  %struct.rrii0 = insertvalue %struct.int4 poison, i32 %a, 0
  %struct.rrii1 = insertvalue %struct.int4 %struct.rrii0, i32 %b, 1
  %struct.rrii2 = insertvalue %struct.int4 %struct.rrii1, i32 3, 2
  %struct.rrii3 = insertvalue %struct.int4 %struct.rrii2, i32 4, 3
  call void @call_v4_i32(%struct.int4 %struct.rrii3)

  %struct.iiir0 = insertvalue %struct.int4 poison, i32 1, 0
  %struct.iiir1 = insertvalue %struct.int4 %struct.iiir0, i32 2, 1
  %struct.iiir2 = insertvalue %struct.int4 %struct.iiir1, i32 3, 2
  %struct.iiir3 = insertvalue %struct.int4 %struct.iiir2, i32 %d, 3
  call void @call_v4_i32(%struct.int4 %struct.iiir3)

  %struct.iiri0 = insertvalue %struct.int4 poison, i32 1, 0
  %struct.iiri1 = insertvalue %struct.int4 %struct.iiri0, i32 2, 1
  %struct.iiri2 = insertvalue %struct.int4 %struct.iiri1, i32 %c, 2
  %struct.iiri3 = insertvalue %struct.int4 %struct.iiri2, i32 4, 3
  call void @call_v4_i32(%struct.int4 %struct.iiri3)

  %struct.irii0 = insertvalue %struct.int4 poison, i32 1, 0
  %struct.irii1 = insertvalue %struct.int4 %struct.irii0, i32 %b, 1
  %struct.irii2 = insertvalue %struct.int4 %struct.irii1, i32 3, 2
  %struct.irii3 = insertvalue %struct.int4 %struct.irii2, i32 4, 3
  call void @call_v4_i32(%struct.int4 %struct.irii3)

  %struct.riii0 = insertvalue %struct.int4 poison, i32 %a, 0
  %struct.riii1 = insertvalue %struct.int4 %struct.riii0, i32 2, 1
  %struct.riii2 = insertvalue %struct.int4 %struct.riii1, i32 3, 2
  %struct.riii3 = insertvalue %struct.int4 %struct.riii2, i32 4, 3
  call void @call_v4_i32(%struct.int4 %struct.riii3)
  ret void
}

; CHECK-LABEL: st_param_v4_f32
; CHECK: st.param.v4.f32 [param0+0], {0f3F800000, 0f40000000, 0f40400000, 0f40800000}
; CHECK: st.param.v4.f32 [param0+0], {0f3F800000, {{%f[0-9]+}}, {{%f[0-9]+}}, {{%f[0-9]+}}}
; CHECK: st.param.v4.f32 [param0+0], {{{%f[0-9]+}}, 0f40000000, {{%f[0-9]+}}, {{%f[0-9]+}}}
; CHECK: st.param.v4.f32 [param0+0], {{{%f[0-9]+}}, {{%f[0-9]+}}, 0f40400000, {{%f[0-9]+}}}
; CHECK: st.param.v4.f32 [param0+0], {{{%f[0-9]+}}, {{%f[0-9]+}}, {{%f[0-9]+}}, 0f40800000}
; CHECK: st.param.v4.f32 [param0+0], {0f3F800000, 0f40000000, {{%f[0-9]+}}, {{%f[0-9]+}}}
; CHECK: st.param.v4.f32 [param0+0], {0f3F800000, {{%f[0-9]+}}, 0f40400000, {{%f[0-9]+}}}
; CHECK: st.param.v4.f32 [param0+0], {0f3F800000, {{%f[0-9]+}}, {{%f[0-9]+}}, 0f40800000}
; CHECK: st.param.v4.f32 [param0+0], {{{%f[0-9]+}}, 0f40000000, 0f40400000, {{%f[0-9]+}}}
; CHECK: st.param.v4.f32 [param0+0], {{{%f[0-9]+}}, 0f40000000, {{%f[0-9]+}}, 0f40800000}
; CHECK: st.param.v4.f32 [param0+0], {{{%f[0-9]+}}, {{%f[0-9]+}}, 0f40400000, 0f40800000}
; CHECK: st.param.v4.f32 [param0+0], {0f3F800000, 0f40000000, 0f40400000, {{%f[0-9]+}}}
; CHECK: st.param.v4.f32 [param0+0], {0f3F800000, 0f40000000, {{%f[0-9]+}}, 0f40800000}
; CHECK: st.param.v4.f32 [param0+0], {0f3F800000, {{%f[0-9]+}}, 0f40400000, 0f40800000}
; CHECK: st.param.v4.f32 [param0+0], {{{%f[0-9]+}}, 0f40000000, 0f40400000, 0f40800000}
define void @st_param_v4_f32(float %a, float %b, float %c, float %d) {
  call void @call_v4_f32(%struct.float4 { float 1.0, float 2.0, float 3.0, float 4.0 })

  %struct.irrr0 = insertvalue %struct.float4 poison, float 1.0, 0
  %struct.irrr1 = insertvalue %struct.float4 %struct.irrr0, float %b, 1
  %struct.irrr2 = insertvalue %struct.float4 %struct.irrr1, float %c, 2
  %struct.irrr3 = insertvalue %struct.float4 %struct.irrr2, float %d, 3
  call void @call_v4_f32(%struct.float4 %struct.irrr3)

  %struct.rirr0 = insertvalue %struct.float4 poison, float %a, 0
  %struct.rirr1 = insertvalue %struct.float4 %struct.rirr0, float 2.0, 1
  %struct.rirr2 = insertvalue %struct.float4 %struct.rirr1, float %c, 2
  %struct.rirr3 = insertvalue %struct.float4 %struct.rirr2, float %d, 3
  call void @call_v4_f32(%struct.float4 %struct.rirr3)

  %struct.rrir0 = insertvalue %struct.float4 poison, float %a, 0
  %struct.rrir1 = insertvalue %struct.float4 %struct.rrir0, float %b, 1
  %struct.rrir2 = insertvalue %struct.float4 %struct.rrir1, float 3.0, 2
  %struct.rrir3 = insertvalue %struct.float4 %struct.rrir2, float %d, 3
  call void @call_v4_f32(%struct.float4 %struct.rrir3)

  %struct.rrri0 = insertvalue %struct.float4 poison, float %a, 0
  %struct.rrri1 = insertvalue %struct.float4 %struct.rrri0, float %b, 1
  %struct.rrri2 = insertvalue %struct.float4 %struct.rrri1, float %c, 2
  %struct.rrri3 = insertvalue %struct.float4 %struct.rrri2, float 4.0, 3
  call void @call_v4_f32(%struct.float4 %struct.rrri3)

  %struct.iirr0 = insertvalue %struct.float4 poison, float 1.0, 0
  %struct.iirr1 = insertvalue %struct.float4 %struct.iirr0, float 2.0, 1
  %struct.iirr2 = insertvalue %struct.float4 %struct.iirr1, float %c, 2
  %struct.iirr3 = insertvalue %struct.float4 %struct.iirr2, float %d, 3
  call void @call_v4_f32(%struct.float4 %struct.iirr3)

  %struct.irir0 = insertvalue %struct.float4 poison, float 1.0, 0
  %struct.irir1 = insertvalue %struct.float4 %struct.irir0, float %b, 1
  %struct.irir2 = insertvalue %struct.float4 %struct.irir1, float 3.0, 2
  %struct.irir3 = insertvalue %struct.float4 %struct.irir2, float %d, 3
  call void @call_v4_f32(%struct.float4 %struct.irir3)

  %struct.irri0 = insertvalue %struct.float4 poison, float 1.0, 0
  %struct.irri1 = insertvalue %struct.float4 %struct.irri0, float %b, 1
  %struct.irri2 = insertvalue %struct.float4 %struct.irri1, float %c, 2
  %struct.irri3 = insertvalue %struct.float4 %struct.irri2, float 4.0, 3
  call void @call_v4_f32(%struct.float4 %struct.irri3)

  %struct.riir0 = insertvalue %struct.float4 poison, float %a, 0
  %struct.riir1 = insertvalue %struct.float4 %struct.riir0, float 2.0, 1
  %struct.riir2 = insertvalue %struct.float4 %struct.riir1, float 3.0, 2
  %struct.riir3 = insertvalue %struct.float4 %struct.riir2, float %d, 3
  call void @call_v4_f32(%struct.float4 %struct.riir3)

  %struct.riri0 = insertvalue %struct.float4 poison, float %a, 0
  %struct.riri1 = insertvalue %struct.float4 %struct.riri0, float 2.0, 1
  %struct.riri2 = insertvalue %struct.float4 %struct.riri1, float %c, 2
  %struct.riri3 = insertvalue %struct.float4 %struct.riri2, float 4.0, 3
  call void @call_v4_f32(%struct.float4 %struct.riri3)

  %struct.rrii0 = insertvalue %struct.float4 poison, float %a, 0
  %struct.rrii1 = insertvalue %struct.float4 %struct.rrii0, float %b, 1
  %struct.rrii2 = insertvalue %struct.float4 %struct.rrii1, float 3.0, 2
  %struct.rrii3 = insertvalue %struct.float4 %struct.rrii2, float 4.0, 3
  call void @call_v4_f32(%struct.float4 %struct.rrii3)

  %struct.iiir0 = insertvalue %struct.float4 poison, float 1.0, 0
  %struct.iiir1 = insertvalue %struct.float4 %struct.iiir0, float 2.0, 1
  %struct.iiir2 = insertvalue %struct.float4 %struct.iiir1, float 3.0, 2
  %struct.iiir3 = insertvalue %struct.float4 %struct.iiir2, float %d, 3
  call void @call_v4_f32(%struct.float4 %struct.iiir3)

  %struct.iiri0 = insertvalue %struct.float4 poison, float 1.0, 0
  %struct.iiri1 = insertvalue %struct.float4 %struct.iiri0, float 2.0, 1
  %struct.iiri2 = insertvalue %struct.float4 %struct.iiri1, float %c, 2
  %struct.iiri3 = insertvalue %struct.float4 %struct.iiri2, float 4.0, 3
  call void @call_v4_f32(%struct.float4 %struct.iiri3)

  %struct.irii0 = insertvalue %struct.float4 poison, float 1.0, 0
  %struct.irii1 = insertvalue %struct.float4 %struct.irii0, float %b, 1
  %struct.irii2 = insertvalue %struct.float4 %struct.irii1, float 3.0, 2
  %struct.irii3 = insertvalue %struct.float4 %struct.irii2, float 4.0, 3
  call void @call_v4_f32(%struct.float4 %struct.irii3)

  %struct.riii0 = insertvalue %struct.float4 poison, float %a, 0
  %struct.riii1 = insertvalue %struct.float4 %struct.riii0, float 2.0, 1
  %struct.riii2 = insertvalue %struct.float4 %struct.riii1, float 3.0, 2
  %struct.riii3 = insertvalue %struct.float4 %struct.riii2, float 4.0, 3
  call void @call_v4_f32(%struct.float4 %struct.riii3)
  ret void
}

declare void @call_v4_i8(%struct.char4)
declare void @call_v4_i16(%struct.short4)
declare void @call_v4_i32(%struct.int4)
declare void @call_v4_f32(%struct.float4)

!nvvm.annotations = !{!1, !2, !3, !4, !5, !6, !7, !8, !9, !10}
!1 = !{ptr @call_v2_i8, !"align", i32 65538}
!2 = !{ptr @call_v2_i16, !"align", i32 65540}
!3 = !{ptr @call_v2_i32, !"align", i32 65544}
!4 = !{ptr @call_v2_i64, !"align", i32 65552}
!5 = !{ptr @call_v2_f32, !"align", i32 65544}
!6 = !{ptr @call_v2_f64, !"align", i32 65552}
!7 = !{ptr @call_v4_i8, !"align", i32 65540}
!8 = !{ptr @call_v4_i16, !"align", i32 65544}
!9 = !{ptr @call_v4_i32, !"align", i32 65552}
!10 = !{ptr @call_v4_f32, !"align", i32 65552}
