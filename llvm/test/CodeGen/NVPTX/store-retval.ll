; RUN: llc < %s --mtriple=nvptx-unknown-unknown | FileCheck %s
; RUN: %if ptxas %{ llc < %s --mtriple=nvptx-unknown-unknown | %ptxas-verify %}
;
; This is IR generated with clang using -O3 optimization level
; and nvptx-unknown-unknown target from the following C code.
;
; struct StNoalign {              unsigned int field[5]; };
; struct StAlign8  { _Alignas(8)  unsigned int field[5]; };
; struct StAlign16 { _Alignas(16) unsigned int field[5]; };
;
; #define DECLARE_FUNC(StName)                        \
; struct StName func_##StName(struct StName in) {     \
;   struct StName ret;                                \
;   ret.field[4]  = in.field[0];                      \
;   return ret;                                       \
; }                                                   \
;
; DECLARE_FUNC(StNoalign)
; DECLARE_FUNC(StAlign8)
; DECLARE_FUNC(StAlign16)

%struct.StNoalign = type { [5 x i32] }

define %struct.StNoalign @func_StNoalign(%struct.StNoalign* nocapture noundef readonly byval(%struct.StNoalign) align 4 %in) {
  ; CHECK-LABEL: .func{{.*}}func_StNoalign
  ; CHECK:       ld.param.u32    [[R1:%r[0-9]+]],   [func_StNoalign_param_0];
  ; CHECK-NOT:   st.param.b32    [func_retval0+0],  %r{{[0-9]+}};
  ; CHECK-NOT:   st.param.b32    [func_retval0+4],  %r{{[0-9]+}};
  ; CHECK-NOT:   st.param.b32    [func_retval0+8],  %r{{[0-9]+}};
  ; CHECK-NOT:   st.param.b32    [func_retval0+12], %r{{[0-9]+}};
  ; CHECK:       st.param.b32    [func_retval0+16], [[R1]];
  ; CHECK-NEXT:  ret;
  %arrayidx = getelementptr inbounds %struct.StNoalign, %struct.StNoalign* %in, i32 0, i32 0, i32 0
  %1 = load i32, i32* %arrayidx, align 4
  %.fca.0.4.insert = insertvalue %struct.StNoalign { [5 x i32] [i32 undef, i32 undef, i32 undef, i32 undef, i32 poison] }, i32 %1, 0, 4
  ret %struct.StNoalign %.fca.0.4.insert
}

%struct.StAlign8 = type { [5 x i32], [4 x i8] }

define %struct.StAlign8 @func_StAlign8(%struct.StAlign8* nocapture noundef readonly byval(%struct.StAlign8) align 8 %in) {
  ; CHECK-LABEL: .func{{.*}}func_StAlign8
  ; CHECK:       ld.param.u32    [[R1:%r[0-9]+]],   [func_StAlign8_param_0];
  ; CHECK-NOT:   st.param.b32    [func_retval0+0],  %r{{[0-9]+}};
  ; CHECK-NOT:   st.param.b32    [func_retval0+4],  %r{{[0-9]+}};
  ; CHECK-NOT:   st.param.b32    [func_retval0+8],  %r{{[0-9]+}};
  ; CHECK-NOT:   st.param.b32    [func_retval0+12], %r{{[0-9]+}};
  ; CHECK:       st.param.b32    [func_retval0+16], [[R1]];
  ; CHECK-NOT:   st.param.v4.b8  [func_retval0+20], {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}};
  ; CHECK-NEXT:  ret;
  %arrayidx = getelementptr inbounds %struct.StAlign8, %struct.StAlign8* %in, i32 0, i32 0, i32 0
  %1 = load i32, i32* %arrayidx, align 8
  %.fca.0.4.insert = insertvalue %struct.StAlign8 { [5 x i32] [i32 undef, i32 undef, i32 undef, i32 undef, i32 poison], [4 x i8] poison }, i32 %1, 0, 4
  ret %struct.StAlign8 %.fca.0.4.insert
}

%struct.StAlign16 = type { [5 x i32], [12 x i8] }

define %struct.StAlign16 @func_StAlign16(%struct.StAlign16* nocapture noundef readonly byval(%struct.StAlign16) align 16 %in) {
  ; CHECK-LABEL: .func{{.*}}func_StAlign16
  ; CHECK:       ld.param.u32    [[R1:%r[0-9]+]],   [func_StAlign16_param_0];
  ; CHECK-NOT:   st.param.b32    [func_retval0+0],  %r{{[0-9]+}};
  ; CHECK-NOT:   st.param.b32    [func_retval0+4],  %r{{[0-9]+}};
  ; CHECK-NOT:   st.param.b32    [func_retval0+8],  %r{{[0-9]+}};
  ; CHECK-NOT:   st.param.b32    [func_retval0+12], %r{{[0-9]+}};
  ; CHECK:       st.param.b32    [func_retval0+16], [[R1]];
  ; CHECK-NOT:   st.param.v4.b8  [func_retval0+20], {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}};
  ; CHECK-NOT:   st.param.v4.b8  [func_retval0+24], {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}};
  ; CHECK-NOT:   st.param.v4.b8  [func_retval0+28], {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}};
  ; CHECK-NEXT:  ret;
  %arrayidx = getelementptr inbounds %struct.StAlign16, %struct.StAlign16* %in, i32 0, i32 0, i32 0
  %1 = load i32, i32* %arrayidx, align 16
  %.fca.0.4.insert = insertvalue %struct.StAlign16 { [5 x i32] [i32 undef, i32 undef, i32 undef, i32 undef, i32 poison], [12 x i8] poison }, i32 %1, 0, 4
  ret %struct.StAlign16 %.fca.0.4.insert
}
