; Function Attrs: nounwind
define <2 x double> @broadcast_v2f64(<2 x double> %a, <2 x double> %b) #0 {
  %s = shufflevector <2 x double> %a, <2 x double> undef, <2 x i32> zeroinitializer
  ret <2 x double> %s
}

; Function Attrs: nounwind
; define <2 x double> @broadcast_v2f64_from_second(<2 x double> %a, <2 x double> %b) #0 {
;   %s = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 2, i32 2>
;   ret <2 x double> %s
; }

; ; Function Attrs: nounwind
; define <4 x float> @broadcast_v4f32(<4 x float> %a, <4 x float> %b) #0 {
;   %s = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> zeroinitializer
;   ret <4 x float> %s
; }

; ; Function Attrs: nounwind
; define <4 x float> @broadcast_v4f32_lane1(<4 x float> %a, <4 x float> %b) #0 {
;   %s = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
;   ret <4 x float> %s
; }

; ; Function Attrs: nounwind
; define <4 x float> @broadcast_v4f32_from_second(<4 x float> %a, <4 x float> %b) #0 {
;   %s = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
;   ret <4 x float> %s
; }

; ; Function Attrs: nounwind
; define <8 x i16> @broadcast_v8i16(<8 x i16> %a, <8 x i16> %b) #0 {
;   %s = shufflevector <8 x i16> %a, <8 x i16> undef, <8 x i32> zeroinitializer
;   ret <8 x i16> %s
; }

; ; Function Attrs: nounwind
; define <8 x i16> @broadcast_v8i16_lane3(<8 x i16> %a, <8 x i16> %b) #0 {
;   %s = shufflevector <8 x i16> %a, <8 x i16> undef, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
;   ret <8 x i16> %s
; }

; ; Function Attrs: nounwind
; define <16 x i8> @broadcast_v16i8(<16 x i8> %a, <16 x i8> %b) #0 {
;   %s = shufflevector <16 x i8> %a, <16 x i8> undef, <16 x i32> zeroinitializer
;   ret <16 x i8> %s
; }

; ; Function Attrs: nounwind
; define <16 x i8> @broadcast_v16i8_lane7(<16 x i8> %a, <16 x i8> %b) #0 {
;   %s = shufflevector <16 x i8> %a, <16 x i8> undef, <16 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
;   ret <16 x i8> %s
; }

; ; Function Attrs: nounwind
; define <2 x i64> @broadcast_v2i64(<2 x i64> %a, <2 x i64> %b) #0 {
;   %s = shufflevector <2 x i64> %a, <2 x i64> undef, <2 x i32> zeroinitializer
;   ret <2 x i64> %s
; }

; ; Function Attrs: nounwind
; define <2 x i64> @broadcast_v2i64_lane1(<2 x i64> %a, <2 x i64> %b) #0 {
;   %s = shufflevector <2 x i64> %a, <2 x i64> undef, <2 x i32> <i32 1, i32 1>
;   ret <2 x i64> %s
; }

attributes #0 = { nounwind }
