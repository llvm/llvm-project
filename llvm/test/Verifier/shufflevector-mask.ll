; RUN: opt -passes=verify -disable-output %s

; concat(V1, V2) = (2 * 1073741824) overflows int range but the mask indices are still within range
define <2 x i8> @shuffle_v1073741824i8_index_zero_first(<1073741824 x i8> %a, <1073741824 x i8> %b) {
  %s = shufflevector <1073741824 x i8> %a, <1073741824 x i8> %b, <2 x i32> <i32 0, i32 1>
  ret <2 x i8> %s
}

; concat(V1, V2) = (2 * 1073741824) overflows int range and 0th and last element of the concatenated vector is referred.
define <2 x i8> @shuffle_v1073741824i8_index_zero_last(<1073741824 x i8> %a, <1073741824 x i8> %b) {
  %s = shufflevector <1073741824 x i8> %a, <1073741824 x i8> %b, <2 x i32> <i32 0, i32 2147483647>
  ret <2 x i8> %s
}

