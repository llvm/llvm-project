; RUN: llc -verify-machineinstrs -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names < %s -mtriple=ppc64le-- -mcpu=pwr8 | FileCheck %s --check-prefixes=CHECK,CHECK-P8
; RUN: llc -verify-machineinstrs -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names < %s -mtriple=ppc64le-- -mcpu=pwr9 | FileCheck %s --check-prefixes=CHECK,CHECK-P9

define <16 x i8> @test1_v16i8(<16 x i8> %a) {
        %tmp.1 = mul nsw <16 x i8> %a, <i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16>         ; <<16 x i8>> [#uses=1]
        ret <16 x i8> %tmp.1
}
; CHECK-LABEL: test1_v16i8:
; CHECK-P8: vspltisb v[[REG1:[0-9]+]], 4
; CHECK-P9: xxspltib v[[REG1:[0-9]+]], 4
; CHECK-NOT: vmul
; CHECK-NEXT: vslb v[[REG2:[0-9]+]], v2, v[[REG1]]

define <16 x i8> @test2_v16i8(<16 x i8> %a) {
        %tmp.1 = mul nsw <16 x i8> %a, <i8 17, i8 17, i8 17, i8 17, i8 17, i8 17, i8 17, i8 17, i8 17, i8 17, i8 17, i8 17, i8 17, i8 17, i8 17, i8 17>         ; <<16 x i8>> [#uses=1]
        ret <16 x i8> %tmp.1
}
; CHECK-LABEL: test2_v16i8:
; CHECK-P8: vspltisb v[[REG1:[0-9]+]], 4
; CHECK-P9: xxspltib v[[REG1:[0-9]+]], 4
; CHECK-NOT: vmul
; CHECK-NEXT: vslb v[[REG2:[0-9]+]], v2, v[[REG1]]
; CHECK-NEXT: vaddubm v[[REG3:[0-9]+]], v2, v[[REG2]]

define <16 x i8> @test3_v16i8(<16 x i8> %a) {
        %tmp.1 = mul nsw <16 x i8> %a, <i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15>         ; <<16 x i8>> [#uses=1]
        ret <16 x i8> %tmp.1
}
; CHECK-LABEL: test3_v16i8:
; CHECK-P8: vspltisb v[[REG1:[0-9]+]], 4
; CHECK-P9: xxspltib v[[REG1:[0-9]+]], 4
; CHECK-NOT: vmul
; CHECK-NEXT: vslb v[[REG2:[0-9]+]], v2, v[[REG1]]
; CHECK-NEXT: vsububm v[[REG3:[0-9]+]], v[[REG2]], v2

; negtive constant

define <16 x i8> @test4_v16i8(<16 x i8> %a) {
        %tmp.1 = mul nsw <16 x i8> %a, <i8 -16, i8 -16, i8 -16, i8 -16, i8 -16, i8 -16, i8 -16, i8 -16, i8 -16, i8 -16, i8 -16, i8 -16, i8 -16, i8 -16, i8 -16, i8 -16>         ; <<16 x i8>> [#uses=1]
        ret <16 x i8> %tmp.1
}
; CHECK-LABEL: test4_v16i8:
; CHECK-P8: vspltisb v[[REG1:[0-9]+]], 4
; CHECK-P9: xxspltib v[[REG1:[0-9]+]], 4
; CHECK-NOT: vmul
; CHECK-NEXT: vslb v[[REG3:[0-9]+]], v2, v[[REG1]]
; CHECK-NEXT: xxlxor v[[REG2:[0-9]+]],
; CHECK-NEXT: vsububm v[[REG4:[0-9]+]], v[[REG2]], v[[REG3]]

define <16 x i8> @test5_v16i8(<16 x i8> %a) {
        %tmp.1 = mul nsw <16 x i8> %a, <i8 -17, i8 -17, i8 -17, i8 -17, i8 -17, i8 -17, i8 -17, i8 -17, i8 -17, i8 -17, i8 -17, i8 -17, i8 -17, i8 -17, i8 -17, i8 -17>         ; <<16 x i8>> [#uses=1]
        ret <16 x i8> %tmp.1
}
; CHECK-LABEL: test5_v16i8:
; CHECK-P8: vspltisb v[[REG1:[0-9]+]], 4
; CHECK-P9: xxspltib v[[REG1:[0-9]+]], 4
; CHECK-NOT: vmul
; CHECK-NEXT: vslb v[[REG3:[0-9]+]], v2, v[[REG1]]
; CHECK-NEXT: vaddubm v[[REG4:[0-9]+]], v2, v[[REG3]]
; CHECK-NEXT: xxlxor v[[REG2:[0-9]+]],
; CHECK-NEXT: vsububm v[[REG5:[0-9]+]], v[[REG2]], v[[REG4]]

define <16 x i8> @test6_v16i8(<16 x i8> %a) {
        %tmp.1 = mul nsw <16 x i8> %a, <i8 -15, i8 -15, i8 -15, i8 -15, i8 -15, i8 -15, i8 -15, i8 -15, i8 -15, i8 -15, i8 -15, i8 -15, i8 -15, i8 -15, i8 -15, i8 -15>         ; <<16 x i8>> [#uses=1]
        ret <16 x i8> %tmp.1
}
; CHECK-LABEL: test6_v16i8:
; CHECK-P8: vspltisb v[[REG1:[0-9]+]], 4
; CHECK-P9: xxspltib v[[REG1:[0-9]+]], 4
; CHECK-NOT: vmul
; CHECK-NEXT: vslb v[[REG2:[0-9]+]], v2, v[[REG1]]
; CHECK-NEXT: vsububm v[[REG3:[0-9]+]], v2, v[[REG2]]

; boundary case

define <16 x i8> @test7_v16i8(<16 x i8> %a) {
        %tmp.1 = mul nsw <16 x i8> %a, <i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128> ; <<16 x i8>> [#uses=1]
        ret <16 x i8> %tmp.1
}
; CHECK-LABEL: test7_v16i8:
; CHECK-P8: vspltisb v[[REG1:[0-9]+]], 7
; CHECK-P9: xxspltib v[[REG1:[0-9]+]], 7
; CHECK-NOT: vmul
; CHECK-NEXT: vslb v[[REG5:[0-9]+]], v2, v[[REG1]]

define <16 x i8> @test8_v16i8(<16 x i8> %a) {
        %tmp.1 = mul nsw <16 x i8> %a, <i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127> ; <<16 x i8>> [#uses=1]
        ret <16 x i8> %tmp.1
}
; CHECK-LABEL: test8_v16i8:
; CHECK-P8: vspltisb v[[REG1:[0-9]+]], 7
; CHECK-P9: xxspltib v[[REG1:[0-9]+]], 7
; CHECK-NOT: vmul
; CHECK-NEXT: vslb v[[REG2:[0-9]+]], v2, v[[REG1]]
; CHECK-NEXT: vsububm v[[REG3:[0-9]+]], v[[REG2]], v2

define <8 x i16> @test1_v8i16(<8 x i16> %a) {
        %tmp.1 = mul nsw <8 x i16> %a, <i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16>         ; <<8 x i16>> [#uses=1]
        ret <8 x i16> %tmp.1
}
; CHECK-LABEL: test1_v8i16:
; CHECK: vspltish v[[REG1:[0-9]+]], 4
; CHECK-NOT: vmul
; CHECK-NEXT: vslh v[[REG2:[0-9]+]], v2, v[[REG1]]

define <8 x i16> @test2_v8i16(<8 x i16> %a) {
        %tmp.1 = mul nsw <8 x i16> %a, <i16 17, i16 17, i16 17, i16 17, i16 17, i16 17, i16 17, i16 17>         ; <<8 x i16>> [#uses=1]
        ret <8 x i16> %tmp.1
}
; CHECK-LABEL: test2_v8i16:
; CHECK: vspltish v[[REG1:[0-9]+]], 4
; CHECK-NOT: vmul
; CHECK-NEXT: vslh v[[REG2:[0-9]+]], v2, v[[REG1]]
; CHECK-NEXT: vadduhm v[[REG3:[0-9]+]], v2, v[[REG2]]

define <8 x i16> @test3_v8i16(<8 x i16> %a) {
        %tmp.1 = mul nsw <8 x i16> %a, <i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15>         ; <<8 x i16>> [#uses=1]
        ret <8 x i16> %tmp.1
}
; CHECK-LABEL: test3_v8i16:
; CHECK: vspltish v[[REG1:[0-9]+]], 4
; CHECK-NOT: vmul
; CHECK-NEXT: vslh v[[REG2:[0-9]+]], v2, v[[REG1]]
; CHECK-NEXT: vsubuhm v[[REG3:[0-9]+]], v[[REG2]], v2

; negtive constant

define <8 x i16> @test4_v8i16(<8 x i16> %a) {
        %tmp.1 = mul nsw <8 x i16> %a, <i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16>         ; <<8 x i16>> [#uses=1]
        ret <8 x i16> %tmp.1
}
; CHECK-LABEL: test4_v8i16:
; CHECK: vspltish v[[REG1:[0-9]+]], 4
; CHECK-NOT: vmul
; CHECK-NEXT: vslh v[[REG3:[0-9]+]], v2, v[[REG1]]
; CHECK-NEXT: xxlxor v[[REG2:[0-9]+]],
; CHECK-NEXT: vsubuhm v[[REG4:[0-9]+]], v[[REG2]], v[[REG3]]

define <8 x i16> @test5_v8i16(<8 x i16> %a) {
        %tmp.1 = mul nsw <8 x i16> %a, <i16 -17, i16 -17, i16 -17, i16 -17, i16 -17, i16 -17, i16 -17, i16 -17>         ; <<8 x i16>> [#uses=1]
        ret <8 x i16> %tmp.1
}
; CHECK-LABEL: test5_v8i16:
; CHECK: vspltish v[[REG1:[0-9]+]], 4
; CHECK-NOT: vmul
; CHECK-NEXT: vslh v[[REG3:[0-9]+]], v2, v[[REG1]]
; CHECK-NEXT: vadduhm v[[REG4:[0-9]+]], v2, v[[REG3]]
; CHECK-NEXT: xxlxor v[[REG2:[0-9]+]],
; CHECK-NEXT: vsubuhm v[[REG5:[0-9]+]], v[[REG2]], v[[REG4]]

define <8 x i16> @test6_v8i16(<8 x i16> %a) {
        %tmp.1 = mul nsw <8 x i16> %a, <i16 -15, i16 -15, i16 -15, i16 -15, i16 -15, i16 -15, i16 -15, i16 -15>         ; <<8 x i16>> [#uses=1]
        ret <8 x i16> %tmp.1
}
; CHECK-LABEL: test6_v8i16:
; CHECK: vspltish v[[REG1:[0-9]+]], 4
; CHECK-NOT: vmul
; CHECK-NEXT: vslh v[[REG2:[0-9]+]], v2, v[[REG1]]
; CHECK-NEXT: vsubuhm v[[REG3:[0-9]+]], v2, v[[REG2]]

; boundary case

define <8 x i16> @test7_v8i16(<8 x i16> %a) {
        %tmp.1 = mul nsw <8 x i16> %a, <i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768> ; <<8 x i16>> [#uses=1]
        ret <8 x i16> %tmp.1
}
; CHECK-LABEL: test7_v8i16:
; CHECK: vspltish v[[REG1:[0-9]+]], 15
; CHECK-NOT: vmul
; CHECK-NEXT: vslh v[[REG5:[0-9]+]], v2, v[[REG1]]

define <8 x i16> @test8_v8i16(<8 x i16> %a) {
        %tmp.1 = mul nsw <8 x i16> %a, <i16 32767, i16 32767, i16 32767, i16 32767, i16 32767, i16 32767, i16 32767, i16 32767> ; <<8 x i16>> [#uses=1]
        ret <8 x i16> %tmp.1
}
; CHECK-LABEL: test8_v8i16:
; CHECK: vspltish v[[REG1:[0-9]+]], 15
; CHECK-NOT: vmul
; CHECK-NEXT: vslh v[[REG2:[0-9]+]], v2, v[[REG1]]
; CHECK-NEXT: vsubuhm v[[REG3:[0-9]+]], v[[REG2]], v2

define <4 x i32> @test1_v4i32(<4 x i32> %a) {
        %tmp.1 = mul nsw <4 x i32> %a, <i32 16, i32 16, i32 16, i32 16>         ; <<4 x i32>> [#uses=1]
        ret <4 x i32> %tmp.1
}
; CHECK-LABEL: test1_v4i32:
; CHECK: vspltisw v[[REG1:[0-9]+]], 4
; CHECK-NOT: vmul
; CHECK-NEXT: vslw v[[REG2:[0-9]+]], v2, v[[REG1]]

define <4 x i32> @test2_v4i32(<4 x i32> %a) {
        %tmp.1 = mul nsw <4 x i32> %a, <i32 17, i32 17, i32 17, i32 17>         ; <<4 x i32>> [#uses=1]
        ret <4 x i32> %tmp.1
}
; CHECK-LABEL: test2_v4i32:
; CHECK: vspltisw v[[REG1:[0-9]+]], 4
; CHECK-NOT: vmul
; CHECK-NEXT: vslw v[[REG2:[0-9]+]], v2, v[[REG1]]
; CHECK-NEXT: vadduwm v[[REG3:[0-9]+]], v2, v[[REG2]]

define <4 x i32> @test3_v4i32(<4 x i32> %a) {
        %tmp.1 = mul nsw <4 x i32> %a, <i32 15, i32 15, i32 15, i32 15>         ; <<4 x i32>> [#uses=1]
        ret <4 x i32> %tmp.1
}
; CHECK-LABEL: test3_v4i32:
; CHECK: vspltisw v[[REG1:[0-9]+]], 4
; CHECK-NOT: vmul
; CHECK-NEXT: vslw v[[REG2:[0-9]+]], v2, v[[REG1]]
; CHECK-NEXT: vsubuwm v[[REG3:[0-9]+]], v[[REG2]], v2

; negtive constant

define <4 x i32> @test4_v4i32(<4 x i32> %a) {
        %tmp.1 = mul nsw <4 x i32> %a, <i32 -16, i32 -16, i32 -16, i32 -16>         ; <<4 x i32>> [#uses=1]
        ret <4 x i32> %tmp.1
}
; CHECK-LABEL: test4_v4i32:
; CHECK: vspltisw v[[REG1:[0-9]+]], 4
; CHECK-NOT: vmul
; CHECK-NEXT: vslw v[[REG2:[0-9]+]], v2, v[[REG1]]
; CHECK-P8-NEXT: xxlxor v[[REG3:[0-9]+]],
; CHECK-P8-NEXT: vsubuwm v{{[0-9]+}}, v[[REG3]], v[[REG2]]
; CHECK-P9-NEXT: vnegw v{{[0-9]+}}, v[[REG2]]

define <4 x i32> @test5_v4i32(<4 x i32> %a) {
        %tmp.1 = mul nsw <4 x i32> %a, <i32 -17, i32 -17, i32 -17, i32 -17>         ; <<4 x i32>> [#uses=1]
        ret <4 x i32> %tmp.1
}
; CHECK-LABEL: test5_v4i32:
; CHECK: vspltisw v[[REG1:[0-9]+]], 4
; CHECK-NOT: vmul
; CHECK-NEXT: vslw v[[REG2:[0-9]+]], v2, v[[REG1]]
; CHECK-NEXT: vadduwm v[[REG3:[0-9]+]], v2, v[[REG2]]
; CHECK-P8-NEXT: xxlxor v[[REG4:[0-9]+]],
; CHECK-P8-NEXT: vsubuwm v{{[0-9]+}}, v[[REG4]], v[[REG3]]
; CHECK-P9-NEXT: vnegw v{{[0-9]+}}, v[[REG3]]

define <4 x i32> @test6_v4i32(<4 x i32> %a) {
        %tmp.1 = mul nsw <4 x i32> %a, <i32 -15, i32 -15, i32 -15, i32 -15>         ; <<4 x i32>> [#uses=1]
        ret <4 x i32> %tmp.1
}
; CHECK-LABEL: test6_v4i32:
; CHECK: vspltisw v[[REG1:[0-9]+]], 4
; CHECK-NOT: vmul
; CHECK-NEXT: vslw v[[REG2:[0-9]+]], v2, v[[REG1]]
; CHECK-NEXT: vsubuwm v[[REG3:[0-9]+]], v2, v[[REG2]]

; boundary case

define <4 x i32> @test7_v4i32(<4 x i32> %a) {
        %tmp.1 = mul nsw <4 x i32> %a, <i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648> ; <<4 x i32>> [#uses=1]
        ret <4 x i32> %tmp.1
}
; CHECK-LABEL: test7_v4i32:
; CHECK-DAG: vspltisw v[[REG2:[0-9]+]], -16
; CHECK-DAG: vspltisw v[[REG3:[0-9]+]], 15
; CHECK-NEXT: vsubuwm v[[REG4:[0-9]+]], v[[REG3]], v[[REG2]]
; CHECK-NOT: vmul
; CHECK-NEXT: vslw v[[REG5:[0-9]+]], v2, v[[REG4]]

define <4 x i32> @test8_v4i32(<4 x i32> %a) {
        %tmp.1 = mul nsw <4 x i32> %a, <i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647> ; <<4 x i32>> [#uses=1]
        ret <4 x i32> %tmp.1
}
; CHECK-LABEL: test8_v4i32:
; CHECK-DAG: vspltisw v[[REG2:[0-9]+]], -16
; CHECK-DAG: vspltisw v[[REG3:[0-9]+]], 15
; CHECK-NEXT: vsubuwm v[[REG4:[0-9]+]], v[[REG3]], v[[REG2]]
; CHECK-NOT: vmul
; CHECK-NEXT: vslw v[[REG5:[0-9]+]], v2, v[[REG4]]
; CHECK-NEXT: vsubuwm v[[REG6:[0-9]+]], v[[REG5]], v2

define <2 x i64> @test1_v2i64(<2 x i64> %a) {
        %tmp.1 = mul nsw <2 x i64> %a, <i64 16, i64 16>         ; <<2 x i64>> [#uses=1]
        ret <2 x i64> %tmp.1
}
; CHECK-LABEL: test1_v2i64:
; CHECK-P8: lxvd2x vs[[REG1:[0-9]+]], 0, r{{[0-9]+}}
; CHECK-P8-NEXT: xxswapd v[[REG2:[0-9]+]], vs[[REG1]]
; CHECK-P9: lxvx v[[REG2:[0-9]+]], 0, r{{[0-9]+}}
; CHECK-NOT: vmul
; CHECK-NEXT: vsld v{{[0-9]+}}, v2, v[[REG2]]

define <2 x i64> @test2_v2i64(<2 x i64> %a) {
        %tmp.1 = mul nsw <2 x i64> %a, <i64 17, i64 17>         ; <<2 x i64>> [#uses=1]
        ret <2 x i64> %tmp.1
}

; CHECK-LABEL: test2_v2i64:
; CHECK-P8: lxvd2x vs[[REG1:[0-9]+]], 0, r{{[0-9]+}}
; CHECK-P8-NEXT: xxswapd v[[REG2:[0-9]+]], vs[[REG1]]
; CHECK-P9: lxvx v[[REG2:[0-9]+]], 0, r{{[0-9]+}}
; CHECK-NOT: vmul
; CHECK-NEXT: vsld v[[REG3:[0-9]+]], v2, v[[REG2]]
; CHECK-NEXT: vaddudm v{{[0-9]+}}, v2, v[[REG3]]

define <2 x i64> @test3_v2i64(<2 x i64> %a) {
        %tmp.1 = mul nsw <2 x i64> %a, <i64 15, i64 15>         ; <<2 x i64>> [#uses=1]
        ret <2 x i64> %tmp.1
}

; CHECK-LABEL: test3_v2i64:
; CHECK-P8: lxvd2x vs[[REG1:[0-9]+]], 0, r{{[0-9]+}}
; CHECK-P8-NEXT: xxswapd v[[REG2:[0-9]+]], vs[[REG1]]
; CHECK-P9: lxvx v[[REG2:[0-9]+]], 0, r{{[0-9]+}}
; CHECK-NOT: vmul
; CHECK-NEXT: vsld v[[REG3:[0-9]+]], v2, v[[REG2]]
; CHECK-NEXT: vsubudm v{{[0-9]+}}, v[[REG3]], v2

; negtive constant

define <2 x i64> @test4_v2i64(<2 x i64> %a) {
        %tmp.1 = mul nsw <2 x i64> %a, <i64 -16, i64 -16>         ; <<2 x i64>> [#uses=1]
        ret <2 x i64> %tmp.1
}

; CHECK-LABEL: test4_v2i64:
; CHECK-P8: lxvd2x vs[[REG1:[0-9]+]], 0, r{{[0-9]+}}
; CHECK-P8-NEXT: xxswapd v[[REG2:[0-9]+]], vs[[REG1]]
; CHECK-P9: lxvx v[[REG2:[0-9]+]], 0, r{{[0-9]+}}
; CHECK-NOT: vmul
; CHECK-NEXT: vsld v[[REG3:[0-9]+]], v2, v[[REG2]]
; CHECK-P8-NEXT: xxlxor v[[REG4:[0-9]+]],
; CHECK-P8-NEXT: vsubudm v{{[0-9]+}}, v[[REG4]], v[[REG3]]
; CHECK-P9-NEXT: vnegd v[[REG4:[0-9]+]], v[[REG3]]

define <2 x i64> @test5_v2i64(<2 x i64> %a) {
        %tmp.1 = mul nsw <2 x i64> %a, <i64 -17, i64 -17>         ; <<2 x i64>> [#uses=1]
        ret <2 x i64> %tmp.1
}

; CHECK-LABEL: test5_v2i64:
; CHECK-P8: lxvd2x vs[[REG1:[0-9]+]], 0, r{{[0-9]+}}
; CHECK-P8-NEXT: xxswapd v[[REG2:[0-9]+]], vs[[REG1]]
; CHECK-P9: lxvx v[[REG2:[0-9]+]], 0, r{{[0-9]+}}
; CHECK-NOT: vmul
; CHECK-NEXT: vsld v[[REG3:[0-9]+]], v2, v[[REG2]]
; CHECK-NEXT: vaddudm v[[REG4:[0-9]+]], v2, v[[REG3]]
; CHECK-P8-NEXT: xxlxor v[[REG5:[0-9]+]],
; CHECK-P8-NEXT: vsubudm v[[REG6:[0-9]+]], v[[REG5]], v[[REG4]]
; CHECK-P9-NEXT: vnegd v{{[0-9]+}}, v[[REG4]]

define <2 x i64> @test6_v2i64(<2 x i64> %a) {
        %tmp.1 = mul nsw <2 x i64> %a, <i64 -15, i64 -15>         ; <<2 x i64>> [#uses=1]
        ret <2 x i64> %tmp.1
}

; CHECK-LABEL: test6_v2i64:
; CHECK-P8: lxvd2x vs[[REG1:[0-9]+]], 0, r{{[0-9]+}}
; CHECK-P8-NEXT: xxswapd v[[REG2:[0-9]+]], vs[[REG1]]
; CHECK-P9: lxvx v[[REG2:[0-9]+]], 0, r{{[0-9]+}}
; CHECK-NOT: vmul
; CHECK-NEXT: vsld v[[REG3:[0-9]+]], v2, v[[REG2]]
; CHECK-NEXT: vsubudm v{{[0-9]+}}, v2, v[[REG3]]


; boundary case

define <2 x i64> @test7_v2i64(<2 x i64> %a) {
        %tmp.1 = mul nsw <2 x i64> %a, <i64 -9223372036854775808, i64 -9223372036854775808> ; <<2 x i64>> [#uses=1]
        ret <2 x i64> %tmp.1
}

; CHECK-LABEL: test7_v2i64:
; CHECK-P8: lxvd2x vs[[REG1:[0-9]+]], 0, r{{[0-9]+}}
; CHECK-P8-NEXT: xxswapd v[[REG2:[0-9]+]], vs[[REG1]]
; CHECK-P9: lxvx v[[REG2:[0-9]+]], 0, r{{[0-9]+}}
; CHECK-NOT: vmul
; CHECK-NEXT: vsld v[[REG4:[0-9]+]], v2, v[[REG2]]

define <2 x i64> @test8_v2i64(<2 x i64> %a) {
        %tmp.1 = mul nsw <2 x i64> %a, <i64 9223372036854775807, i64 9223372036854775807> ; <<2 x i64>> [#uses=1]
        ret <2 x i64> %tmp.1
}

; CHECK-LABEL: test8_v2i64:
; CHECK-P8: lxvd2x vs[[REG1:[0-9]+]], 0, r{{[0-9]+}}
; CHECK-P8-NEXT: xxswapd v[[REG2:[0-9]+]], vs[[REG1]]
; CHECK-P9: lxvx v[[REG2:[0-9]+]], 0, r{{[0-9]+}}
; CHECK-NOT: vmul
; CHECK-NEXT: vsld v[[REG3:[0-9]+]], v2, v[[REG2]]
; CHECK-NEXT: vsubudm v{{[0-9]+}}, v[[REG3]], v2
