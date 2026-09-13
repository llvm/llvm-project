; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=skylake-avx512 -passes='print<cost-model>' -cost-kind=all -disable-output 2>&1 | FileCheck %s --check-prefixes=COST,COST64,DQ
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=skylake-avx512 -mattr=-avx512dq -passes='print<cost-model>' -cost-kind=all -disable-output 2>&1 | FileCheck %s --check-prefixes=COST,COST64,NODQ
; RUN: opt < %s -mtriple=i386-unknown-linux-gnu -mcpu=skylake-avx512 -passes='print<cost-model>' -cost-kind=all -disable-output 2>&1 | FileCheck %s --check-prefixes=COST,COST32,DQ

; Cost tuples below are reciprocal throughput, code size, latency and size+latency.

define <16 x i1> @lane16_bw_0(<16 x i1> %src) #0 {
; COST-LABEL: 'lane16_bw_0'
; COST: Cost Model: Found costs of RThru:1 CodeSize:3 Lat:5 SizeLat:3 for: {{ *}}%s = shufflevector
  %s = shufflevector <16 x i1> %src, <16 x i1> poison, <16 x i32> splat (i32 0)
  ret <16 x i1> %s
}

define <16 x i1> @lane16_bw_1(<16 x i1> %src) #0 {
; COST-LABEL: 'lane16_bw_1'
; COST: Cost Model: Found costs of RThru:2 CodeSize:4 Lat:8 SizeLat:5 for: {{ *}}%s = shufflevector
  %s = shufflevector <16 x i1> %src, <16 x i1> poison, <16 x i32> splat (i32 1)
  ret <16 x i1> %s
}

define <16 x i1> @lane16_dword_0(<16 x i1> %src) #1 {
; COST-LABEL: 'lane16_dword_0'
; DQ: Cost Model: Found costs of RThru:1 CodeSize:3 Lat:5 SizeLat:3 for: {{ *}}%s = shufflevector
; NODQ: Cost Model: Found costs of RThru:2 CodeSize:3 Lat:8 SizeLat:3 for: {{ *}}%s = shufflevector
  %s = shufflevector <16 x i1> %src, <16 x i1> poison, <16 x i32> splat (i32 0)
  ret <16 x i1> %s
}

define <64 x i1> @lane64_no512_0(<64 x i1> %src) #2 {
; COST-LABEL: 'lane64_no512_0'
; COST64: Cost Model: Found costs of RThru:1 CodeSize:5 Lat:7 SizeLat:5 for: {{ *}}%s = shufflevector
; COST32: Cost Model: Found costs of RThru:2 CodeSize:5 Lat:10 SizeLat:5 for: {{ *}}%s = shufflevector
  %s = shufflevector <64 x i1> %src, <64 x i1> poison, <64 x i32> splat (i32 0)
  ret <64 x i1> %s
}

define <64 x i1> @lane64_no512_17(<64 x i1> %src) #2 {
; COST-LABEL: 'lane64_no512_17'
; COST64: Cost Model: Found costs of RThru:2 CodeSize:6 Lat:11 SizeLat:6 for: {{ *}}%s = shufflevector
; COST32: Cost Model: Found costs of RThru:3 CodeSize:6 Lat:14 SizeLat:6 for: {{ *}}%s = shufflevector
  %s = shufflevector <64 x i1> %src, <64 x i1> poison, <64 x i32> splat (i32 17)
  ret <64 x i1> %s
}

define <64 x i1> @lane64_512_0(<64 x i1> %src) #0 {
; COST-LABEL: 'lane64_512_0'
; COST: Cost Model: Found costs of RThru:1 CodeSize:3 Lat:5 SizeLat:3 for: {{ *}}%s = shufflevector
  %s = shufflevector <64 x i1> %src, <64 x i1> poison, <64 x i32> splat (i32 0)
  ret <64 x i1> %s
}

define <64 x i1> @lane64_512_17(<64 x i1> %src) #0 {
; COST-LABEL: 'lane64_512_17'
; COST: Cost Model: Found costs of RThru:2 CodeSize:5 Lat:6 SizeLat:7 for: {{ *}}%s = shufflevector
  %s = shufflevector <64 x i1> %src, <64 x i1> poison, <64 x i32> splat (i32 17)
  ret <64 x i1> %s
}

attributes #0 = { "prefer-vector-width"="256" "min-legal-vector-width"="512" }
attributes #1 = { "prefer-vector-width"="512" "min-legal-vector-width"="512" }
attributes #2 = { "prefer-vector-width"="256" "min-legal-vector-width"="0" }
