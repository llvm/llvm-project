; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_100a -mattr=+ptx86 -o /dev/null
; RUN: %if ptxas-sm_100a && ptxas-isa-8.6 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_100a -mattr=+ptx86 | %ptxas-verify -arch=sm_100a %}

; Verify that inserting a bitcasted v2f16 into a vector that type-legalizes to
; v2i32 does not crash during DAG legalization on sm_100a+ (where v2i32 is a
; legal type). The DAG combiner would previously create an illegal v4f16
; vector_shuffle after type legalization.

define half @test(i64 %idx, <2 x half> %input) {
  %val = bitcast <2 x half> %input to <1 x i32>
  %scalar = extractelement <1 x i32> %val, i64 0
  %vec = insertelement <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>, i32 %scalar, i64 1

  ; prevent simplification to trigger the illegal pattern
  %cast = bitcast <4 x i32> %vec to <8 x half>
  %elt0 = extractelement <8 x half> %cast, i64 0
  %elt3 = extractelement <8 x half> %cast, i64 3
  %big = insertelement <16 x half> zeroinitializer, half %elt0, i64 1
  %big2 = insertelement <16 x half> %big, half %elt3, i64 %idx
  %big3 = insertelement <16 x half> %big2, half 0xH0000, i64 0
  %result = extractelement <16 x half> %big3, i64 %idx

  ret half %result
}
