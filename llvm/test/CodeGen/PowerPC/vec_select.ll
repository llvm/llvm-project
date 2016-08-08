; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-linux-gnu -mattr=+altivec | FileCheck %s

; CHECK: vsel_float
define <4 x float> @vsel_float(<4 x float> %v1, <4 x float> %v2) {
  %vsel = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x float> %v1, <4 x float> %v2
  ret <4 x float> %vsel
}
