; RUN: opt < %s -cost-model -analyze -mtriple=systemz-unknown -mcpu=z13 | FileCheck %s
;
; Costs for conversion of i1 vectors to vectors of double.

define <2 x double> @fun0(<2 x i8> %val1, <2 x i8> %val2) {
  %cmp = icmp eq <2 x i8> %val1, %val2
  %v = uitofp <2 x i1> %cmp to <2 x double>
  ret <2 x double> %v

; CHECK: fun0
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i8> %val1, %val2
; CHECK: cost of 5 for instruction:   %v = uitofp <2 x i1> %cmp to <2 x double>
}

define <2 x double> @fun1(<2 x i8> %val1, <2 x i8> %val2) {
  %cmp = icmp eq <2 x i8> %val1, %val2
  %v = sitofp <2 x i1> %cmp to <2 x double>
  ret <2 x double> %v

; CHECK: fun1
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i8> %val1, %val2
; CHECK: cost of 4 for instruction:   %v = sitofp <2 x i1> %cmp to <2 x double>
}

define <2 x double> @fun2(<2 x i64> %val1, <2 x i64> %val2) {
  %cmp = icmp eq <2 x i64> %val1, %val2
  %v = uitofp <2 x i1> %cmp to <2 x double>
  ret <2 x double> %v

; CHECK: fun2
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i64> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = uitofp <2 x i1> %cmp to <2 x double>
}

define <2 x double> @fun3(<2 x i64> %val1, <2 x i64> %val2) {
  %cmp = icmp eq <2 x i64> %val1, %val2
  %v = sitofp <2 x i1> %cmp to <2 x double>
  ret <2 x double> %v

; CHECK: fun3
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i64> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sitofp <2 x i1> %cmp to <2 x double>
}
