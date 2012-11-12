;RUN: llc < %s -march=x86 -mcpu=yonah -mattr=+sse2,-sse41 | FileCheck %s

;CHECK: @max
;CHECK: cmplepd
;CHECK: ret

define <2 x double> @max(<2 x double> %x, <2 x double> %y) {
   %max_is_x = fcmp oge <2 x double> %x, %y
   %max = select <2 x i1> %max_is_x, <2 x double> %x, <2 x double> %y
   ret <2 x double> %max
}

