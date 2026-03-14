; RUN: llc < %s -mtriple=x86_64-- -mcpu=corei7 -mattr=+sse4.1

; Make sure we are not crashing on this code.

define void @load_4_i8(ptr %k, ptr %y, ptr %A1, ptr %A0)  {
   %A = load <4 x i8>, ptr %k
   %B = load <4 x i8>, ptr %y
   %C = load <4 x double>, ptr %A0
   %D= load <4 x double>, ptr %A1
   %M = icmp uge <4 x i8> %A, %B
   %T = select <4 x i1> %M, <4 x double> %C, <4 x double> %D
   store <4 x double> %T, ptr undef
   ret void
}


define void @load_256_i8(ptr %k, ptr %y, ptr %A1, ptr %A0)  {
   %A = load <256 x i8>, ptr %k
   %B = load <256 x i8>, ptr %y
   %C = load <256 x double>, ptr %A0
   %D= load <256 x double>, ptr %A1
   %M = icmp uge <256 x i8> %A, %B
   %T = select <256 x i1> %M, <256 x double> %C, <256 x double> %D
   store <256 x double> %T, ptr undef
   ret void
}

