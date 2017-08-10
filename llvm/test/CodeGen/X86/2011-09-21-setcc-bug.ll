; RUN: llc < %s -mtriple=x86_64-- -mcpu=corei7 -mattr=+sse4.1

; Make sure we are not crashing on this code.

define void @load_4_i8(<4 x i8>* %k, <4 x i8>* %y, <4 x double>* %A1, <4 x double>* %A0)  {
   %A = load <4 x i8>, <4 x i8>* %k
   %B = load <4 x i8>, <4 x i8>* %y
   %C = load <4 x double>, <4 x double>* %A0
   %D= load <4 x double>, <4 x double>* %A1
   %M = icmp uge <4 x i8> %A, %B
   %T = select <4 x i1> %M, <4 x double> %C, <4 x double> %D
   store <4 x double> %T, <4 x double>* undef
   ret void
}


define void @load_256_i8(<256 x i8>* %k, <256 x i8>* %y, <256 x double>* %A1, <256 x double>* %A0)  {
   %A = load <256 x i8>, <256 x i8>* %k
   %B = load <256 x i8>, <256 x i8>* %y
   %C = load <256 x double>, <256 x double>* %A0
   %D= load <256 x double>, <256 x double>* %A1
   %M = icmp uge <256 x i8> %A, %B
   %T = select <256 x i1> %M, <256 x double> %C, <256 x double> %D
   store <256 x double> %T, <256 x double>* undef
   ret void
}

