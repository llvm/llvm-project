; RUN: llc -mtriple=x86_64-pc-linux < %s
; PR9309

define <4 x double> @f_fu(<4 x float>) nounwind {
  %float2double.i = fpext <4 x float> %0 to <4 x double>
  ret <4 x double> %float2double.i
}
