; RUN: llc < %s -soft-float
; PR3899

@m = external global <2 x double>

define double @vector_ex() nounwind {
       %v = load <2 x double>* @m
       %x = extractelement <2 x double> %v, i32 1
       ret double %x
}
