; XFAIL: target={{.*}}-aix{{.*}}
; RUN: llc < %s
; PR3899

@m = external global <2 x double>

define double @vector_ex() nounwind #0 {
       %v = load <2 x double>, ptr @m
       %x = extractelement <2 x double> %v, i32 1
       ret double %x
}

; Soft-float attribute so that targets that pay attention to soft float will
; make floating point types illegal and we'll exercise the legalizer code.
attributes #0 = { "use-soft-float" = "true" }
