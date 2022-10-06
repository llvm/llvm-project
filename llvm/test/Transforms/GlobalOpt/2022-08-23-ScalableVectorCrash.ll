; RUN: opt -globalopt < %s

; Ensure we don't ICE by trying to optimize a scalable vector load of a global
; variable.

%struct.xxx = type <{ [96 x i8] }>

@.bss = internal unnamed_addr global %struct.xxx zeroinitializer, align 32

define dso_local void @foo() local_unnamed_addr align 16 #0 {
L.entry:
  store <vscale x 2 x double> zeroinitializer, ptr @.bss, align 1
  %0 = load <vscale x 2 x double>, ptr @.bss, align 8
  unreachable
}
