; Check the HLX vector stack limit set with -hlx-vector-stack-limit and the
; "hlx-vector-stack-limit" function attribute, which overrides the option.
; Each function with allocas needs a 4096-byte vector stack frame.

; RUN: not llc -mtriple=hexagon -mcpu=hexagonv85 -mattr=+hlx -filetype=null \
; RUN:   < %s 2>&1 | FileCheck %s --implicit-check-not=error:
; RUN: not llc -mtriple=hexagon -mcpu=hexagonv85 -mattr=+hlx -filetype=null \
; RUN:   -hlx-vector-stack-limit=4096 < %s 2>&1 \
; RUN:   | FileCheck %s --implicit-check-not=error:
; RUN: not llc -mtriple=hexagon -mcpu=hexagonv85 -mattr=+hlx -filetype=null \
; RUN:   -hlx-vector-stack-limit=4095 < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=CHECK,LIMIT --implicit-check-not=error:

; LIMIT: error: {{.*}}HLX vector stack size (4096) exceeds limit (4095) in function 'no_attr'
; CHECK: error: {{.*}}HLX vector stack size (4096) exceeds limit (2048) in function 'attr_below'
; CHECK: error: {{.*}}HLX vector stack size (2048) exceeds limit (2047) in function 'caller_only'

define void @no_attr(<1024 x i16> %v) {
  %a = alloca <1024 x i16>, align 2048
  %b = alloca <1024 x i16>, align 2048
  store volatile <1024 x i16> %v, ptr %a, align 2048
  store volatile <1024 x i16> %v, ptr %b, align 2048
  ret void
}

define void @attr_equal(<1024 x i16> %v) #0 {
  %a = alloca <1024 x i16>, align 2048
  %b = alloca <1024 x i16>, align 2048
  store volatile <1024 x i16> %v, ptr %a, align 2048
  store volatile <1024 x i16> %v, ptr %b, align 2048
  ret void
}

define void @attr_below(<1024 x i16> %v) #1 {
  %a = alloca <1024 x i16>, align 2048
  %b = alloca <1024 x i16>, align 2048
  store volatile <1024 x i16> %v, ptr %a, align 2048
  store volatile <1024 x i16> %v, ptr %b, align 2048
  ret void
}

; The outgoing argument area counts: one 2048-byte HLX argument on the stack.
declare void @callee(<1024 x i16>, <1024 x i16>, <1024 x i16>, <1024 x i16>, <1024 x i16>, <1024 x i16>, <1024 x i16>, <1024 x i16>, <1024 x i16>, <1024 x i16>, <1024 x i16>, <1024 x i16>, <1024 x i16>)

define void @caller_only(<1024 x i16> %x) #2 {
  call void @callee(<1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x)
  ret void
}

attributes #0 = { "hlx-vector-stack-limit"="4096" }
attributes #1 = { "hlx-vector-stack-limit"="2048" }
attributes #2 = { "hlx-vector-stack-limit"="2047" }
