; RUN: llc -mtriple=sparc64-linux-gnu -verify-machineinstrs -o - %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=sparc64-linux-gnu -verify-machineinstrs -stop-after=finalize-isel -o - %s | FileCheck %s --check-prefix=NAMES
; RUN: llc -mtriple=sparc64-linux-gnu -verify-machineinstrs -stop-after=prolog-epilog -simplify-mir -o - %s | FileCheck %s --check-prefix=LIVEINS

; NAMES-LABEL: name: {{ *}}read_o6
; NAMES: {{%[0-9]+}}:i64regs = COPY $o6
; ASM-LABEL: read_o6:
; ASM: save %sp
define i64 @read_o6() {
entry:
  %value = call i64 @llvm.read_register.i64(metadata !0)
  ret i64 %value
}

; LIVEINS-LABEL: name: {{ *}}read_i7
; LIVEINS-NOT: liveins:
; LIVEINS: bb.0.entry:
; LIVEINS-NOT: liveins:
; LIVEINS: $o0 = COPY $o7
define i64 @read_i7() {
entry:
  %value = call i64 @llvm.read_register.i64(metadata !1)
  ret i64 %value
}

; LIVEINS-LABEL: name: {{ *}}read_i7_nonentry
; LIVEINS: bb.0.entry:
; LIVEINS: liveins: $o0, $o1{{$}}
; LIVEINS: bb.{{[0-9]+}}.read:
; LIVEINS-NOT: liveins:
; LIVEINS: $o0 = COPY $o7
define i64 @read_i7_nonentry(i64 %zero_value, i1 %cond) {
entry:
  br i1 %cond, label %read, label %zero

read:
  %value = call i64 @llvm.read_register.i64(metadata !1)
  ret i64 %value

zero:
  ret i64 %zero_value
}

; NAMES-LABEL: name: {{ *}}read_i7_nonleaf
; NAMES: {{%[0-9]+}}:i64regs = COPY $i7
; ASM-LABEL: read_i7_nonleaf:
; ASM: save %sp
; ASM: call callee
; ASM: ret
define i64 @read_i7_nonleaf() {
entry:
  call void @callee()
  %value = call i64 @llvm.read_register.i64(metadata !1)
  ret i64 %value
}

; ASM-LABEL: reserved_l0_unused:
; ASM: save %sp
; ASM: ret
define void @reserved_l0_unused() #0 {
entry:
  ret void
}

declare i64 @llvm.read_register.i64(metadata)
declare void @callee()

attributes #0 = { "target-features"="+reserve-l0" }

!0 = !{!"o6"}
!1 = !{!"i7"}
