; RUN: llc -mtriple=sparc -verify-machineinstrs -stop-after=prolog-epilog < %s | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=sparc -verify-machineinstrs < %s | FileCheck %s

;; A leaf procedure remaps %i7 to %o7. A read of the reserved %i7 through
;; llvm.read_register has no live-in, so the remapped %o7 has to be made
;; live-in explicitly, including in blocks other than the entry block.

; MIR-LABEL: name: read_i7_entry
; MIR:       bb.0.entry:
; MIR-NEXT:    liveins: $o7
; MIR:         $o0 = COPY $o7
; MIR-NEXT:    RETL 8, implicit $o0

; CHECK-LABEL: read_i7_entry:
; CHECK-NOT:   save
; CHECK:       retl
; CHECK-NEXT:  mov %o7, %o0
define i32 @read_i7_entry() nounwind {
entry:
  %ra = call i32 @llvm.read_register.i32(metadata !0)
  ret i32 %ra
}

; MIR-LABEL: name: read_i7_nonentry
; MIR:       bb.0.entry:
; MIR:         liveins: $o0, $o7
; MIR:       bb.1.then:
; MIR:         liveins: $o7
; MIR:         $o0 = COPY $o7
; MIR:       bb.2.exit:
; MIR:         liveins: $o0, $o7

; CHECK-LABEL: read_i7_nonentry:
; CHECK-NOT:   save
; CHECK:       %then
; CHECK-NEXT:  mov %o7, %o0
; CHECK:       retl
define i32 @read_i7_nonentry(i1 %c) nounwind {
entry:
  br i1 %c, label %then, label %exit

then:
  %ra = call i32 @llvm.read_register.i32(metadata !0)
  br label %exit

exit:
  %r = phi i32 [ %ra, %then ], [ 0, %entry ]
  ret i32 %r
}

declare i32 @llvm.read_register.i32(metadata)

!0 = !{!"i7"}
