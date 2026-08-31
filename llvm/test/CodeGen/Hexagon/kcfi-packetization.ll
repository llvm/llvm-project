;; Object-level checks of the KCFI check sequence, built by hand in
;; HexagonAsmPrinter::LowerKCFI_CHECK().
;;
;; -filetype=obj, not asm: the assembler re-inserts extenders when it parses
;; the text back in, so a missing one is invisible in -S output.  A truncated
;; hash makes every passing check trap; a truncated 0xBADC0FEE assembles as a
;; well-formed GP-relative load that does not fault at all.
;;
;; The { } delimiters are checked, not decoration: an extender only applies to
;; the instruction after it in slot order, so the grouping and order are the
;; property at issue.

; RUN: llc -mtriple=hexagon -filetype=obj < %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s
; RUN: llc -mtriple=hexagon -filetype=obj -mno-compound < %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s
; RUN: llc -mtriple=hexagon -filetype=obj -mno-pairing < %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s
; RUN: llc -mtriple=hexagon -filetype=obj -mattr=-packets < %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

;; The Hexagon Linux kernel builds with --disable-packetizer as a workaround
;; for an unrelated backend issue, so that combination has to keep working.
; RUN: llc -mtriple=hexagon -filetype=obj --disable-packetizer < %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

;; One trap-table entry per check and no others; a missing entry leaves a
;; trapping check looking like an ordinary misaligned access.  Checked here
;; because objcopy will not extract a section whose relocations point at .text.
; RUN: llc -mtriple=hexagon -filetype=obj < %s -o %t.o
; RUN: llvm-readobj -r %t.o | FileCheck %s --check-prefix=TRAPS
; TRAPS:         .rela.kcfi_traps {
; TRAPS-COUNT-6:   R_HEX_32_PCREL .text
; TRAPS-NEXT:    }

;; ...and across architecture versions, which differ in compound/duplex support.
; RUN: llc -mtriple=hexagon -mcpu=hexagonv62 -filetype=obj < %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s
; RUN: llc -mtriple=hexagon -mcpu=hexagonv68 -filetype=obj < %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s
; RUN: llc -mtriple=hexagon -mcpu=hexagonv73 -filetype=obj < %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s
; RUN: llc -mtriple=hexagon -mcpu=hexagonv79 -filetype=obj < %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

;; A hash that needs a constant extender (0xBC614E does not fit an s16).
define void @big_hash(ptr noundef %fp) {
; CHECK-LABEL: <big_hash>:
; CHECK:        { immext(#
; CHECK-NEXT:     r{{[0-9]+}} = ##0xbc614e
; CHECK-NEXT:     r{{[0-9]+}} = memw(r0+#-0x4) }
;; One packet whether or not the two compound, so do not pin the jump's line.
; CHECK-NEXT:   { p0 = cmp.eq(r{{[0-9]+}},r{{[0-9]+}})
; CHECK:          if (p0.new) jump:t {{.*}} }
; CHECK-NEXT:   { immext(#0xbadc0fc0)
; CHECK-NEXT:     r{{[0-9]+}}:{{[0-9]+}} = memd(##0xbadc0fee) }
; CHECK-NEXT:   { callr r0 }
  call void %fp() [ "kcfi"(i32 12345678) ]
  ret void
}

;; A small hash is extended too, so the compare sees the full 32 bits.
define void @small_hash(ptr noundef %fp) {
; CHECK-LABEL: <small_hash>:
; CHECK:        { immext(#
; CHECK-NEXT:     r{{[0-9]+}} = ##0x4d2
; CHECK-NEXT:     r{{[0-9]+}} = memw(r0+#-0x4) }
; CHECK:        { immext(#0xbadc0fc0)
; CHECK-NEXT:     r{{[0-9]+}}:{{[0-9]+}} = memd(##0xbadc0fee) }
  call void %fp() [ "kcfi"(i32 1234) ]
  ret void
}

;; Two checks in one function: each needs its own extenders and trap entry.
define void @two_checks(ptr noundef %f, ptr noundef %g) {
; CHECK-LABEL: <two_checks>:
; CHECK:        { immext(#
; CHECK-NEXT:     r{{[0-9]+}} = ##0xbc614e
; CHECK-NEXT:     r{{[0-9]+}} = memw({{r[0-9]+}}+#-0x4) }
; CHECK:        { immext(#0xbadc0fc0)
; CHECK-NEXT:     r{{[0-9]+}}:{{[0-9]+}} = memd(##0xbadc0fee) }
; CHECK:        { immext(#
; CHECK-NEXT:     r{{[0-9]+}} = ##0x4d2
; CHECK-NEXT:     r{{[0-9]+}} = memw({{r[0-9]+}}+#-0x4) }
; CHECK:        { immext(#0xbadc0fc0)
; CHECK-NEXT:     r{{[0-9]+}}:{{[0-9]+}} = memd(##0xbadc0fee) }
  call void %f() [ "kcfi"(i32 12345678) ]
  call void %g() [ "kcfi"(i32 1234) ]
  ret void
}

;; patchable-function-prefix moves the hash back; the offset still fits.
define void @prefixed(ptr noundef %fp) #0 {
; CHECK-LABEL: <prefixed>:
; CHECK:        { immext(#
; CHECK-NEXT:     r{{[0-9]+}} = ##0xbc614e
; CHECK-NEXT:     r{{[0-9]+}} = memw(r0+#-0xc) }
; CHECK:        { immext(#0xbadc0fc0)
; CHECK-NEXT:     r{{[0-9]+}}:{{[0-9]+}} = memd(##0xbadc0fee) }
  call void %fp() [ "kcfi"(i32 12345678) ]
  ret void
}

;; A prefix big enough to push the offset out of the 13-bit field (-4404), so
;; the load needs an extender too -- four instructions, exactly a full packet.
define void @big_prefix(ptr noundef %fp) #1 {
; CHECK-LABEL: <big_prefix>:
; CHECK:        { immext(#
; CHECK-NEXT:     r{{[0-9]+}} = ##0xbc614e
; CHECK-NEXT:     immext(#
; CHECK-NEXT:     r{{[0-9]+}} = memw(r0+##-0x1134) }
; CHECK:        { immext(#0xbadc0fc0)
; CHECK-NEXT:     r{{[0-9]+}}:{{[0-9]+}} = memd(##0xbadc0fee) }
  call void %fp() [ "kcfi"(i32 12345678) ]
  ret void
}

attributes #0 = { "patchable-function-prefix"="2" }
attributes #1 = { "patchable-function-prefix"="1100" }

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"kcfi", i32 1}
