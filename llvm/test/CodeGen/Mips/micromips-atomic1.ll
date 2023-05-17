; RUN: llc -march=mipsel -filetype=obj --disable-machine-licm -mattr=micromips < %s -o - \
; RUN:   | llvm-objdump --no-print-imm-hex --no-show-raw-insn -d - | FileCheck %s --check-prefix=MICROMIPS

; Use llvm-objdump to check wheter the encodings of microMIPS atomic instructions are correct.
; While emitting assembly files directly when in microMIPS mode, it is possible to emit a mips32r2
; instruction instead of microMIPS instruction, and since many mips32r2 and microMIPS
; instructions have identical assembly formats, invalid instruction cannot be detected.

@y = common global i8 0, align 1

define signext i8 @AtomicLoadAdd8(i8 signext %incr) nounwind {
entry:
  %0 = atomicrmw add ptr @y, i8 %incr monotonic
  ret i8 %0

; MICROMIPS:     ll      ${{[0-9]+}}, 0(${{[0-9]+}})
; MICROMIPS:     sc      ${{[0-9]+}}, 0(${{[0-9]+}})
}

define signext i8 @AtomicCmpSwap8(i8 signext %oldval, i8 signext %newval) nounwind {
entry:
  %pair0 = cmpxchg ptr @y, i8 %oldval, i8 %newval monotonic monotonic
  %0 = extractvalue { i8, i1 } %pair0, 0
  ret i8 %0

; MICROMIPS:     ll      ${{[0-9]+}}, 0(${{[0-9]+}})
; MICROMIPS:     sc      ${{[0-9]+}}, 0(${{[0-9]+}})
}
