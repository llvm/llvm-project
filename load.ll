
; load.ll
; Isolates the LLVM IR `load` instruction for RISC-V instruction selection.
;
; What to look for in the asm:
;   i64                  -> ld
;   i32 (plain)          -> lw   (LW sign-extends to XLEN=64 on rv64)
;   i32 -> sext i64      -> lw
;   i32 -> zext i64      -> lwu
;   i8  -> sext i64      -> lb
;   i8  -> zext i64      -> lbu
; The base address comes in a0; the loaded value is returned in a0.

define i64 @load_i64(ptr %p) {
  %v = load i64, ptr %p, align 8
  ret i64 %v
}
