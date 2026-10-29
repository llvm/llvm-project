; RUN: llc < %s -mtriple=sparc   -relocation-model=pic | FileCheck %s --check-prefix=V8
; RUN: llc < %s -mtriple=sparcv9 -relocation-model=pic | FileCheck %s --check-prefix=V9
; RUN: llc < %s -mtriple=sparcv9 -relocation-model=pic -filetype=obj | \
; RUN:   llvm-readobj -r - | FileCheck %s --check-prefix=RELOC

;; In the pic32 code model, addresses are loaded from the GOT with the
;; GOTDATA_OP sequence, so that the linker can replace the GOT load with a
;; GOT-relative address computation when the symbol binds locally.
;; https://github.com/llvm/llvm-project/issues/225923

@local = internal global i32 0
@ext = external global i32

define void @store_local(i32 %v) {
; V8-LABEL: store_local:
; V8:         sethi %gdop_hix22(local), [[HI:%[gilo][0-7]]]
; V8-NEXT:    xor [[HI]], %gdop_lox10(local), [[OFF:%[gilo][0-7]]]
; V8:         ld [[[GOT:%[gilo][0-7]]]+[[OFF]]], [[ADDR:%[gilo][0-7]]], %gdop(local)
; V8:         st %i0, [[[ADDR]]]
;
; V9-LABEL: store_local:
; V9:         sethi %gdop_hix22(local), [[HI:%[gilo][0-7]]]
; V9-NEXT:    xor [[HI]], %gdop_lox10(local), [[OFF:%[gilo][0-7]]]
; V9:         ldx [[[GOT:%[gilo][0-7]]]+[[OFF]]], [[ADDR:%[gilo][0-7]]], %gdop(local)
; V9:         st %i0, [[[ADDR]]]
  store volatile i32 %v, ptr @local
  ret void
}

define i32 @load_ext() {
; V8-LABEL: load_ext:
; V8:         sethi %gdop_hix22(ext), [[HI:%[gilo][0-7]]]
; V8-NEXT:    xor [[HI]], %gdop_lox10(ext), [[OFF:%[gilo][0-7]]]
; V8:         ld [{{%[gilo][0-7]}}+[[OFF]]], [[ADDR:%[gilo][0-7]]], %gdop(ext)
; V8:         ld [[[ADDR]]], %i0
;
; V9-LABEL: load_ext:
; V9:         sethi %gdop_hix22(ext), [[HI:%[gilo][0-7]]]
; V9-NEXT:    xor [[HI]], %gdop_lox10(ext), [[OFF:%[gilo][0-7]]]
; V9:         ldx [{{%[gilo][0-7]}}+[[OFF]]], [[ADDR:%[gilo][0-7]]], %gdop(ext)
; V9:         ld [[[ADDR]]], %i0
  %r = load i32, ptr @ext
  ret i32 %r
}

define float @const_pool() {
; V8-LABEL: const_pool:
; V8:         sethi %gdop_hix22(.LCPI2_0), [[HI:%[gilo][0-7]]]
; V8-NEXT:    xor [[HI]], %gdop_lox10(.LCPI2_0), [[OFF:%[gilo][0-7]]]
; V8:         ld [{{%[gilo][0-7]}}+[[OFF]]], [[ADDR:%[gilo][0-7]]], %gdop(.LCPI2_0)
; V8:         ld [[[ADDR]]], %f0
;
; V9-LABEL: const_pool:
; V9:         sethi %gdop_hix22(.LCPI2_0), [[HI:%[gilo][0-7]]]
; V9-NEXT:    xor [[HI]], %gdop_lox10(.LCPI2_0), [[OFF:%[gilo][0-7]]]
; V9:         ldx [{{%[gilo][0-7]}}+[[OFF]]], [[ADDR:%[gilo][0-7]]], %gdop(.LCPI2_0)
; V9:         ld [[[ADDR]]], %f0
  ret float 0x400921FA00000000
}

; RELOC:      Section ({{[0-9]+}}) .rela.text {
; RELOC:        R_SPARC_GOTDATA_OP_HIX22 local 0x0
; RELOC-NEXT:   R_SPARC_GOTDATA_OP_LOX10 local 0x0
; RELOC-NEXT:   R_SPARC_GOTDATA_OP local 0x0
; RELOC:        R_SPARC_GOTDATA_OP_HIX22 ext 0x0
; RELOC-NEXT:   R_SPARC_GOTDATA_OP_LOX10 ext 0x0
; RELOC-NEXT:   R_SPARC_GOTDATA_OP ext 0x0
; RELOC:        R_SPARC_GOTDATA_OP_HIX22 .LCPI2_0 0x0
; RELOC-NEXT:   R_SPARC_GOTDATA_OP_LOX10 .LCPI2_0 0x0
; RELOC-NEXT:   R_SPARC_GOTDATA_OP .LCPI2_0 0x0
; RELOC:      }
