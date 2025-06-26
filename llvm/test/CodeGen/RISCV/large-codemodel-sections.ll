; RUN: llc -mtriple=riscv64 -mattr=+f,+zfh -target-abi=lp64f -code-model=large -verify-machineinstrs < %s \
; RUN:   -filetype=obj -o - | llvm-objdump -htr - \
; RUN:   | FileCheck %s -check-prefix=RV64I
; RUN: llc -mtriple=riscv64 -mattr=+zfinx,+zhinx -target-abi=lp64 -code-model=large -verify-machineinstrs < %s \
; RUN:   -filetype=obj -o - | llvm-objdump -htr - \
; RUN:   | FileCheck %s -check-prefix=RV64I


;; This tests that we are lowering large code model constants into `.text`
;; constant pools, so that accessing them is close to `.text`, rather than
;; far away in `.data`. The other choices are `.rodata` and `.data.rel.ro`,
;; both of which may not be close enough to `.text` to be referenced.

; RV64I-LABEL: Sections:
; RV64I: .text 000000b4
; RV64I: .rela.text 00000060
; RV64I: .bss 00000010
; RV64I: .data 00000002

; RV64I-LABEL: SYMBOL TABLE:
; RV64I: g O .bss 0000000000000004 G
; RV64I: g F .text {{[0-9a-f]+}} lower_global
; RV64I: g O .bss 0000000000000008 addr
; RV64I: g F .text {{[0-9a-f]+}} lower_blockaddress
; RV64I: g F .text {{[0-9a-f]+}} lower_blockaddress_displ
; RV64I: g F .text {{[0-9a-f]+}} lower_constantpool
; RV64I: w *UND* 0000000000000000 W
; RV64I: g F .text {{[0-9a-f]+}} lower_extern_weak
; RV64I: g O .data 0000000000000002 X
; RV64I: g F .text {{[0-9a-f]+}} lower_global_half

; RV64I-LABEL: RELOCATION RECORDS FOR [.text]:
; RV64I: R_RISCV_64 G
; RV64I: R_RISCV_64 addr
; RV64I: R_RISCV_64 W
; RV64I: R_RISCV_64 X


; Check lowering of globals
@G = global i32 0
define i32 @lower_global(i32 %a) nounwind {
  %1 = load volatile i32, ptr @G
  ret i32 %1
}

; Check lowering of blockaddresses
@addr = global ptr null
define void @lower_blockaddress() nounwind {
  store volatile ptr blockaddress(@lower_blockaddress, %block), ptr @addr
  ret void

block:
  unreachable
}

; Check lowering of blockaddress that forces a displacement to be added
define signext i32 @lower_blockaddress_displ(i32 signext %w) nounwind {
entry:
  %x = alloca ptr, align 8
  store ptr blockaddress(@lower_blockaddress_displ, %test_block), ptr %x, align 8
  %cmp = icmp sgt i32 %w, 100
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %addr = load ptr, ptr %x, align 8
  br label %indirectgoto

if.end:
  br label %return

test_block:
  br label %return

return:
  %retval = phi i32 [ 3, %if.end ], [ 4, %test_block ]
  ret i32 %retval

indirectgoto:
  indirectbr ptr %addr, [ label %test_block ]
}

; Check lowering of constantpools
define float @lower_constantpool(float %a) nounwind {
  %1 = fadd float %a, 1.000244140625
  ret float %1
}

; Check lowering of extern_weaks
@W = extern_weak global i32

define i32 @lower_extern_weak(i32 %a) nounwind {
  %1 = load volatile i32, ptr @W
  ret i32 %1
}

@X = global half 1.5

define half @lower_global_half(half %a) nounwind {
  %b = load half, ptr @X
  %1 = fadd half %a, %b
  ret half %1
}
