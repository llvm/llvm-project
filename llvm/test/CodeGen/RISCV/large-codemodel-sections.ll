; RUN: llc -mtriple=riscv64 -mattr=+f,+zfh -target-abi=lp64f -code-model=large -verify-machineinstrs < %s \
; RUN:   -filetype=obj -o - | llvm-readobj -r - \
; RUN:   | FileCheck %s -check-prefix=RV64I
; RUN: llc -mtriple=riscv64 -mattr=+zfinx,+zhinx -target-abi=lp64 -code-model=large -verify-machineinstrs < %s \
; RUN:   -filetype=obj -o - | llvm-readobj -r - \
; RUN:   | FileCheck %s -check-prefix=RV64I


;; This tests that we are lowering large code model constants into `.text`
;; constant pools, so that accessing them is close to `.text`, rather than
;; far away in `.data`. The other choices are `.rodata` and `.data.rel.ro`,
;; both of which may not be close enough to `.text` to be referenced.
;;
;; The test uses `readobj` to check that there are relocations against the
;; `.text` section for these addresses. This is not compatible with PIC,
;; just like the rest of the large code model.

; RV64I: Section (3) .rela.text {
; RV64I-NEXT: R_RISCV_64 G 0x0
; RV64I-NEXT: R_RISCV_64 addr 0x0
; RV64I-NEXT: R_RISCV_64 W 0x0
; RV64I-NEXT: R_RISCV_64 X 0x0

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
