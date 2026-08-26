; RUN: llc -mtriple=x86_64-unknown-linux %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=x86_64-unknown-linux %s -o - --filetype=obj | llvm-readelf -s - | FileCheck %s --check-prefix=SYM
; RUN: llc -mtriple=x86_64-unknown-linux %s -o - --filetype=obj | llvm-objdump -d - | FileCheck %s --check-prefix=OBJ

;; Check "tail-pad-to-size" fills the ends of functions @a and @b with
;; "tail-pad-value" value bytes. @c is larger than the "tail-pad-to-size" value
;; so shouldn't be affected by either attribute.

;; Check the fill bytes are included in function size (unlike alignment
;; padding).
; SYM:    Num:    Value          Size Type    Bind   Vis       Ndx Name
; SYM:  [[#]]: 0000000000000000     5 FUNC    GLOBAL HIDDEN  [[#]] a
; SYM:  [[#]]: 0000000000000010     5 FUNC    GLOBAL HIDDEN  [[#]] b
; SYM:  [[#]]: 0000000000000018     6 FUNC    GLOBAL HIDDEN  [[#]] c

; ASM:      a:                                      # @a
; ASM:              retq
; ASM-NEXT: .Ltail_pad_start0:
; ASM-NEXT:         .zero (5-(.Ltail_pad_start0-a))&((5-(.Ltail_pad_start0-a))>=0),144
; ASM-NEXT: .Lfunc_end0:
; ASM-NEXT:         .size   a, .Lfunc_end0-a
;
; OBJ: 0000000000000000 <a>:
; OBJ-NEXT: 0: c3                            retq
;; tail pdading:
; OBJ-NEXT: 1: 90                            nop
; OBJ-NEXT: 2: 90                            nop
; OBJ-NEXT: 3: 90                            nop
; OBJ-NEXT: 4: 90                            nop
;; alignment padding for @b:
; OBJ-NEXT: 5: 66 2e 0f 1f 84 00 00 00 00 00 nopw    %cs:(%rax,%rax)
; OBJ-NEXT: f: 90                            nop
define hidden void @a() #0 {
entry:
  ret void
}

; ASM: b:                                      # @b
; ASM:        movl    %edi, %eax
; ASM-NEXT:   retq
; ASM-NEXT: .Ltail_pad_start1:
; ASM-NEXT:         .zero   (5-(.Ltail_pad_start1-b))&((5-(.Ltail_pad_start1-b))>=0),144
; ASM-NEXT: .Lfunc_end1:
; ASM-NEXT:        .size   b, .Lfunc_end1-b
;
; OBJ: 0000000000000010 <b>:
; OBJ-NEXT: 10: 89 f8                         movl    %edi, %eax
; OBJ-NEXT: 12: c3                            retq
;; tail padding:
; OBJ-NEXT: 13: 90                            nop
; OBJ-NEXT: 14: 90                            nop
;; alignment padding for @c:
; OBJ-NEXT: 15: 0f 1f 00                      nopl    (%rax)
define hidden i32 @b(i32 %a) align 16 #0 {
entry:
  ret i32 %a
}

; ASM: c:                                      # @c
; ASM:        movl    %edi, %eax
; ASM-NEXT:   imull   %esi, %eax
; ASM-NEXT:   retq
; ASM-NEXT: .Ltail_pad_start2:
; ASM-NEXT:         .zero   (5-(.Ltail_pad_start2-c))&((5-(.Ltail_pad_start2-c))>=0),144
; ASM-NEXT: .Lfunc_end2:
; ASM-NEXT:         .size   c, .Lfunc_end2-c
;
; OBJ: 0000000000000018 <c>:
; OBJ-NEXT: 18: 89 f8                         movl    %edi, %eax
; OBJ-NEXT: 1a: 0f af c6                      imull   %esi, %eax
; OBJ-NEXT: 1d: c3                            retq
;; No tail or alignment padding follows.
; OBJ-NOT: nop
define hidden i32 @c(i32 %a, i32 %b) #0 {
entry:
  %c = mul i32 %a, %b
  ret i32 %c
}

attributes #0 = { "tail-pad-to-size"="5" "tail-pad-value"="144" }
