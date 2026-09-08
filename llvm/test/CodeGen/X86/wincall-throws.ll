; RUN: llc -mtriple=x86_64apx-unknown-windows-msvc < %s | FileCheck %s

; WinCall + herbceptions: a throws function keeps the regular wincall payload
; rules (scalar returns in RAX/RDX), while the success/failure discriminant is
; carried out-of-band in the carry flag (CF).  The callee sets CF before the
; ret (addb $-1 trick), and the caller reads CF right after the call (setb)
; before CALLSEQ_END can clobber EFLAGS.

; Success: discriminant false -> CF clear at return.
define x86_wincallcc { i64, i1 } @wc_ret_success(i64 %x) #0 {
; CHECK-LABEL: wc_ret_success:
; CHECK:       movq %rcx, %rax
; CHECK-NEXT:  xorl %ecx, %ecx
; CHECK-NEXT:  addb $-1, %cl
; CHECK:       retq
entry:
  %r.i = insertvalue { i64, i1 } poison, i64 %x, 0
  %r = insertvalue { i64, i1 } %r.i, i1 false, 1
  ret { i64, i1 } %r
}

; Error: discriminant true -> CF set at return.
define x86_wincallcc { i64, i1 } @wc_ret_error(i64 %x) #0 {
; CHECK-LABEL: wc_ret_error:
; CHECK:       movq %rcx, %rax
; CHECK-NEXT:  movb $1, %cl
; CHECK-NEXT:  addb $-1, %cl
; CHECK:       retq
entry:
  %r.i = insertvalue { i64, i1 } poison, i64 %x, 0
  %r = insertvalue { i64, i1 } %r.i, i1 true, 1
  ret { i64, i1 } %r
}

; Two-word std::error-style payload returned in RAX (domain) and RDX (code),
; discriminant in CF.
define x86_wincallcc { i64, i64, i1 } @wc_ret_stderror(i64 %dom, i64 %code) #0 {
; CHECK-LABEL: wc_ret_stderror:
; CHECK:       movq %rcx, %rax
; CHECK-NEXT:  xorl %ecx, %ecx
; CHECK-NEXT:  addb $-1, %cl
; CHECK-NOT:   addq $-1, %rdx
; CHECK:       retq
entry:
  %r0 = insertvalue { i64, i64, i1 } poison, i64 %dom, 0
  %r1 = insertvalue { i64, i64, i1 } %r0, i64 %code, 1
  %r = insertvalue { i64, i64, i1 } %r1, i1 false, 2
  ret { i64, i64, i1 } %r
}

; The caller reads CF immediately after the call.
define x86_wincallcc i64 @wc_call_and_select(i64 %x) nounwind {
; CHECK-LABEL: wc_call_and_select:
; CHECK:       callq wc_ret_error
; CHECK-NEXT:  setb %cl
; CHECK-NEXT:  testb $1, %cl
; CHECK-NEXT:  movl $100, %ecx
; CHECK-NEXT:  cmovneq %rcx, %rax
; CHECK:       retq
entry:
  %c = call x86_wincallcc { i64, i1 } @wc_ret_error(i64 %x)
  %val = extractvalue { i64, i1 } %c, 0
  %disc = extractvalue { i64, i1 } %c, 1
  %sel = select i1 %disc, i64 100, i64 %val
  ret i64 %sel
}

attributes #0 = { throws }
