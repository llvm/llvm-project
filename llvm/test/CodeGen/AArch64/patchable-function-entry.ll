; RUN: llc -mtriple=aarch64 %s -o - | FileCheck --check-prefixes=CHECK,NOFSECT %s
; RUN: llc -mtriple=aarch64 -function-sections %s -o - | FileCheck --check-prefixes=CHECK,FSECT %s
; RUN: llc -mtriple=aarch64 -no-integrated-as %s -o - | FileCheck --check-prefix=NOLINK %s

; NOLINK-NOT: "awo"
; NOLINK-NOT: ,unique,0

define i32 @f0() "patchable-function-entry"="0" {
; CHECK-LABEL: f0:
; CHECK-NEXT: .Lfunc_begin0:
; CHECK-NOT:   nop
; CHECK:       mov w0, wzr
; CHECK:       .section __patchable_function_entries,"awo",@progbits,f0,unique,0
; CHECK-NEXT:  .p2align 3
; CHECK-NEXT:  .xword .Lfunc_begin0
  ret i32 0
}

define i32 @f1() "patchable-function-entry"="1" {
; CHECK-LABEL: f1:
; CHECK-NEXT: .Lfunc_begin1:
; CHECK:       nop
; CHECK-NEXT:  mov w0, wzr
; NOFSECT:     .section __patchable_function_entries,"awo",@progbits,f0,unique,0
; FSECT:       .section __patchable_function_entries,"awo",@progbits,f1,unique,1
; CHECK-NEXT:  .p2align 3
; CHECK-NEXT:  .xword .Lfunc_begin1
  ret i32 0
}

$f3 = comdat any
define void @f3() "patchable-function-entry"="3" comdat {
; CHECK-LABEL: f3:
; CHECK-NEXT: .Lfunc_begin2:
; CHECK-COUNT-3: nop
; CHECK-NEXT:  ret
; NOFSECT:     .section __patchable_function_entries,"aGwo",@progbits,f3,comdat,f3,unique,1
; FSECT:       .section __patchable_function_entries,"aGwo",@progbits,f3,comdat,f3,unique,2
; CHECK-NEXT:  .p2align 3
; CHECK-NEXT:  .xword .Lfunc_begin2
  ret void
}

$f5 = comdat any
define void @f5() "patchable-function-entry"="5" comdat {
; CHECK-LABEL: f5:
; CHECK-NEXT: .Lfunc_begin3:
; CHECK-COUNT-5: nop
; CHECK-NEXT:  sub sp, sp, #16
; NOFSECT      .section __patchable_function_entries,"aGwo",@progbits,f5,comdat,f5,unique,2
; FSECT:       .section __patchable_function_entries,"aGwo",@progbits,f5,comdat,f5,unique,3
; CHECK:       .p2align 3
; CHECK-NEXT:  .xword .Lfunc_begin3
  %frame = alloca i8, i32 16
  ret void
}
