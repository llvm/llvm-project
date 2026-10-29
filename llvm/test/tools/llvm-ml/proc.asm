; RUN: llvm-ml -filetype=s %s /Fo - | FileCheck %s
; RUN: llvm-ml64 -filetype=s %s /Fo - | FileCheck %s
; RUN: llvm-ml %s /Fo - | llvm-readobj --syms - | FileCheck %s --check-prefix=CHECK-OBJ
; RUN: llvm-ml64 %s /Fo - | llvm-readobj --syms - | FileCheck %s --check-prefix=CHECK-OBJ
; RUN: not llvm-ml -filetype=s %s /Fo /dev/null /DERR 2>&1 | FileCheck %s --check-prefix=CHECK-ERR --implicit-check-not=error:
; RUN: not llvm-ml64 -filetype=s %s /Fo /dev/null /DERR 2>&1 | FileCheck %s --check-prefix=CHECK-ERR --implicit-check-not=error:

.code

t1 PROC
  ret
t1 ENDP

; CHECK: t1:
; CHECK: ret

; CHECK-OBJ-LABEL: Name: t1
; CHECK-OBJ:       ComplexType: Function
; CHECK-OBJ-NEXT:  StorageClass: External

t2 PROC NEAR PUBLIC
  ret
t2 ENDP

; CHECK-OBJ-LABEL: Name: t2
; CHECK-OBJ:       ComplexType: Function
; CHECK-OBJ-NEXT:  StorageClass: External

t3 PROC PRIVATE
  ret
t3 ENDP

; CHECK-OBJ-LABEL: Name: t3
; CHECK-OBJ:       ComplexType: Function
; CHECK-OBJ-NEXT:  StorageClass: Static

PUBLIC t4
t4 PROC PRIVATE
  ret
t4 ENDP

; CHECK-OBJ-LABEL: Name: t4
; CHECK-OBJ:       ComplexType: Function
; CHECK-OBJ-NEXT:  StorageClass: External

ifdef ERR
; CHECK-ERR: :[[# @LINE + 1]]:16: error: expected newline in 'PROC' directive
t5 PROC PUBLIC NEAR
; CHECK-ERR: :[[# @LINE + 1]]:4: error: endp outside of procedure block
t5 ENDP

t6 PROC
; CHECK-ERR: :[[# @LINE + 1]]:9: error: expected newline in 'ENDP' directive
t6 ENDP extra
endif

END
