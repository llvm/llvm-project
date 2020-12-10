; RUN: llvm-ml -filetype=s %s /Fo - /DT1=test1 /D T2=test2 | FileCheck %s

.code

t1:
  ret
; CHECK-NOT: t1:
; CHECK-LABEL: test1:
; CHECK-NOT: t1:

t2:
  ret
; CHECK-NOT: t2:
; CHECK-LABEL: test2:
; CHECK-NOT: t2:

t3:
ifdef t1
  xor eax, eax
endif
  ret
; CHECK-LABEL: t3:
; CHECK: xor eax, eax
; CHECK: ret

t4:
ifdef undefined
  xor eax, eax
elseifdef t2
  xor ebx, ebx
endif
  ret
; CHECK-LABEL: t4:
; CHECK-NOT: xor eax, eax
; CHECK: xor ebx, ebx
; CHECK: ret

end
