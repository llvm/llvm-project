; RUN: llvm-ml -m32 -filetype=s %s /Fo - | FileCheck %s

.386p
.model flat

.code
mov eax, eax
end

; CHECK-NOT: 386p
; CHECK-NOT: model
; CHECK-NOT: flat
