; RUN: llvm-ml -filetype=obj -m32 %s /Fo %t.32.obj
; RUN: llvm-readobj --relocations %t.32.obj | FileCheck %s --check-prefix=CHECK-I386

; RUN: llvm-ml -filetype=obj -m64 %s /Fo %t.64.obj
; RUN: llvm-readobj --relocations %t.64.obj | FileCheck %s --check-prefix=CHECK-AMD64

.data
sym1 dd 42

; CHECK-I386:      Relocations [
; CHECK-I386:        IMAGE_REL_I386_DIR32NB sym1
; CHECK-I386:      ]

; CHECK-AMD64:      Relocations [
; CHECK-AMD64:        IMAGE_REL_AMD64_ADDR32NB sym1
; CHECK-AMD64:      ]

rva_data dd IMAGEREL sym1

END
