; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

target triple = "dxil-pc-shadermodel6.7-library"

; CHECK: ; Combined Shader Flags for Module
; CHECK-NEXT: ; Shader Flags Value: 0x00000000
; CHECK-NEXT: ;
; CHECK-NOT:  ; Note: shader requires additional functionality:
; CHECK-NOT:  ;       64-Bit integer
; CHECK-NOT:  ; Note: extra DXIL module flags:
; CHECK-NOT:  ;
; CHECK-NEXT: ; Shader Flags for Module Functions
; CHECK-NEXT: ; Function lifetimes : 0x00000000

define void @lifetimes() #0 {
  %a = alloca [4 x i32], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %a)
  call void @llvm.lifetime.end.p0(ptr nonnull %a)
  ret void
}

; Function Attrs: nounwind memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr) #1

; Function Attrs: nounwind memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr) #1

attributes #0 = { convergent norecurse nounwind "hlsl.export"}
attributes #1 = { nounwind memory(argmem: readwrite) }

; DXC: - Name:            SFI0
; DXC-NEXT:     Size:            8
; DXC-NOT:     Flags:
; DXC-NOT:         Int64Ops:        true
; DXC: ...
