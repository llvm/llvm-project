; RUN: llc -emit-gnuas-syntax-on-zos=false < %s -mtriple=s390x-ibm-zos | \
; RUN: FileCheck --check-prefixes=CHECK %s

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 22, ptr @cfuncctor, ptr null }]
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 22, ptr @cfuncdtor, ptr null }]

; Check for presence of C_@@SQINIT:
; CHECK: 	.xtor.22
; CHECK: 	DC XL4'7FFF0017'
; Check direct relocation and low bit on ctor.
; CHECK:        DC AD(QD(stdin#S)+XL8'0')
; CHECK: 	DC XL8'0000000000000000'
; CHECK:        DC XL4'7FFF0017'
; CHECK:        DC XL8'0000000000000000'
; Check direct relocation and low bit on dtor.
; CHECK:        DC AD(QD(stdin#S)+XL8'16')

; Check for function descriptors in ADA section:
; CHECK:        * Offset 0 function descriptor of cfuncctor
; CHECK:        DC RD(cfuncctor)
; CHECK:        DC VD(cfuncctor)
; CHECK:        * Offset 16 function descriptor of cfuncdtor
; CHECK:        DC RD(cfuncdtor)
; CHECK:        DC VD(cfuncdtor)

define hidden void @cfuncctor() {
entry:
  ret void
}

define hidden void @cfuncdtor() {
entry:
  ret void
}
