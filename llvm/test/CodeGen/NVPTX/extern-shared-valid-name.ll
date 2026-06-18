; RUN: llc -mtriple=nvptx64-nvidia-cuda %s -o - | FileCheck %s

; Make sure the .extern .shared have the name legalization for nvptx applied

; CHECK: .extern .shared .align 4 .b8 anon_$_715df5978ff3b97886f2449b20e80729_$_0[]
@anon.715df5978ff3b97886f2449b20e80729.0 = external local_unnamed_addr addrspace(3) global [0 x i8], align 4
