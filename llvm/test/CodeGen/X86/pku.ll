; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl --show-mc-encoding -verify-machineinstrs | FileCheck %s
declare i32 @llvm.x86.rdpkru()
declare void @llvm.x86.wrpkru(i32)

define void @test_x86_wrpkru(i32 %src) {
; CHECK-LABEL: test_x86_wrpkru:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    xorl    %ecx, %ecx
; CHECK-NEXT:    xorl    %edx, %edx
; CHECK-NEXT:    movl    %edi, %eax
; CHECK-NEXT:    wrpkru
; CHECK-NEXT:    retq
  call void @llvm.x86.wrpkru(i32 %src) 
  ret void
}

define i32 @test_x86_rdpkru() {
; CHECK-LABEL: test_x86_rdpkru:
; CHECK:      ## %bb.0:
; CHECK-NEXT: xorl    %ecx, %ecx
; CHECK-NEXT: rdpkru
; CHECK-NEXT: retq
  %res = call i32 @llvm.x86.rdpkru() 
  ret i32 %res 
}
