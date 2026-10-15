; RUN: llubi --verbose < %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-p1:64:64:64:32"

define void @main() {
  %wide1 = inttoptr i64 4294967301 to ptr addrspace(1)
  %repr1 = ptrtoint ptr addrspace(1) %wide1 to i64
  %addr1 = ptrtoaddr ptr addrspace(1) %wide1 to i32

  %wide2 = inttoptr <2 x i64> <i64 4294967301, i64 4294967303> to <2 x ptr addrspace(1)>
  %repr2 = ptrtoint <2 x ptr addrspace(1)> %wide2 to <2 x i64>
  %addr2 = ptrtoaddr <2 x ptr addrspace(1)> %wide2 to <2 x i32>

  ret void
}

; CHECK: Entering function: main
; CHECK-NEXT:   %wide1 = inttoptr i64 4294967301 to ptr addrspace(1) => ptr 0x100000005 [nullary]
; CHECK-NEXT:   %repr1 = ptrtoint ptr addrspace(1) %wide1 to i64 => i64 4294967301
; CHECK-NEXT:   %addr1 = ptrtoaddr ptr addrspace(1) %wide1 to i32 => i32 5
; CHECK-NEXT:   %wide2 = inttoptr <2 x i64> <i64 4294967301, i64 4294967303> to <2 x ptr addrspace(1)> => { ptr 0x100000005 [nullary], ptr 0x100000007 [nullary] }
; CHECK-NEXT:   %repr2 = ptrtoint <2 x ptr addrspace(1)> %wide2 to <2 x i64> => { i64 4294967301, i64 4294967303 }
; CHECK-NEXT:   %addr2 = ptrtoaddr <2 x ptr addrspace(1)> %wide2 to <2 x i32> => { i32 5, i32 7 }
; CHECK-NEXT:   ret void
; CHECK-NEXT: Exiting function: main
