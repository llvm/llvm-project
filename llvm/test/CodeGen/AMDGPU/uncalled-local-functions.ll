; RUN: llc -O0 -march=amdgcn -mcpu=gfx90a < %s | FileCheck %s
; REQUIRES: asserts

@alias = internal alias i32, i32* @aliased_internal_func
@alias_taken = internal alias i32, i32* @aliased_taken_func

; CHECK-NOT: internal_func
define internal i32 @internal_func() {
  ret i32 0
}

; CHECK-NOT: private_func
define private i32 @private_func() {
  ret i32 0
}

; CHECK-NOT: aliased_internal_func
define internal i32 @aliased_internal_func() {
  ret i32 0
}

; CHECK-LABEL: take_alias_addr
; CHECK:      Function info:
; CHECK-NEXT: codeLenInByte = 60
; CHECK-NEXT: NumSgprs: 37
; CHECK-NEXT: NumVgprs: 1
; CHECK-NEXT: NumAgprs: 0
; CHECK-NEXT: TotalNumVgprs: 1
; CHECK-NEXT: ScratchSize: 16
; CHECK-NEXT: MemoryBound: 0
define void @take_alias_addr() {
  %addr_loc = alloca ptr, addrspace(5)
  store ptr @alias_taken, ptr addrspace(5) %addr_loc
  ret void
}

; CHECK: aliased_taken_func
; CHECK:      Function info:
; CHECK-NEXT: codeLenInByte = 12
; CHECK-NEXT: NumSgprs: 36
; CHECK-NEXT: NumVgprs: 1
; CHECK-NEXT: NumAgprs: 0
; CHECK-NEXT: TotalNumVgprs: 1
; CHECK-NEXT: ScratchSize: 0
; CHECK-NEXT: MemoryBound: 0
define internal i32 @aliased_taken_func() {
  ret i32 0
}

; CHECK-LABEL: addr_taken
; CHECK:      Function info:
; CHECK-NEXT: codeLenInByte = 12
; CHECK-NEXT: NumSgprs: 36
; CHECK-NEXT: NumVgprs: 1
; CHECK-NEXT: NumAgprs: 0
; CHECK-NEXT: TotalNumVgprs: 1
; CHECK-NEXT: ScratchSize: 0
; CHECK-NEXT: MemoryBound: 0
define internal i32 @addr_taken() {
  ret i32 0
}

; CHECK-LABEL: non_local
; CHECK:      Function info:
; CHECK-NEXT: codeLenInByte = 12
; CHECK-NEXT: NumSgprs: 36
; CHECK-NEXT: NumVgprs: 1
; CHECK-NEXT: NumAgprs: 0
; CHECK-NEXT: TotalNumVgprs: 1
; CHECK-NEXT: ScratchSize: 0
; CHECK-NEXT: MemoryBound: 0
define i32 @non_local() {
  ret i32 0
}

; CHECK-LABEL: take_addr
; CHECK:      Function info:
; CHECK-NEXT: codeLenInByte = 60
; CHECK-NEXT: NumSgprs: 37
; CHECK-NEXT: NumVgprs: 1
; CHECK-NEXT: NumAgprs: 0
; CHECK-NEXT: TotalNumVgprs: 1
; CHECK-NEXT: ScratchSize: 16
; CHECK-NEXT: MemoryBound: 0
define void @take_addr() {
  %addr_loc = alloca ptr, addrspace(5)
  store ptr @addr_taken, ptr addrspace(5) %addr_loc
  ret void
}
