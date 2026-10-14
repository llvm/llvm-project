; RUN: llc -mtriple=hexagon < %s | FileCheck %s

;; Verify that the SafeStack pass can locate __safestack_unsafe_stack_ptr
;; for Hexagon (added via DefaultSafeStackGlobals in HexagonSystemLibrary).

define void @test_safestack() safestack {
entry:
  %x = alloca i32, align 4
  call void @capture(ptr nonnull %x)
  ret void
}

declare void @capture(ptr)

; CHECK-LABEL: test_safestack:
; CHECK: __safestack_unsafe_stack_ptr
