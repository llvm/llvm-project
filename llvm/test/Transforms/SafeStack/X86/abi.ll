; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s --check-prefix=TLS
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s --check-prefix=TLS
; RUN: opt -safe-stack -S -mtriple=i686-linux-android < %s -o - | FileCheck %s --check-prefix=DIRECT-TLS32
; RUN: opt -safe-stack -S -mtriple=x86_64-linux-android < %s -o - | FileCheck %s --check-prefix=DIRECT-TLS64


define void @foo() nounwind uwtable safestack {
entry:
; TLS: %[[USP:.*]] = load ptr, ptr @__safestack_unsafe_stack_ptr
; TLS: %[[USST:.*]] = getelementptr i8, ptr %[[USP]], i32 -16
; TLS: store ptr %[[USST]], ptr @__safestack_unsafe_stack_ptr

; DIRECT-TLS32: %[[USP:.*]] = load ptr, ptr addrspace(256) inttoptr (i32 36 to ptr addrspace(256))
; DIRECT-TLS32: %[[USST:.*]] = getelementptr i8, ptr %[[USP]], i32 -16
; DIRECT-TLS32: store ptr %[[USST]], ptr addrspace(256) inttoptr (i32 36 to ptr addrspace(256))

; DIRECT-TLS64: %[[USP:.*]] = load ptr, ptr addrspace(257) inttoptr (i32 72 to ptr addrspace(257))
; DIRECT-TLS64: %[[USST:.*]] = getelementptr i8, ptr %[[USP]], i32 -16
; DIRECT-TLS64: store ptr %[[USST]], ptr addrspace(257) inttoptr (i32 72 to ptr addrspace(257))

  %a = alloca i8, align 8
  call void @Capture(ptr %a)

; TLS: store ptr %[[USP]], ptr @__safestack_unsafe_stack_ptr
; DIRECT-TLS32: store ptr %[[USP]], ptr addrspace(256) inttoptr (i32 36 to ptr addrspace(256))
; DIRECT-TLS64: store ptr %[[USP]], ptr addrspace(257) inttoptr (i32 72 to ptr addrspace(257))
  ret void
}

declare void @Capture(ptr)
