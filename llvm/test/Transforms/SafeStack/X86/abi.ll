; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s --check-prefix=TLS
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s --check-prefix=TLS
; RUN: opt -safe-stack -S -mtriple=i686-linux-android < %s -o - | FileCheck %s --check-prefix=DIRECT-TLS32
; RUN: opt -safe-stack -S -mtriple=x86_64-linux-android < %s -o - | FileCheck %s --check-prefix=DIRECT-TLS64


define void @foo() nounwind uwtable safestack {
entry:
; TLS: %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
; TLS: %[[USST:.*]] = getelementptr i8, i8* %[[USP]], i32 -16
; TLS: store i8* %[[USST]], i8** @__safestack_unsafe_stack_ptr

; DIRECT-TLS32: %[[USP:.*]] = load i8*, i8* addrspace(256)* inttoptr (i32 36 to i8* addrspace(256)*)
; DIRECT-TLS32: %[[USST:.*]] = getelementptr i8, i8* %[[USP]], i32 -16
; DIRECT-TLS32: store i8* %[[USST]], i8* addrspace(256)* inttoptr (i32 36 to i8* addrspace(256)*)

; DIRECT-TLS64: %[[USP:.*]] = load i8*, i8* addrspace(257)* inttoptr (i32 72 to i8* addrspace(257)*)
; DIRECT-TLS64: %[[USST:.*]] = getelementptr i8, i8* %[[USP]], i32 -16
; DIRECT-TLS64: store i8* %[[USST]], i8* addrspace(257)* inttoptr (i32 72 to i8* addrspace(257)*)

  %a = alloca i8, align 8
  call void @Capture(i8* %a)

; TLS: store i8* %[[USP]], i8** @__safestack_unsafe_stack_ptr
; DIRECT-TLS32: store i8* %[[USP]], i8* addrspace(256)* inttoptr (i32 36 to i8* addrspace(256)*)
; DIRECT-TLS64: store i8* %[[USP]], i8* addrspace(257)* inttoptr (i32 72 to i8* addrspace(257)*)
  ret void
}

declare void @Capture(i8*)
