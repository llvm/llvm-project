; RUN: opt < %s -passes=deadargelim -S | FileCheck %s
; PR36867

; CHECK-LABEL: @MagickMallocAligned
; CHECK-NOT: allocsize
define internal i64 @MagickMallocAligned(i64 %DEADARG1, i64 %s) allocsize(1) {
        ret i64 %s
}

define i64 @NeedsArg(i64 %s) {
	%c = call i64 @MagickMallocAligned(i64 0, i64 %s)
	ret i64 %c
}

; CHECK-LABEL: define i64 @Test2(
; CHECK-NEXT: %c = call i64 @MagickMallocAligned(i64 %s){{$}}
define i64 @Test2(i64 %s) {
	%c = call i64 @MagickMallocAligned(i64 0, i64 %s) allocsize(1)
	ret i64 %c
}

; An allocator keeps its allocsize argument and attribute, even though the
; argument is unused in the body.
; CHECK-LABEL: define internal ptr @alloc_unused_size(i64 %s) #0
define internal ptr @alloc_unused_size(i64 %s) allocsize(0) allockind("alloc,uninitialized") "alloc-family"="test" {
        ret ptr null
}

; Callers must keep passing the size, also to an allocator whose signature
; cannot change.
; CHECK-LABEL: define ptr @ext_alloc_unused_size(i64 %s) #0
define ptr @ext_alloc_unused_size(i64 %s) allocsize(0) allockind("alloc,uninitialized") "alloc-family"="test" {
        ret ptr null
}

; CHECK-LABEL: define ptr @CallsAllocators(
; CHECK: call ptr @alloc_unused_size(i64 8) #1
; CHECK: call ptr @ext_alloc_unused_size(i64 16) #1
define ptr @CallsAllocators() {
	%p = call ptr @alloc_unused_size(i64 8) allocsize(0)
	%q = call ptr @ext_alloc_unused_size(i64 16) allocsize(0)
	ret ptr %p
}

; CHECK: attributes #0 = { allockind("alloc,uninitialized") allocsize(0) "alloc-family"="test" }
; CHECK: attributes #1 = { allocsize(0) }
