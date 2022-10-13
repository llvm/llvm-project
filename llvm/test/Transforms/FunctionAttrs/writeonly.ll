; RUN: opt < %s -function-attrs         -S | FileCheck %s
; RUN: opt < %s -passes=function-attrs -S | FileCheck %s

; CHECK: define void @nouses-argworn-funrn(ptr nocapture readnone %.aaa) #0 {
define void @nouses-argworn-funrn(ptr writeonly %.aaa) {
nouses-argworn-funrn_entry:
  ret void
}

; CHECK: define void @nouses-argworn-funro(ptr nocapture readnone %.aaa, ptr nocapture readonly %.bbb) #1 {
define void @nouses-argworn-funro(ptr writeonly %.aaa, ptr %.bbb) {
nouses-argworn-funro_entry:
  %val = load i32 , ptr %.bbb
  ret void
}

%_type_of_d-ccc = type <{ ptr, i8, i8, i8, i8 }>

@d-ccc = internal global %_type_of_d-ccc <{ ptr null, i8 1, i8 13, i8 0, i8 -127 }>, align 8

; CHECK: define void @nouses-argworn-funwo(ptr nocapture readnone %.aaa) #2 {
define void @nouses-argworn-funwo(ptr writeonly %.aaa) {
nouses-argworn-funwo_entry:
  store i8 0, ptr getelementptr inbounds (%_type_of_d-ccc, ptr @d-ccc, i32 0, i32 3)
  ret void
}

; CHECK: define void @test_store(ptr nocapture writeonly %p)
define void @test_store(ptr %p) {
  store i8 0, ptr %p
  ret void
}

@G = external global ptr
; CHECK: define i8 @test_store_capture(ptr %p)
define i8 @test_store_capture(ptr %p) {
  store ptr %p, ptr @G
  %p2 = load ptr, ptr @G
  %v = load i8, ptr %p2
  ret i8 %v
}

; CHECK: define void @test_addressing(ptr nocapture writeonly %p)
define void @test_addressing(ptr %p) {
  %gep = getelementptr i8, ptr %p, i64 8
  store i32 0, ptr %gep
  ret void
}

; CHECK: define void @test_readwrite(ptr nocapture %p)
define void @test_readwrite(ptr %p) {
  %v = load i8, ptr %p
  store i8 %v, ptr %p
  ret void
}

; CHECK: define void @test_volatile(ptr %p)
define void @test_volatile(ptr %p) {
  store volatile i8 0, ptr %p
  ret void
}

; CHECK: define void @test_atomicrmw(ptr nocapture %p)
define void @test_atomicrmw(ptr %p) {
  atomicrmw add ptr %p, i8 0  seq_cst
  ret void
}


declare void @direct1_callee(ptr %p)

; CHECK: define void @direct1(ptr %p)
define void @direct1(ptr %p) {
  call void @direct1_callee(ptr %p)
  ret void
}

declare void @direct2_callee(ptr %p) writeonly

; writeonly w/o nocapture is not enough
; CHECK: define void @direct2(ptr %p)
define void @direct2(ptr %p) {
  call void @direct2_callee(ptr %p)
  ; read back from global, read through pointer...
  ret void
}

; CHECK: define void @direct2b(ptr nocapture writeonly %p)
define void @direct2b(ptr %p) {
  call void @direct2_callee(ptr nocapture %p)
  ret void
}

declare void @direct3_callee(ptr nocapture writeonly %p)

; CHECK: define void @direct3(ptr nocapture writeonly %p)
define void @direct3(ptr %p) {
  call void @direct3_callee(ptr %p)
  ret void
}

; CHECK: define void @direct3b(ptr %p)
define void @direct3b(ptr %p) {
  call void @direct3_callee(ptr %p) ["may-read-and-capture"(ptr %p)]
  ret void
}

; CHECK: define void @fptr_test1(ptr %p, ptr nocapture readonly %f)
define void @fptr_test1(ptr %p, ptr %f) {
  call void %f(ptr %p)
  ret void
}

; CHECK: define void @fptr_test2(ptr nocapture writeonly %p, ptr nocapture readonly %f)
define void @fptr_test2(ptr %p, ptr %f) {
  call void %f(ptr nocapture writeonly %p)
  ret void
}

; CHECK: define void @fptr_test3(ptr nocapture writeonly %p, ptr nocapture readonly %f)
define void @fptr_test3(ptr %p, ptr %f) {
  call void %f(ptr nocapture %p) writeonly
  ret void
}

; CHECK: attributes #0 = { {{.*}}readnone{{.*}} }
; CHECK: attributes #1 = { {{.*}}readonly{{.*}} }
; CHECK: attributes #2 = { {{.*}}writeonly{{.*}} }
