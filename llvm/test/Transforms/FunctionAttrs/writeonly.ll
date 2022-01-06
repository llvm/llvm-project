; RUN: opt < %s -function-attrs         -S | FileCheck %s
; RUN: opt < %s -passes=function-attrs -S | FileCheck %s

; CHECK: define void @nouses-argworn-funrn(i32* nocapture readnone %.aaa) #0 {
define void @nouses-argworn-funrn(i32* writeonly %.aaa) {
nouses-argworn-funrn_entry:
  ret void
}

; CHECK: define void @nouses-argworn-funro(i32* nocapture readnone %.aaa, i32* nocapture readonly %.bbb) #1 {
define void @nouses-argworn-funro(i32* writeonly %.aaa, i32* %.bbb) {
nouses-argworn-funro_entry:
  %val = load i32 , i32* %.bbb
  ret void
}

%_type_of_d-ccc = type <{ i8*, i8, i8, i8, i8 }>

@d-ccc = internal global %_type_of_d-ccc <{ i8* null, i8 1, i8 13, i8 0, i8 -127 }>, align 8

; CHECK: define void @nouses-argworn-funwo(i32* nocapture readnone %.aaa) #2 {
define void @nouses-argworn-funwo(i32* writeonly %.aaa) {
nouses-argworn-funwo_entry:
  store i8 0, i8* getelementptr inbounds (%_type_of_d-ccc, %_type_of_d-ccc* @d-ccc, i32 0, i32 3)
  ret void
}

; CHECK: define void @test_store(i8* nocapture writeonly %p)
define void @test_store(i8* %p) {
  store i8 0, i8* %p
  ret void
}

@G = external global i8*
; CHECK: define i8 @test_store_capture(i8* %p)
define i8 @test_store_capture(i8* %p) {
  store i8* %p, i8** @G
  %p2 = load i8*, i8** @G
  %v = load i8, i8* %p2
  ret i8 %v
}

; CHECK: define void @test_addressing(i8* nocapture writeonly %p)
define void @test_addressing(i8* %p) {
  %gep = getelementptr i8, i8* %p, i64 8
  %bitcast = bitcast i8* %gep to i32*
  store i32 0, i32* %bitcast
  ret void
}

; CHECK: define void @test_readwrite(i8* nocapture %p)
define void @test_readwrite(i8* %p) {
  %v = load i8, i8* %p
  store i8 %v, i8* %p
  ret void
}

; CHECK: define void @test_volatile(i8* %p)
define void @test_volatile(i8* %p) {
  store volatile i8 0, i8* %p
  ret void
}

; CHECK: define void @test_atomicrmw(i8* nocapture %p)
define void @test_atomicrmw(i8* %p) {
  atomicrmw add i8* %p, i8 0  seq_cst
  ret void
}


declare void @direct1_callee(i8* %p)

; CHECK: define void @direct1(i8* %p)
define void @direct1(i8* %p) {
  call void @direct1_callee(i8* %p)
  ret void
}

declare void @direct2_callee(i8* %p) writeonly

; writeonly w/o nocapture is not enough
; CHECK: define void @direct2(i8* %p)
define void @direct2(i8* %p) {
  call void @direct2_callee(i8* %p)
  ; read back from global, read through pointer...
  ret void
}

; CHECK: define void @direct2b(i8* nocapture writeonly %p)
define void @direct2b(i8* %p) {
  call void @direct2_callee(i8* nocapture %p)
  ret void
}

declare void @direct3_callee(i8* nocapture writeonly %p)

; CHECK: define void @direct3(i8* nocapture writeonly %p)
define void @direct3(i8* %p) {
  call void @direct3_callee(i8* %p)
  ret void
}

; CHECK: define void @fptr_test1(i8* %p, void (i8*)* nocapture readonly %f)
define void @fptr_test1(i8* %p, void (i8*)* %f) {
  call void %f(i8* %p)
  ret void
}

; CHECK: define void @fptr_test2(i8* nocapture writeonly %p, void (i8*)* nocapture readonly %f)
define void @fptr_test2(i8* %p, void (i8*)* %f) {
  call void %f(i8* nocapture writeonly %p)
  ret void
}

; CHECK: define void @fptr_test3(i8* nocapture writeonly %p, void (i8*)* nocapture readonly %f)
define void @fptr_test3(i8* %p, void (i8*)* %f) {
  call void %f(i8* nocapture %p) writeonly
  ret void
}

; CHECK: attributes #0 = { {{.*}}readnone{{.*}} }
; CHECK: attributes #1 = { {{.*}}readonly{{.*}} }
; CHECK: attributes #2 = { {{.*}}writeonly{{.*}} }
