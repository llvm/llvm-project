; RUN: opt < %s -instcombine -S -data-layout="E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128" | FileCheck %s -check-prefix=CHECK -check-prefix=ALL
; RUN: opt < %s -instcombine -S -data-layout="E-p:32:32:32-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128" | FileCheck %s -check-prefix=P32 -check-prefix=ALL
; RUN: opt < %s -instcombine -S | FileCheck %s -check-prefix=NODL -check-prefix=ALL


declare void @use(...)

@int = global i32 zeroinitializer

; Zero byte allocas should be merged if they can't be deleted.
; CHECK-LABEL: @test(
; CHECK: alloca
; CHECK-NOT: alloca
define void @test() {
        %X = alloca [0 x i32]           ; <[0 x i32]*> [#uses=1]
        call void (...) @use( [0 x i32]* %X )
        %Y = alloca i32, i32 0          ; <i32*> [#uses=1]
        call void (...) @use( i32* %Y )
        %Z = alloca {  }                ; <{  }*> [#uses=1]
        call void (...) @use( {  }* %Z )
        %size = load i32, i32* @int
        %A = alloca {{}}, i32 %size
        call void (...) @use( {{}}* %A )
        ret void
}

; Zero byte allocas should be deleted.
; CHECK-LABEL: @test2(
; CHECK-NOT: alloca
define void @test2() {
        %A = alloca i32         ; <i32*> [#uses=1]
        store i32 123, i32* %A
        ret void
}

; Zero byte allocas should be deleted.
; CHECK-LABEL: @test3(
; CHECK-NOT: alloca
define void @test3() {
        %A = alloca { i32 }             ; <{ i32 }*> [#uses=1]
        %B = getelementptr { i32 }, { i32 }* %A, i32 0, i32 0            ; <i32*> [#uses=1]
        store i32 123, i32* %B
        ret void
}

; CHECK-LABEL: @test4(
; CHECK: = zext i32 %n to i64
; CHECK: %A = alloca i32, i64 %
define i32* @test4(i32 %n) {
  %A = alloca i32, i32 %n
  ret i32* %A
}

; Allocas which are only used by GEPs, bitcasts, addrspacecasts, and stores
; (transitively) should be deleted.
define void @test5() {
; CHECK-LABEL: @test5(
; CHECK-NOT: alloca
; CHECK-NOT: store
; CHECK: ret

entry:
  %a = alloca { i32 }
  %b = alloca i32*
  %c = alloca i32
  %a.1 = getelementptr { i32 }, { i32 }* %a, i32 0, i32 0
  store i32 123, i32* %a.1
  store i32* %a.1, i32** %b
  %b.1 = bitcast i32** %b to i32*
  store i32 123, i32* %b.1
  %a.2 = getelementptr { i32 }, { i32 }* %a, i32 0, i32 0
  store atomic i32 2, i32* %a.2 unordered, align 4
  %a.3 = getelementptr { i32 }, { i32 }* %a, i32 0, i32 0
  store atomic i32 3, i32* %a.3 release, align 4
  %a.4 = getelementptr { i32 }, { i32 }* %a, i32 0, i32 0
  store atomic i32 4, i32* %a.4 seq_cst, align 4
  %c.1 = addrspacecast i32* %c to i32 addrspace(1)*
  store i32 123, i32 addrspace(1)* %c.1
  ret void
}

declare void @f(i32* %p)

; Check that we don't delete allocas in some erroneous cases.
define void @test6() {
; CHECK-LABEL: @test6(
; CHECK-NOT: ret
; CHECK: alloca
; CHECK-NEXT: alloca
; CHECK: ret

entry:
  %a = alloca { i32 }
  %b = alloca i32
  %a.1 = getelementptr { i32 }, { i32 }* %a, i32 0, i32 0
  store volatile i32 123, i32* %a.1
  tail call void @f(i32* %b)
  ret void
}

; PR14371
%opaque_type = type opaque
%real_type = type { { i32, i32* } }

@opaque_global = external constant %opaque_type, align 4

define void @test7() {
entry:
  %0 = alloca %real_type, align 4
  %1 = bitcast %real_type* %0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %1, i8* bitcast (%opaque_type* @opaque_global to i8*), i32 8, i32 1, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind


; Check that the GEP indices use the pointer size, or 64 if unknown
define void @test8() {
; CHECK-LABEL: @test8(
; CHECK: alloca [100 x i32]
; CHECK: getelementptr inbounds [100 x i32], [100 x i32]* %x1, i64 0, i64 0

; P32-LABEL: @test8(
; P32: alloca [100 x i32]
; P32: getelementptr inbounds [100 x i32], [100 x i32]* %x1, i32 0, i32 0

; NODL-LABEL: @test8(
; NODL: alloca [100 x i32]
; NODL: getelementptr inbounds [100 x i32], [100 x i32]* %x1, i64 0, i64 0
  %x = alloca i32, i32 100
  call void (...) @use(i32* %x)
  ret void
}

; PR19569
%struct_type = type { i32, i32 }
declare void @test9_aux(<{ %struct_type }>* inalloca)
declare i8* @llvm.stacksave()
declare void @llvm.stackrestore(i8*)

define void @test9(%struct_type* %a) {
; CHECK-LABEL: @test9(
entry:
  %inalloca.save = call i8* @llvm.stacksave()
  %argmem = alloca inalloca <{ %struct_type }>
; CHECK: alloca inalloca i64, align 8
  %0 = getelementptr inbounds <{ %struct_type }>, <{ %struct_type }>* %argmem, i32 0, i32 0
  %1 = bitcast %struct_type* %0 to i8*
  %2 = bitcast %struct_type* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %1, i8* %2, i32 8, i32 4, i1 false)
  call void @test9_aux(<{ %struct_type }>* inalloca %argmem)
  call void @llvm.stackrestore(i8* %inalloca.save)
  ret void
}

define void @test10() {
entry:
; ALL-LABEL: @test10(
; ALL: %v32 = alloca i1, align 8
; ALL: %v64 = alloca i1, align 8
; ALL: %v33 = alloca i1, align 8
  %v32 = alloca i1, align 8
  %v64 = alloca i1, i64 1, align 8
  %v33 = alloca i1, i33 1, align 8
  call void (...) @use(i1* %v32, i1* %v64, i1* %v33)
  ret void
}

define void @test11() {
entry:
; ALL-LABEL: @test11(
; ALL: %y = alloca i32
; ALL: call void (...) @use(i32* nonnull @int) [ "blah"(i32* %y) ]
; ALL: ret void
  %y = alloca i32
  call void (...) @use(i32* nonnull @int) [ "blah"(i32* %y) ]
  ret void
}
