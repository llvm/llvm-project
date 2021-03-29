; RUN: opt -gvn-hoist -S < %s | FileCheck %s

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

%struct.S = type { i8* }

declare void @f(<{ %struct.S }>* inalloca(<{ %struct.S }>))


; Check that we don't clone the %x alloca and insert it in the live range of
; %argmem, which would break the inalloca contract.
;
; CHECK-LABEL: @test
; CHECK: alloca i8
; CHECK: stacksave
; CHECK: alloca inalloca
; CHECK-NOT: alloca i8

; Check that store instructions are hoisted.
; CHECK: store i8
; CHECK-NOT: store i8
; CHECK: stackrestore

define void @test(i1 %b) {
entry:
  %x = alloca i8
  %inalloca.save = call i8* @llvm.stacksave()
  %argmem = alloca inalloca <{ %struct.S }>, align 4
  %0 = getelementptr inbounds <{ %struct.S }>, <{ %struct.S }>* %argmem, i32 0, i32 0
  br i1 %b, label %true, label %false

true:
  %p = getelementptr inbounds %struct.S, %struct.S* %0, i32 0, i32 0
  store i8* %x, i8** %p, align 4
  br label %exit

false:
  %p2 = getelementptr inbounds %struct.S, %struct.S* %0, i32 0, i32 0
  store i8* %x, i8** %p2, align 4
  br label %exit

exit:
  call void @f(<{ %struct.S }>* inalloca(<{ %struct.S }>) %argmem)
  call void @llvm.stackrestore(i8* %inalloca.save)
  ret void
}

declare i8* @llvm.stacksave()
declare void @llvm.stackrestore(i8*)
