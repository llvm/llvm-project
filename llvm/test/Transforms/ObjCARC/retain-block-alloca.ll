; RUN: opt -S -objc-arc < %s | FileCheck %s
; rdar://10209613

%0 = type opaque
%struct.__block_descriptor = type { i64, i64 }

@_NSConcreteStackBlock = external global i8*
@__block_descriptor_tmp = external hidden constant { i64, i64, i8*, i8*, i8*, i8* }
@"\01L_OBJC_SELECTOR_REFERENCES_" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"

; CHECK: define void @test(
; CHECK: %3 = call i8* @objc_retainBlock(i8* %2) nounwind
; CHECK: @objc_msgSend
; CHECK-NEXT: @objc_release(i8* %3)
define void @test(%0* %array) uwtable {
entry:
  %block = alloca <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>, align 8
  %0 = bitcast %0* %array to i8*
  %1 = tail call i8* @objc_retain(i8* %0) nounwind
  %block.isa = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block, i64 0, i32 0
  store i8* bitcast (i8** @_NSConcreteStackBlock to i8*), i8** %block.isa, align 8
  %block.flags = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block, i64 0, i32 1
  store i32 1107296256, i32* %block.flags, align 8
  %block.reserved = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block, i64 0, i32 2
  store i32 0, i32* %block.reserved, align 4
  %block.invoke = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block, i64 0, i32 3
  store i8* bitcast (void (i8*)* @__test_block_invoke_0 to i8*), i8** %block.invoke, align 8
  %block.descriptor = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block, i64 0, i32 4
  store %struct.__block_descriptor* bitcast ({ i64, i64, i8*, i8*, i8*, i8* }* @__block_descriptor_tmp to %struct.__block_descriptor*), %struct.__block_descriptor** %block.descriptor, align 8
  %block.captured = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block, i64 0, i32 5
  store %0* %array, %0** %block.captured, align 8
  %2 = bitcast <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block to i8*
  %3 = call i8* @objc_retainBlock(i8* %2) nounwind
  %tmp2 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i8*)*)(i8* %0, i8* %tmp2, i8* %3)
  call void @objc_release(i8* %3) nounwind
  %strongdestroy = load %0** %block.captured, align 8
  %4 = bitcast %0* %strongdestroy to i8*
  call void @objc_release(i8* %4) nounwind, !clang.imprecise_release !0
  ret void
}

; Same as test, but the objc_retainBlock has a clang.arc.copy_on_escape
; tag so it's safe to delete.

; CHECK: define void @test_with_COE(
; CHECK-NOT: @objc_retainBlock
; CHECK: @objc_msgSend
; CHECK: @objc_release
; CHECK-NOT: @objc_release
; CHECK: }
define void @test_with_COE(%0* %array) uwtable {
entry:
  %block = alloca <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>, align 8
  %0 = bitcast %0* %array to i8*
  %1 = tail call i8* @objc_retain(i8* %0) nounwind
  %block.isa = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block, i64 0, i32 0
  store i8* bitcast (i8** @_NSConcreteStackBlock to i8*), i8** %block.isa, align 8
  %block.flags = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block, i64 0, i32 1
  store i32 1107296256, i32* %block.flags, align 8
  %block.reserved = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block, i64 0, i32 2
  store i32 0, i32* %block.reserved, align 4
  %block.invoke = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block, i64 0, i32 3
  store i8* bitcast (void (i8*)* @__test_block_invoke_0 to i8*), i8** %block.invoke, align 8
  %block.descriptor = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block, i64 0, i32 4
  store %struct.__block_descriptor* bitcast ({ i64, i64, i8*, i8*, i8*, i8* }* @__block_descriptor_tmp to %struct.__block_descriptor*), %struct.__block_descriptor** %block.descriptor, align 8
  %block.captured = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block, i64 0, i32 5
  store %0* %array, %0** %block.captured, align 8
  %2 = bitcast <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block to i8*
  %3 = call i8* @objc_retainBlock(i8* %2) nounwind, !clang.arc.copy_on_escape !0
  %tmp2 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i8*)*)(i8* %0, i8* %tmp2, i8* %3)
  call void @objc_release(i8* %3) nounwind
  %strongdestroy = load %0** %block.captured, align 8
  %4 = bitcast %0* %strongdestroy to i8*
  call void @objc_release(i8* %4) nounwind, !clang.imprecise_release !0
  ret void
}

declare i8* @objc_retain(i8*)

declare void @__test_block_invoke_0(i8* nocapture) uwtable

declare i8* @objc_retainBlock(i8*)

declare i8* @objc_msgSend(i8*, i8*, ...) nonlazybind

declare void @objc_release(i8*)

!0 = metadata !{}
