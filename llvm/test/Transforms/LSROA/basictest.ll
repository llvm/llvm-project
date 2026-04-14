; RUN: opt < %s -passes='lsroa' -S -debug | FileCheck %s --check-prefixes=CHECK

declare void @llvm.lifetime.start.p0(ptr nocapture)
declare void @llvm.lifetime.end.p0(ptr nocapture)
declare ptr @llvm.structured.alloca.p0()
declare ptr @llvm.structured.gep.p0(ptr, ...)

define i32 @test_simple_scalar() {
; CHECK-LABEL: @test_simple_scalar(
; CHECK-NEXT:  entry:
entry:
  %tmp = call elementtype(i32) ptr @llvm.structured.alloca.p0()
  call void @llvm.lifetime.start.p0(ptr %tmp)
  store i32 0, ptr %tmp
  %res = load i32, ptr %tmp
  call void @llvm.lifetime.end.p0(ptr %tmp)
  ret i32 %res
; CHECK:  %tmp = call elementtype(i32) ptr @llvm.structured.alloca.p0()
; CHECK:  call void @llvm.lifetime.start.p0(ptr %tmp)
; CHECK:  store i32 0, ptr %tmp
; CHECK:  %res = load i32, ptr %tmp
; CHECK:  call void @llvm.lifetime.end.p0(ptr %tmp)
}

define i32 @test_simple_struct_entire_write_read() {
; CHECK-LABEL: @test_simple_struct_entire_write_read(
; CHECK-NEXT:  entry:
entry:
  %tmp = call elementtype({ i32, i32 }) ptr @llvm.structured.alloca.p0()
  %ptr0 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype({ i32, i32 }) %tmp, i32 0)
  %ptr1 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype({ i32, i32 }) %tmp, i32 1)

  call void @llvm.lifetime.start.p0(ptr %tmp)
  store i32 0, ptr %ptr0
  store i32 1, ptr %ptr1
  %a = load i32, ptr %ptr0
  %b = load i32, ptr %ptr1
  call void @llvm.lifetime.end.p0(ptr %tmp)

  %res = add i32 %a, %b
  ret i32 %res

; CHECK-NEXT:  %[[#a:]] = call elementtype(i32) ptr @llvm.structured.alloca.p0()
; CHECK-NEXT:  %[[#b:]] = call elementtype(i32) ptr @llvm.structured.alloca.p0()

; CHECK-NEXT:  call void @llvm.lifetime.start.p0(ptr %[[#a]])
; CHECK-NEXT:  call void @llvm.lifetime.start.p0(ptr %[[#b]])
; CHECK-NEXT:  store i32 0, ptr %[[#a]]
; CHECK-NEXT:  store i32 1, ptr %[[#b]]
; CHECK-NEXT:  %a = load i32, ptr %[[#a]]
; CHECK-NEXT:  %b = load i32, ptr %[[#b]]
; CHECK-NEXT:  call void @llvm.lifetime.end.p0(ptr %[[#a]])
; CHECK-NEXT:  call void @llvm.lifetime.end.p0(ptr %[[#b]])
}

define i32 @test_simple_struct_aliasing() {
; CHECK-LABEL: @test_simple_struct_aliasing(
; CHECK-NEXT:  entry:
entry:
  %tmp = call elementtype({ i32, i32 }) ptr @llvm.structured.alloca.p0()
  %ptr0 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype({ i32, i32 }) %tmp, i32 0)
  %ptr1 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype({ i32, i32 }) %tmp, i32 0)

  call void @llvm.lifetime.start.p0(ptr %tmp)
  store i32 0, ptr %ptr0
  store i32 1, ptr %ptr1
  %a = load i32, ptr %ptr0
  %b = load i32, ptr %ptr1
  call void @llvm.lifetime.end.p0(ptr %tmp)

  %res = add i32 %a, %b
  ret i32 %res

; CHECK-NEXT:  %[[#a:]] = call elementtype(i32) ptr @llvm.structured.alloca.p0()

; CHECK-NEXT:  call void @llvm.lifetime.start.p0(ptr %[[#a]])
; CHECK-NEXT:  store i32 0, ptr %[[#a]]
; CHECK-NEXT:  store i32 1, ptr %[[#a]]
; CHECK-NEXT:  %a = load i32, ptr %[[#a]]
; CHECK-NEXT:  %b = load i32, ptr %[[#a]]
; CHECK-NEXT:  call void @llvm.lifetime.end.p0(ptr %[[#a]])
}

define i32 @test_simple_struct_partial_write_read() {
; CHECK-LABEL: @test_simple_struct_partial_write_read(
; CHECK-NEXT:  entry:
entry:
  %tmp = call elementtype({ i32, i32 }) ptr @llvm.structured.alloca.p0()
  call void @llvm.lifetime.start.p0(ptr %tmp)
  %ptr0 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype({ i32, i32 }) %tmp, i32 0)
  store i32 0, ptr %ptr0
  %a = load i32, ptr %ptr0
  call void @llvm.lifetime.end.p0(ptr %tmp)

; CHECK-NEXT:  %[[#a:]] = call elementtype(i32) ptr @llvm.structured.alloca.p0()
; CHECK-NEXT:  call void @llvm.lifetime.start.p0(ptr %[[#a]])
; CHECK-NEXT:  store i32 0, ptr %[[#a]]
; CHECK-NEXT:  %a = load i32, ptr %[[#a]]
; CHECK-NEXT:  call void @llvm.lifetime.end.p0(ptr %[[#a]])

  ret i32 %a
}

define i32 @test_struct_use_across_lifetime() {
; CHECK-LABEL: @test_struct_use_across_lifetime(
; CHECK-NEXT:  entry:
entry:
  %tmp = call elementtype({ i32, i32 }) ptr @llvm.structured.alloca.p0()
  %ptr0 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype({ i32, i32 }) %tmp, i32 0)
  %ptr1 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype({ i32, i32 }) %tmp, i32 1)
; CHECK-NEXT:  %[[#a:]] = call elementtype(i32) ptr @llvm.structured.alloca.p0()
; CHECK-NEXT:  %[[#b:]] = call elementtype(i32) ptr @llvm.structured.alloca.p0()

  call void @llvm.lifetime.start.p0(ptr %tmp)
  store i32 0, ptr %ptr0
  %a = load i32, ptr %ptr0
  call void @llvm.lifetime.end.p0(ptr %tmp)
; CHECK-NEXT:  call void @llvm.lifetime.start.p0(ptr %[[#a]])
; CHECK-NEXT:  call void @llvm.lifetime.start.p0(ptr %[[#b]])
; CHECK-NEXT:  store i32 0, ptr %[[#a]]
; CHECK-NEXT:  %a = load i32, ptr %[[#a]]
; CHECK-NEXT:  call void @llvm.lifetime.end.p0(ptr %[[#a]])
; CHECK-NEXT:  call void @llvm.lifetime.end.p0(ptr %[[#b]])

  call void @llvm.lifetime.start.p0(ptr %tmp)
  store i32 0, ptr %ptr1
  %b = load i32, ptr %ptr1
  call void @llvm.lifetime.end.p0(ptr %tmp)
; CHECK-NEXT:  call void @llvm.lifetime.start.p0(ptr %[[#a]])
; CHECK-NEXT:  call void @llvm.lifetime.start.p0(ptr %[[#b]])
; CHECK-NEXT:  store i32 0, ptr %[[#b]]
; CHECK-NEXT:  %b = load i32, ptr %[[#b]]
; CHECK-NEXT:  call void @llvm.lifetime.end.p0(ptr %[[#a]])
; CHECK-NEXT:  call void @llvm.lifetime.end.p0(ptr %[[#b]])

  %c = add i32 %b, %a
  ret i32 %c
}

define i32 @test_partial_use_phi_node(i1 %cond) {
; CHECK-LABEL: @test_partial_use_phi_node(
; CHECK-NEXT:  entry:
entry:
  %tmp = call elementtype({ i32, i32 }) ptr @llvm.structured.alloca.p0()
; CHECK-NEXT:  %[[#a:]] = call elementtype(i32) ptr @llvm.structured.alloca.p0()
; CHECK-NEXT:  %[[#b:]] = call elementtype(i32) ptr @llvm.structured.alloca.p0()
  %ptr0 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype({ i32, i32 }) %tmp, i32 0)
  %ptr1 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype({ i32, i32 }) %tmp, i32 1)
  br i1 %cond, label %l1, label %l2

l1:
  call void @llvm.lifetime.start.p0(ptr %tmp)
  store i32 0, ptr %ptr0
  br label %l3
; CHECK:       l1: ; preds = %entry
; CHECK-NEXT:     call void @llvm.lifetime.start.p0(ptr %[[#a]])
; CHECK-NEXT:     call void @llvm.lifetime.start.p0(ptr %[[#b]])
; CHECK-NEXT:     store i32 0, ptr %[[#a]]
; CHECK-NEXT:     br label %l3

l2:
  call void @llvm.lifetime.start.p0(ptr %tmp)
  store i32 1, ptr %ptr1
  br label %l3
; CHECK:       l2: ; preds = %entry
; CHECK-NEXT:     call void @llvm.lifetime.start.p0(ptr %[[#a]])
; CHECK-NEXT:     call void @llvm.lifetime.start.p0(ptr %[[#b]])
; CHECK-NEXT:     store i32 1, ptr %[[#b]]
; CHECK-NEXT:     br label %l3

l3:
  %ptr = phi ptr [ %ptr0, %l1 ], [ %ptr1, %l2 ]
  %a = load i32, ptr %ptr
  call void @llvm.lifetime.end.p0(ptr %tmp)
  br label %exit
; CHECK:       l3: ; preds = %l2, %l1
; CHECK-NEXT:     %ptr = phi ptr [ %[[#a]], %l1 ], [ %[[#b]], %l2 ]
; CHECK-NEXT:     %a = load i32, ptr %ptr
; CHECK-NEXT:     call void @llvm.lifetime.end.p0(ptr %[[#a]])
; CHECK-NEXT:     call void @llvm.lifetime.end.p0(ptr %[[#b]])
; CHECK-NEXT:     br label %exit

exit:
  ret i32 %a
}
